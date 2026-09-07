import torch

# Keep on the fp32 rail: modulation/adaLN (pruned to lookup tables), embedders,
# time, rope/pos, final out, audio heads. Substring match on lowercased name.
SKIP = ("mod", "adaln", "embed", "time", "rope", "pos_", "pe_", "_in", "input_proj",
        "proj_out", "final", "last", "head", "audio_out", "lut", "curve")

_PROBE = {"left": 8, "gmax": 0.0, "over": 0}


def _is_norm(m):
    n = type(m).__name__
    return "RMSNorm" in n or "LayerNorm" in n


def _is_linear(m):
    return "Linear" in type(m).__name__ and hasattr(m, "weight")


def _is_attention(m):
    n = type(m).__name__
    return n in ("Attention", "SelfAttention", "JointAttention", "CrossAttention")


def _plain_bias(m):
    b = getattr(m, "bias", None)
    if b is None:
        return None
    try:
        if isinstance(b, torch.Tensor) and b.dtype in (
                torch.float16, torch.bfloat16, torch.float32, torch.float64):
            return b
    except Exception:
        pass
    return "exotic"


def _wrap_norm_fp32(m):
    if getattr(m, "_v100_fp32", False):
        return
    orig = m.forward
    def fwd(x, *a, **k):
        return orig(x.float(), *a, **k).to(x.dtype)
    m.forward = fwd
    m._v100_fp32 = True


def _wrap_gemm(m, name, target, debug):
    # Lazy: plain fp16 first; only layers that overflow latch onto bias-correct
    # activation scaling (out = s*(W*(x/s)) + b). Rail stays fp32.
    if getattr(m, "_v100_fp16", False):
        return
    orig = m.forward
    inv = 1.0 / float(target)
    scalable = _plain_bias(m) != "exotic"

    def fwd(x, *a, **k):
        if not getattr(m, "_needs_scale", False):
            o = orig(x.to(torch.float16), *a, **k)
            if torch.isfinite(o).all():
                if debug and _PROBE["left"] > 0:
                    _PROBE["left"] -= 1
                    try:
                        print(f"[h3 probe] {name}: out={o.dtype} absmax={float(o.detach().abs().max().item()):.1f}")
                    except Exception:
                        pass
                return o.to(x.dtype)
            if scalable:
                m._needs_scale = True
                _PROBE["over"] += 1
                if debug:
                    print(f"[h3 probe] OVERFLOW -> scaling enabled on {name}")
            else:
                return torch.nan_to_num(o, nan=0.0, posinf=65504.0,
                                        neginf=-65504.0).to(x.dtype)
        s = (x.detach().abs().amax() * inv).clamp_(min=1.0)
        b = getattr(m, "bias", None)   # read live: weights move device lazily
        if b is not None:
            m.bias = None
        try:
            o = orig((x / s).to(torch.float16)).float().mul_(s)
        finally:
            if b is not None:
                m.bias = b
        if b is not None:
            o = o + b.to(device=o.device, dtype=o.dtype)
        o = torch.nan_to_num(o, nan=0.0, posinf=65504.0, neginf=-65504.0)
        return o.to(x.dtype)

    m.forward = fwd
    m._v100_fp16 = True


def _wrap_attention(m):
    import torch.nn.functional as F
    if getattr(m, "_v100_attn", False):
        return
    real = m.forward
    def fwd(*a, **k):
        prev = F.scaled_dot_product_attention
        def sdpa16(q, kk, v, *aa, **kw):
            dt = q.dtype
            return prev(q.to(torch.float16), kk.to(torch.float16),
                        v.to(torch.float16), *aa, **kw).to(dt)
        F.scaled_dot_product_attention = sdpa16
        try:
            return real(*a, **k)
        finally:
            F.scaled_dot_product_attention = prev
    m.forward = fwd
    m._v100_attn = True


def _patch(dit, target, fp16_attn, debug):
    nn = ng = na = 0
    for name, m in dit.named_modules():
        low = name.lower()
        if _is_norm(m):
            _wrap_norm_fp32(m); nn += 1
        elif _is_linear(m) and not any(s in low for s in SKIP):
            _wrap_gemm(m, name, target, debug); ng += 1
        elif fp16_attn and _is_attention(m):
            _wrap_attention(m); na += 1
    print(f"[MiniMax-H3 V100] norms->fp32:{nn} | fp16 GEMMs:{ng} | fp16 attn:{na} | target={target}")


class MiniMaxH3V100Patch:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "fp16_gemm": ("BOOLEAN", {"default": True}),
            "fp16_attention": ("BOOLEAN", {"default": True}),
            "scale_target": ("INT", {"default": 32, "min": 8, "max": 512}),
            "debug": ("BOOLEAN", {"default": False}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "advanced/precision"
    TITLE = "MiniMax-H3 V100 Hybrid Precision"

    def patch(self, model, fp16_gemm, fp16_attention, scale_target, debug):
        try:
            torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
        except Exception:
            pass
        _PROBE["left"] = 8 if debug else 0
        _PROBE["over"] = 0
        m = model.clone()
        dit = m.model.diffusion_model
        if fp16_gemm:
            _patch(dit, scale_target, fp16_attention, debug)
        else:
            for _, mod in dit.named_modules():
                if _is_norm(mod):
                    _wrap_norm_fp32(mod)
        return (m,)


NODE_CLASS_MAPPINGS = {"MiniMaxH3V100Patch": MiniMaxH3V100Patch}
NODE_DISPLAY_NAME_MAPPINGS = {"MiniMaxH3V100Patch": "MiniMax-H3 V100 Hybrid Precision"}
