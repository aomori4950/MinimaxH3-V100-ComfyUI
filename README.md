# MinimaxH3-V100-ComfyUI
Run Minimax H3 in FP16 on GPUs that do not support BF16 tensor cores in ComfyUI.

Setting	Result
ModelComputeDtype = fp16	black video (NaN in latent)
default (BF16 emulated - FP32)	correct output, but very slow

Tesla V100-SXM2 32GB, 0.4mp, 5s T2V
Config	s/it
BF16 emulated → FP32 fallback (default)	94
FP16 (ModelComputeDtype = fp16)	~14, but black video
This node	14, correct output.

~6.7× faster than default, with usable output.

This node also likely works on Turing GPUs (RTX 20 series), untested.

Put this node only right before your BasicGuider in the default workflow.
