```bash
uv run python examples/models/diffusion_planner/diffusion_onnx_trt.py \
    --pretrained_model_path "./examples/models/diffusion_planner/models/ReCogDrive_Diffusion_Planner_2B_RL.ckpt" \
    --onnxFile ./examples/models/diffusion_planner/models/diffusion_denoising_step.onnx \
    --planFile ./models/diffusion_denoising_step_fp16.plan
```

This exports the denoising-step network only. The diffusion sampler loop,
conditioning encoders, clipping, and trajectory denormalization stay in PyTorch.

```bash
uv run pytest -q tests/test_diffusion_denoising_step_trt.py
```
