```bash
uv run python examples/models/diffusion_planner/diffusion_onnx_trt.py \
    --pretrained_model_path "./examples/models/diffusion_planner/models/ReCogDrive_Diffusion_Planner_2B_RL.ckpt" \
    --onnxFile ./examples/models/diffusion_planner/models/diffusion_planner.onnx \
    --planFile ./models/diffusion_planner_fp16.plan
```