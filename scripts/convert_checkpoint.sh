uv run python src/models/vit_onnx_trt.py \
    --pretrained_model_path "owl10/ReCogDrive-VLM-2B" \
    --onnxFile ./models/DiffusionVLM/onnx/visual_encoder.onnx \
    --planFile ./models/DiffusionVLM/plan/visual_encoder_fp16.plan \
    --image_url ./models/demo.jpeg > logs/convert_checkpoint/vision_encoder.log 2>&1

uv run python src/models/convert_checkpoint.py \
    --model_dir "./models/Qwen" \
    --output_dir "./models/DiffusionVLM/safetensors/llm" \
    --dtype float16 > logs/convert_checkpoint/llm.log 2>&1

uv run trtllm-build \
    --checkpoint_dir "./models/DiffusionVLM/safetensors/llm" \
    --output_dir "./models/DiffusionVLM/plan/llm_fp16" \
    --gemm_plugin float16 \
    --max_batch_size 8  > logs/convert_checkpoint/trtllm.log 2>&1

uv run python src/models/export_mlp.py \
    --model-path "owl10/ReCogDrive-VLM-2B" \
    --save-dir ./models/DiffusionVLM/onnx/mlp  > logs/convert_checkpoint/mlp.log 2>&1

export PYTHONPATH=.:$PYTHONPATH
uv run python src/models/export_planner.py \
    --ckpt models/ReCogDrive_Diffusion_Planner_2B_RL.ckpt \
    --save-dir ./models/DiffusionVLM/onnx/planner  > logs/convert_checkpoint/planner.log 2>&1
