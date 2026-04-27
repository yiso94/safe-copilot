import torch
from torch.nn import functional as F
from transformers import AutoModel, BatchFeature

from models.action_head import DiffusionPlanner
from models.internvl_chat import InternVLChatTRT
from models.intern_vit import InternVisionModel
from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent


def test_vision_model():
    engine_path = "./models/vision_projector_bf16.plan"
    stream = torch.cuda.Stream().cuda_stream
    vision_model = InternVisionModel(engine_path, stream)
    pixel_values = torch.randn(1, 3, 448, 448).cuda().to(torch.bfloat16)

    internvl = (
        AutoModel.from_pretrained(
            "owl10/ReCogDrive-VLM-2B",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=True,
            device_map="cuda",
        )
        .eval()
        .cuda()
    )

    trt_output = vision_model.infer(pixel_values)
    with torch.inference_mode():
        reference_output = internvl.extract_feature(pixel_values)
        reference_output = reference_output
    assert trt_output.shape == reference_output.shape, f"Unexpected output shape: {trt_output.shape}"
    assert trt_output.dtype == reference_output.dtype, f"Expected dtype torch.bfloat16, but got {trt_output.dtype}"
    cos_sim = F.cosine_similarity(reference_output, trt_output, dim=-1)
    assert torch.mean(cos_sim) > 0.99, "Cosine similarity is too low, outputs may not be similar enough"


def test_qwen_model():
    if not torch.cuda.is_available():
        import pytest

        pytest.skip("CUDA is required for the Qwen TRT hidden-state test")

    prompt = "Describe the driving scene."
    qwen_engine_dir = "./models/qwen_bf16"
    runtime = InternVLChatTRT(qwen_engine_dir=qwen_engine_dir)
    try:
        trt_hidden_state = runtime.get_text_last_hidden_state(prompt).detach().float().cpu()
        tokenizer = runtime.tokenizer
        max_input_len = runtime.max_input_len
        use_padding = False if runtime.qwen_model.decoder.remove_input_padding else "max_length"
    finally:
        runtime.close()

    model = (
        AutoModel.from_pretrained(
            "owl10/ReCogDrive-VLM-2B",
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=True,
            device_map="cuda",
        )
        .eval()
        .cuda()
    )

    try:
        model_inputs = tokenizer(
            [prompt],
            return_tensors="pt",
            padding=use_padding,
            truncation=True,
            max_length=max_input_len,
        )
        with torch.inference_mode():
            reference_outputs = model.language_model(
                input_ids=model_inputs["input_ids"].to("cuda"),
                attention_mask=model_inputs["attention_mask"].to("cuda"),
                output_hidden_states=True,
                return_dict=True,
            )
            reference_hidden_state = reference_outputs.hidden_states[-1].detach().float().cpu()
    finally:
        del model
        torch.cuda.empty_cache()

    assert trt_hidden_state.ndim == 3
    assert trt_hidden_state.shape == reference_hidden_state.shape

    max_abs_diff = (trt_hidden_state - reference_hidden_state).abs().max().item()
    mean_abs_diff = (trt_hidden_state - reference_hidden_state).abs().mean().item()
    assert max_abs_diff < 2.5 and mean_abs_diff < 5e-2, (
        "Qwen TRT last hidden state is too far from owl10/ReCogDrive-VLM-2B. "
        f"max_abs_diff={max_abs_diff:.6f}, mean_abs_diff={mean_abs_diff:.6f}"
    )


def test_diffusion_planner():
    last_hidden_state = torch.randn(1, 2800, 1536)
    input_state = torch.randn(1, 20)
    history_trajectory_reshaped = torch.randn(1, 12)
    status_feature = torch.randn(1, 8)

    diffusion_engine_path = "./models/diffusion_planner_bf16.plan"
    stream = torch.cuda.Stream().cuda_stream
    diffusion_planner = DiffusionPlanner(diffusion_engine_path, stream)
    pred_traj_tft = diffusion_planner.infer(last_hidden_state, input_state, history_trajectory_reshaped, status_feature)

    checkpoint_path = "./examples/models/diffusion_planner/models/ReCogDrive_Diffusion_Planner_2B_RL.ckpt"
    agent = (
        ReCogDriveAgent(
            None,
            checkpoint_path=checkpoint_path,
            vlm_path="owl10/ReCogDrive-VLM-2B",
            cam_type="single",
            grpo=False,
            cache_hidden_state=False,
            vlm_type="internvl",
            dit_type="small",
            vlm_size="small",
            sampling_method="ddim",
        )
        .eval()
        .to(torch.float16)
        .cuda()
    )
    agent.initialize()
    action_inputs = BatchFeature(
        {
            "state": input_state.to(torch.float16).cuda(),
            "his_traj": history_trajectory_reshaped.to(torch.float16).cuda(),
            "status_feature": status_feature.to(torch.float16).cuda(),
        }
    )
    pred_traj_ckpt = agent.action_head.get_action(
        last_hidden_state.to(torch.float16).cuda(),
        action_inputs,
    )["pred_traj"].float()

    assert pred_traj_tft.dtype == torch.float32, f"Expected dtype torch.float32, but got {pred_traj_tft.dtype}"
    assert pred_traj_ckpt.dtype == torch.float32, f"Expected dtype torch.float32, but got {pred_traj_ckpt.dtype}"
    assert pred_traj_tft.shape == pred_traj_ckpt.shape, (
        f"Shape mismatch: {pred_traj_tft.shape} vs {pred_traj_ckpt.shape}"
    )
    assert torch.allclose(pred_traj_tft.cpu(), pred_traj_ckpt.cpu(), atol=5e-2), "Outputs are not close enough"


if __name__ == "__main__":
    test_vision_model()
