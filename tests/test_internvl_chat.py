from pathlib import Path

import pytest
import torch
from transformers import AutoModel

from models.internvl_chat import InternVLChatTRT

REQUIRED_ARTIFACTS = [
    Path("/workspaces/safe-copilot/models/vision_projector_fp16.plan"),
    Path("/workspaces/safe-copilot/models/qwen/config.json"),
    Path("/workspaces/safe-copilot/models/qwen/rank0.engine"),
]
PROMPT = "Describe the driving scene."


def test_internvl_trt_runtime_returns_last_hidden_state():
    missing = [str(path) for path in REQUIRED_ARTIFACTS if not path.exists()]
    if missing:
        pytest.skip(f"InternVL TRT artifacts are missing: {missing}")
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for the InternVL TRT runtime parity test")

    runtime = InternVLChatTRT()
    try:
        trt_hidden_state = runtime.get_text_last_hidden_state(PROMPT).detach().float().cpu()
        tokenizer = runtime.tokenizer
        max_input_len = runtime.max_input_len
        use_padding = False if runtime.qwen_model.decoder.remove_input_padding else "max_length"
    finally:
        runtime.close()

    model = AutoModel.from_pretrained(
        "owl10/ReCogDrive-VLM-2B",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=True,
        device_map="cuda",
    ).eval()

    try:
        model_inputs = tokenizer(
            [PROMPT],
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
        "TRT last hidden state is too far from owl10/ReCogDrive-VLM-2B. "
        f"max_abs_diff={max_abs_diff:.6f}, mean_abs_diff={mean_abs_diff:.6f}"
    )
