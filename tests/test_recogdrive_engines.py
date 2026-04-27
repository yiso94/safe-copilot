import gc
import sys
from pathlib import Path

import pytest
import torch

EXAMPLES_MODELS_ROOT = Path(__file__).resolve().parents[1] / "examples" / "models"
if str(EXAMPLES_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_MODELS_ROOT))

from recogdrive import recogdrive_vlm_engines as rv

VISION_COSINE_MEAN_MIN = 0.999
VISION_COSINE_MIN_MIN = 0.99
LLM_COSINE_MEAN_MIN = 0.995
LLM_COSINE_MIN_MIN = 0.99

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for ReCogDrive engine parity tests"),
    pytest.mark.skipif(not rv.SNAPSHOT.exists(), reason=f"Local ReCogDrive snapshot not found: {rv.SNAPSHOT}"),
]


def _release_model(model) -> None:
    del model
    gc.collect()
    torch.cuda.empty_cache()


def _make_sample_input_ids(model) -> torch.Tensor:
    llm_config = model.config.llm_config
    bos_token_id = getattr(llm_config, "bos_token_id", 0) or 0
    eos_token_id = getattr(llm_config, "eos_token_id", bos_token_id) or bos_token_id
    vocab_size = getattr(llm_config, "vocab_size", 151682)
    middle_tokens = [min(374, vocab_size - 1), min(1492, vocab_size - 1)]
    return torch.tensor([[bos_token_id, *middle_tokens, eos_token_id]], dtype=torch.long)


def test_recogdrive_vision_engine_matches_transformers() -> None:
    engine_path = rv.build_vision_trt_engine()
    assert engine_path.exists()

    torch.cuda.empty_cache()
    model = rv.load_reference_model(device_map="cuda:0", use_flash_attn=False)
    image = torch.zeros(1, 3, 448, 448, device="cuda", dtype=torch.bfloat16)

    with torch.inference_mode():
        reference_vision = model.extract_feature(image)

    converted_vision = rv.run_vision_trt_engine(image.to(dtype=torch.float16))
    metrics = rv._metric_summary(reference_vision, converted_vision)

    assert metrics["reference_shape"] == metrics["candidate_shape"]
    assert metrics["cosine_mean"] >= VISION_COSINE_MEAN_MIN
    assert metrics["cosine_min"] >= VISION_COSINE_MIN_MIN

    _release_model(model)


def test_recogdrive_hidden_state_engine_matches_transformers() -> None:
    max_input_len = 64
    engine_dir = rv.build_language_trtllm_engine(max_input_len=max_input_len)
    assert engine_dir.exists()
    assert (engine_dir / "rank0.engine").exists()

    torch.cuda.empty_cache()
    model = rv.load_reference_model(device_map="cuda:0")
    sample_input_ids = _make_sample_input_ids(model)
    input_ids = sample_input_ids.to(device="cuda")
    attention_mask = torch.ones_like(input_ids)

    with torch.inference_mode():
        reference_hidden = model.language_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]

    converted_hidden = rv.get_language_last_hidden_state_from_input_ids(
        [sample_input_ids[0]],
        max_input_len=max_input_len,
    )
    metrics = rv._metric_summary(reference_hidden, converted_hidden)

    assert metrics["reference_shape"] == metrics["candidate_shape"]
    assert metrics["cosine_mean"] >= LLM_COSINE_MEAN_MIN
    assert metrics["cosine_min"] >= LLM_COSINE_MIN_MIN

    _release_model(model)
