import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from recogdrive.recogdrive_vlm_engines import (
    build_diffusion_trt_engine,
    build_language_generation_trtllm_engine,
    build_language_trtllm_engine,
    build_vision_trt_engine,
)


def main() -> None:
    hidden_state_max_input_len = int(os.getenv("HIDDEN_STATE_MAX_INPUT_LEN", "2800"))
    hidden_state_max_prompt_embedding_table_size = int(
        os.getenv("HIDDEN_STATE_MAX_PROMPT_EMBEDDING_TABLE_SIZE", "3328")
    )
    hidden_state_remove_input_padding = os.getenv("HIDDEN_STATE_REMOVE_INPUT_PADDING", "0") == "1"
    hidden_state_gpt_attention_plugin = os.getenv("HIDDEN_STATE_GPT_ATTENTION_PLUGIN")
    if hidden_state_gpt_attention_plugin == "none":
        hidden_state_gpt_attention_plugin = None
    diffusion_max_vl_seq_len = int(
        os.getenv("DIFFUSION_MAX_VL_SEQ_LEN", str(hidden_state_max_input_len))
    )
    diffusion_sampling_method = os.getenv("RECOGDRIVE_DIFFUSION_SAMPLING_METHOD", "ddim")
    generation_max_input_len = int(os.getenv("GENERATION_MAX_INPUT_LEN", "1024"))
    generation_max_output_len = int(os.getenv("GENERATION_MAX_OUTPUT_LEN", "128"))
    generation_max_prompt_embedding_table_size = int(os.getenv("GENERATION_MAX_PROMPT_EMBEDDING_TABLE_SIZE", "3072"))

    vision = build_vision_trt_engine()
    hidden_state = build_language_trtllm_engine(
        hidden_state_max_input_len,
        max_prompt_embedding_table_size=hidden_state_max_prompt_embedding_table_size,
        remove_input_padding=hidden_state_remove_input_padding,
        gpt_attention_plugin=hidden_state_gpt_attention_plugin,
    )
    diffusion = build_diffusion_trt_engine(
        max_vl_seq_len=diffusion_max_vl_seq_len,
        sampling_method=diffusion_sampling_method,
    )
    generation = build_language_generation_trtllm_engine(
        max_input_len=generation_max_input_len,
        max_output_len=generation_max_output_len,
        max_prompt_embedding_table_size=generation_max_prompt_embedding_table_size,
    )
    print(f"vision_trt_engine={vision}")
    print(f"llm_hidden_state_trtllm_engine={hidden_state}")
    print(f"diffusion_trt_engine={diffusion}")
    print(f"llm_generation_trtllm_engine={generation}")


if __name__ == "__main__":
    main()
