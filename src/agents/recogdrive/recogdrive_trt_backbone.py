from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

import tensorrt as trt
import torch
from tensorrt_llm.builder import Engine
from tensorrt_llm.runtime import GenerationSession
from tensorrt_llm.runtime.generation import _prepare_input_ids
from tensorrt_llm.runtime.model_runner import _engine_config_to_model_config
from transformers import AutoConfig, AutoTokenizer

from .recogdrive_backbone import (
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    system_message,
)
from .utils.conversation import get_conv_template

MODEL_ID = "owl10/ReCogDrive-VLM-2B"
MODEL_OUTPUT_ROOT = Path(
    os.getenv(
        "RECOGDRIVE_MODEL_OUTPUT_ROOT",
        "/workspaces/safe-copilot/models",
    )
)
ROOT = MODEL_OUTPUT_ROOT / "recogdrive"
SNAPSHOT = Path(
    "/root/.cache/huggingface/hub/models--owl10--ReCogDrive-VLM-2B/snapshots/16873acca08e3c04ab229b3d973f39aeba9db68d"
)
ENGINE_ROOT = Path(
    os.getenv(
        "TRT_HIDDEN_STATE_ENGINE_ROOT",
        "/workspaces/safe-copilot/models/trt_llm_hidden_state_engines",
    )
)
VISION_PLAN = ROOT / "vision_trt" / "vision_projector.plan"
LLM_DIR = ROOT / "qwen2_submodel"


def _local_files_only() -> bool:
    return SNAPSHOT.exists()


def resolve_model_source() -> str:
    if SNAPSHOT.exists():
        return str(SNAPSHOT)
    return MODEL_ID


def hidden_state_engine_dir_for(
    model_id: str,
    max_input_len: int,
    max_prompt_embedding_table_size: int,
    *,
    remove_input_padding: bool = False,
    gpt_attention_plugin: str | None = None,
) -> Path:
    safe_model_id = model_id.replace("/", "__")
    suffix = f"max_input_len_{max_input_len}"
    if max_prompt_embedding_table_size > 0:
        suffix += f"__max_prompt_embedding_table_size_{max_prompt_embedding_table_size}"
    if not remove_input_padding:
        suffix += "__remove_input_padding_false"
    if gpt_attention_plugin is None:
        suffix += "__gpt_attention_plugin_none"
    return ENGINE_ROOT / safe_model_id / suffix


def _resolve_tokenizer_source(llm_dir: Path) -> str:
    tokenizer_source = os.getenv("RECOGDRIVE_TOKENIZER_SOURCE")
    if tokenizer_source:
        return tokenizer_source

    if (llm_dir / "tokenizer.json").exists() or (llm_dir / "tokenizer.model").exists():
        return str(llm_dir)

    return resolve_model_source()


def _load_tokenizer(llm_dir: Path) -> AutoTokenizer:
    kwargs = dict(
        trust_remote_code=True,
        local_files_only=_local_files_only(),
        use_fast=False,
    )

    try:
        return AutoTokenizer.from_pretrained(str(llm_dir), **kwargs)
    except Exception:
        source = _resolve_tokenizer_source(llm_dir)

    full_config = AutoConfig.from_pretrained(
        resolve_model_source(),
        trust_remote_code=True,
        local_files_only=_local_files_only(),
    )

    try:
        return AutoTokenizer.from_pretrained(
            source,
            config=full_config.llm_config,
            fix_mistral_regex=True,
            **kwargs,
        )
    except TypeError:
        return AutoTokenizer.from_pretrained(
            source,
            config=full_config.llm_config,
            **kwargs,
        )


def _prepare_prompt_table_inputs(
    *,
    batch_size: int,
    prompt_embedding_table: torch.Tensor | None,
    max_prompt_embedding_table_size: int,
    hidden_size: int,
) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
    if max_prompt_embedding_table_size == 0:
        return None, None, None

    if prompt_embedding_table is not None:
        prompt_embedding_table = prompt_embedding_table.to(
            device="cuda",
            dtype=torch.float16,
        ).contiguous()
        task_vocab_size, prompt_hidden_size = prompt_embedding_table.shape
        prompt_vocab_size = torch.tensor([task_vocab_size], dtype=torch.int32, device="cuda")
    else:
        prompt_embedding_table = torch.empty(
            [1, hidden_size],
            dtype=torch.float16,
            device="cuda",
        )
        prompt_vocab_size = torch.zeros([1], dtype=torch.int32, device="cuda")

    tasks = torch.zeros([batch_size], dtype=torch.int32, device="cuda")
    return prompt_embedding_table, tasks, prompt_vocab_size


class RecogDriveTRTBackbone(torch.nn.Module):
    def __init__(
        self,
        *,
        device: str = "cuda:0",
        llm_dir: str | Path | None = None,
        vision_engine_path: str | Path | None = None,
        hidden_state_engine_dir: str | Path | None = None,
        hidden_state_max_input_len: int = 2800,
        hidden_state_max_prompt_embedding_table_size: int = 3328,
        hidden_state_remove_input_padding: bool = False,
        hidden_state_gpt_attention_plugin: str | None = None,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.num_image_token = 256
        self.hidden_state_max_input_len = hidden_state_max_input_len
        self.hidden_state_max_prompt_embedding_table_size = hidden_state_max_prompt_embedding_table_size
        self.hidden_state_remove_input_padding = hidden_state_remove_input_padding
        self.hidden_state_gpt_attention_plugin = hidden_state_gpt_attention_plugin
        self.llm_dir = Path(llm_dir or LLM_DIR)
        self.vision_engine_path = Path(vision_engine_path or VISION_PLAN)
        self.hidden_state_engine_dir = Path(
            hidden_state_engine_dir
            or hidden_state_engine_dir_for(
                str(self.llm_dir),
                hidden_state_max_input_len,
                hidden_state_max_prompt_embedding_table_size,
                remove_input_padding=hidden_state_remove_input_padding,
                gpt_attention_plugin=hidden_state_gpt_attention_plugin,
            )
        )

        if not self.vision_engine_path.exists():
            raise FileNotFoundError(
                f"Vision TRT engine was not found at {self.vision_engine_path}. "
                "Build it with examples/models/recogdrive/convert_checkpoint.py first."
            )
        if not self.hidden_state_engine_dir.exists():
            raise FileNotFoundError(
                f"Hidden-state TRT-LLM engine was not found at {self.hidden_state_engine_dir}. "
                "Build it with examples/models/recogdrive/convert_checkpoint.py first."
            )
        if not self.llm_dir.exists():
            raise FileNotFoundError(
                f"Extracted language submodel directory was not found at {self.llm_dir}. "
                "Build it with examples/models/recogdrive/convert_checkpoint.py first."
            )

        self.tokenizer = _load_tokenizer(self.llm_dir)
        self.tokenizer.padding_side = "left"
        self.llm_config = AutoConfig.from_pretrained(
            str(self.llm_dir),
            trust_remote_code=True,
            local_files_only=True,
        )
        self.vocab_size = int(self.llm_config.vocab_size)
        self.pad_token_id = int(self.tokenizer.pad_token_id or self.tokenizer.eos_token_id or 0)
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        self._vision_logger = trt.Logger(trt.Logger.WARNING)
        self._vision_runtime = trt.Runtime(self._vision_logger)
        self._vision_engine = self._vision_runtime.deserialize_cuda_engine(self.vision_engine_path.read_bytes())
        self._vision_context = self._vision_engine.create_execution_context()

        self._hidden_state_engine = Engine.from_dir(str(self.hidden_state_engine_dir))
        model_config = _engine_config_to_model_config(self._hidden_state_engine.config)
        mapping = self._hidden_state_engine.config.pretrained_config.mapping
        self._hidden_state_session = GenerationSession(
            model_config,
            self._hidden_state_engine.engine,
            mapping,
            debug_mode=True,
        )

    def _build_query(self, question: str, num_patches: int) -> str:
        if "<image>" not in question:
            question = "<image>\n" + question

        template = get_conv_template("internvl2_5")
        template.system_message = system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        return query.replace("<image>", image_tokens, 1)

    def _run_single_vision_patch(self, pixel_values: torch.Tensor) -> torch.Tensor:
        pixel_values = pixel_values.to(device=self.device, dtype=torch.float16).contiguous()
        self._vision_context.set_input_shape("pixel_values", tuple(pixel_values.shape))
        output_shape = tuple(self._vision_context.get_tensor_shape("image_embeds"))
        output = torch.empty(output_shape, device=self.device, dtype=torch.float16)
        self._vision_context.set_tensor_address("pixel_values", pixel_values.data_ptr())
        self._vision_context.set_tensor_address("image_embeds", output.data_ptr())
        ok = self._vision_context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT vision engine execution failed")
        torch.cuda.synchronize()
        return output

    def _extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        image_embeds = []
        for patch in pixel_values:
            image_embeds.append(self._run_single_vision_patch(patch.unsqueeze(0)))
        return torch.cat(image_embeds, dim=0)

    def _prepare_virtual_input_ids(
        self,
        input_ids: torch.Tensor,
        prompt_embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        selected = input_ids == self.img_context_token_id
        num_virtual_tokens = int(selected.sum().item())
        if num_virtual_tokens == 0:
            raise ValueError("The TRT backbone expected at least one <IMG_CONTEXT> token in the prompt.")

        if num_virtual_tokens != int(prompt_embedding_table.shape[0]):
            raise ValueError(
                "The number of <IMG_CONTEXT> tokens does not match the number of TRT vision embeddings: "
                f"{num_virtual_tokens} tokens vs {prompt_embedding_table.shape[0]} embeddings."
            )

        if num_virtual_tokens > self.hidden_state_max_prompt_embedding_table_size:
            raise ValueError(
                "The TRT hidden-state engine does not have enough prompt-embedding capacity for this image. "
                f"Required {num_virtual_tokens}, but the engine was built with "
                f"{self.hidden_state_max_prompt_embedding_table_size}."
            )

        remapped_input_ids = input_ids.clone()
        remapped_input_ids[selected] = torch.arange(
            self.vocab_size,
            self.vocab_size + num_virtual_tokens,
            dtype=remapped_input_ids.dtype,
        )
        return remapped_input_ids

    def _run_hidden_state_engine(
        self,
        batch_input_ids: Sequence[torch.Tensor],
        *,
        prompt_embedding_table: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        session = self._hidden_state_session
        batch_size = len(batch_input_ids)

        if session.remove_input_padding:
            prepared_batch_input_ids = [
                input_ids.to(dtype=torch.int32, device=self.device).contiguous() for input_ids in batch_input_ids
            ]
            input_ids, context_lengths = _prepare_input_ids(prepared_batch_input_ids)
            host_context_lengths = context_lengths.cpu()
            max_context_length = int(context_lengths.max().item())
            prepare_kwargs = {
                "input_ids": input_ids,
                "pad_id": self.pad_token_id,
                "eos_id": self.tokenizer.eos_token_id or 0,
            }
        else:
            if attention_mask is None:
                raise ValueError("attention_mask is required when remove_input_padding is disabled.")
            input_ids = torch.stack(batch_input_ids).to(dtype=torch.int32, device=self.device).contiguous()
            attention_mask = attention_mask.to(dtype=torch.int32, device=self.device).contiguous()
            context_lengths = attention_mask.sum(dim=-1).to(dtype=torch.int32)
            host_context_lengths = context_lengths.cpu()
            max_context_length = int(input_ids.shape[1])
            prepare_kwargs = {
                "input_ids": input_ids,
                "pad_id": self.pad_token_id,
                "eos_id": self.tokenizer.eos_token_id or 0,
            }

        session.setup(
            batch_size=batch_size,
            max_context_length=max_context_length,
            max_new_tokens=1,
            beam_width=1,
        )

        model_inputs = session._prepare_context_inputs(
            batch_size=batch_size,
            context_lengths=context_lengths,
            host_context_lengths=host_context_lengths,
            use_gpt_attention_plugin=session.use_gpt_attention_plugin,
            remove_input_padding=session.remove_input_padding,
            max_context_length=max_context_length,
            **prepare_kwargs,
        )
        prompt_embedding_table, tasks, prompt_vocab_size = _prepare_prompt_table_inputs(
            batch_size=batch_size,
            prompt_embedding_table=prompt_embedding_table,
            max_prompt_embedding_table_size=self.hidden_state_max_prompt_embedding_table_size,
            hidden_size=session.hidden_size,
        )

        dummy_cache_indirection = torch.zeros(
            (batch_size, 1, session.max_seq_length),
            dtype=torch.int32,
            device=self.device,
        )
        context_tensors = session._get_context_shape_buffer(
            input_ids=input_ids,
            context_lengths=context_lengths,
            host_context_lengths=host_context_lengths,
            position_ids=model_inputs.get("position_ids"),
            last_token_ids=model_inputs.get("last_token_ids"),
            attention_mask=model_inputs.get("attention_mask"),
            cross_attention_mask=None,
            cache_indirection=dummy_cache_indirection,
            kv_cache_block_offsets=None,
            host_kv_cache_block_offsets=None,
            cross_kv_cache_block_offsets=None,
            host_cross_kv_cache_block_offsets=None,
            hidden_states_input=None,
            prompt_embedding_table=prompt_embedding_table,
            tasks=tasks,
            prompt_vocab_size=prompt_vocab_size,
            encoder_output=None,
            encoder_input_lengths=None,
            host_runtime_perf_knobs=model_inputs.get("host_runtime_perf_knobs"),
            host_context_progress=torch.tensor([0], dtype=torch.int64),
            skip_cross_attn_blocks=None,
            language_adapter_routings=None,
        )

        context = session.runtime.ctx_context
        session.runtime._set_tensors(context, context_tensors)
        ok = session.runtime._run(context, torch.cuda.current_stream().cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT-LLM hidden-state engine execution failed")
        torch.cuda.synchronize()

        last_hidden_state = context_tensors["last_hidden_state_output"].to_torch()
        return last_hidden_state.reshape(batch_size, -1, session.hidden_size)

    def _get_last_hidden_state(self, question: str, pixel_values: torch.Tensor) -> torch.Tensor:
        query = self._build_query(question, num_patches=pixel_values.shape[0])
        model_inputs = self.tokenizer(
            [query],
            return_tensors="pt",
            padding="max_length",
            max_length=self.hidden_state_max_input_len,
        )
        prompt_embedding_table = self._extract_feature(pixel_values).reshape(-1, self._hidden_state_session.hidden_size)
        remapped_input_ids = self._prepare_virtual_input_ids(
            model_inputs["input_ids"],
            prompt_embedding_table=prompt_embedding_table,
        )
        return self._run_hidden_state_engine(
            [remapped_input_ids[0]],
            prompt_embedding_table=prompt_embedding_table,
            attention_mask=model_inputs["attention_mask"],
        )

    def forward(
        self,
        pixel_values: torch.Tensor,
        questions: list[str],
        num_patches_list: list[int],
    ) -> SimpleNamespace:
        if len(questions) != len(num_patches_list):
            raise ValueError(
                "questions and num_patches_list must have the same length, "
                f"but received {len(questions)} and {len(num_patches_list)}."
            )

        outputs = []
        patch_offset = 0
        for question, num_patches in zip(questions, num_patches_list):
            sample_pixel_values = pixel_values[patch_offset : patch_offset + num_patches]
            patch_offset += num_patches
            outputs.append(self._get_last_hidden_state(question, sample_pixel_values))

        if patch_offset != pixel_values.shape[0]:
            raise ValueError(
                "The provided num_patches_list does not match the number of image patches. "
                f"Consumed {patch_offset}, but received {pixel_values.shape[0]}."
            )

        return SimpleNamespace(hidden_states=(torch.cat(outputs, dim=0),))
