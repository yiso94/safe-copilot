from __future__ import annotations

import importlib.util
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import torch
from tensorrt_llm.builder import Engine
from tensorrt_llm.runtime import GenerationSession
from tensorrt_llm.runtime import SamplingConfig as RuntimeSamplingConfig
from tensorrt_llm.runtime.generation import RuntimeTensor, _prepare_input_ids
from tensorrt_llm.runtime.model_runner import _engine_config_to_model_config
from transformers import AutoTokenizer

from .conversation import get_conv_template
from .utils import str_dtype_to_torch

MODEL_ID = "owl10/ReCogDrive-VLM-2B"
MODEL_OUTPUT_ROOT = Path(
    os.getenv(
        "INTERNVL_MODEL_OUTPUT_ROOT",
        "/workspaces/safe-copilot/models",
    )
)
SNAPSHOT = Path(
    "/root/.cache/huggingface/hub/models--owl10--ReCogDrive-VLM-2B/snapshots/16873acca08e3c04ab229b3d973f39aeba9db68d"
)
VISION_ENGINE_PATH = MODEL_OUTPUT_ROOT / "vision_projector_fp16.plan"
QWEN_ENGINE_DIR = MODEL_OUTPUT_ROOT / "qwen"

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
DEFAULT_IMAGE_SIZE = 448
DEFAULT_NUM_IMAGE_TOKEN = 256
DEFAULT_TEMPLATE = "internvl2_5"
SYSTEM_MESSAGE = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
""".strip()


def _load_intern_vit_module():
    module_name = "src.models.intern_vit_runtime"
    module = sys.modules.get(module_name)
    if module is not None:
        return module

    module_path = Path(__file__).with_name("intern_vit.py")
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load vision runtime module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_intern_vit = _load_intern_vit_module()
Preprocess = _intern_vit.Preprocess
InternVisionModel = _intern_vit.InternVisionModel


def _local_files_only(source: str | Path) -> bool:
    return Path(source).exists() or SNAPSHOT.exists()


def resolve_model_source(tokenizer_dir: str | Path | None = None) -> str:
    if tokenizer_dir is not None and Path(tokenizer_dir).exists():
        return str(Path(tokenizer_dir).resolve())
    if SNAPSHOT.exists():
        return str(SNAPSHOT)
    return MODEL_ID


def _load_tokenizer(tokenizer_dir: str | Path | None = None) -> AutoTokenizer:
    tokenizer_source = resolve_model_source(tokenizer_dir)
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_source,
        trust_remote_code=True,
        local_files_only=_local_files_only(tokenizer_source),
        use_fast=False,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def _pad_and_eos_ids(tokenizer: AutoTokenizer) -> tuple[int, int]:
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = eos_id if eos_id is not None else 0
    if eos_id is None:
        eos_id = pad_id
    return int(pad_id), int(eos_id)


class QWenInfer:
    def __init__(
        self,
        qwen_engine_dir: str | Path,
        device: torch.device = torch.device("cuda"),
    ):
        self.qwen_engine_dir = Path(qwen_engine_dir)
        self.device = device
        self.decoder = None
        self.engine = None
        self.config = None
        self.model_config = None
        self.model_dtype = None
        self.hidden_size = None
        self.vocab_size = None
        self.max_input_len = None
        self.max_prompt_embedding_table_size = None
        self.qwen_model_init()

    def get_model(self):
        if not (self.qwen_engine_dir / "config.json").exists() or not (self.qwen_engine_dir / "rank0.engine").exists():
            raise FileNotFoundError(
                f"Qwen TensorRT-LLM engine not found under {self.qwen_engine_dir}. "
                "Expected config.json and rank0.engine."
            )

        with open(self.qwen_engine_dir / "config.json") as file:
            config = json.load(file)

        engine = Engine.from_dir(str(self.qwen_engine_dir))
        model_config = _engine_config_to_model_config(engine.config)
        mapping = engine.config.pretrained_config.mapping
        decoder = GenerationSession(model_config, engine.engine, mapping, debug_mode=True)
        model_dtype = str_dtype_to_torch(str(config["pretrained_config"]["dtype"]))
        hidden_size = int(decoder.hidden_size)
        vocab_size = int(config["pretrained_config"]["vocab_size"])
        max_input_len = int(config["build_config"]["max_input_len"])
        max_prompt_embedding_table_size = int(config["build_config"].get("max_prompt_embedding_table_size", 0))

        return (
            config,
            engine,
            model_config,
            decoder,
            model_dtype,
            hidden_size,
            vocab_size,
            max_input_len,
            max_prompt_embedding_table_size,
        )

    def qwen_model_init(self):
        (
            self.config,
            self.engine,
            self.model_config,
            self.decoder,
            self.model_dtype,
            self.hidden_size,
            self.vocab_size,
            self.max_input_len,
            self.max_prompt_embedding_table_size,
        ) = self.get_model()

    def close(self):
        self.decoder = None
        self.engine = None

    def ptuning_setup(
        self,
        *,
        batch_size: int,
        prompt_embedding_table: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if self.max_prompt_embedding_table_size == 0:
            return None, None, None

        if prompt_embedding_table is not None:
            prompt_embedding_table = prompt_embedding_table.to(
                device=self.device,
                dtype=self.model_dtype,
            ).contiguous()
            task_vocab_size = int(prompt_embedding_table.shape[0])
            prompt_vocab_size = torch.tensor([task_vocab_size], dtype=torch.int32, device=self.device)
        else:
            prompt_embedding_table = torch.empty(
                [1, self.hidden_size],
                dtype=self.model_dtype,
                device=self.device,
            )
            prompt_vocab_size = torch.zeros([1], dtype=torch.int32, device=self.device)

        tasks = torch.zeros([batch_size], dtype=torch.int32, device=self.device)
        return prompt_embedding_table, tasks, prompt_vocab_size

    def _prepare_prompt_table_inputs(
        self,
        *,
        batch_size: int,
        prompt_embedding_table: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        return self.ptuning_setup(batch_size=batch_size, prompt_embedding_table=prompt_embedding_table)

    def get_last_hidden_state_from_input_ids(
        self,
        batch_input_ids: Sequence[torch.Tensor],
        *,
        attention_mask: torch.Tensor | None = None,
        prompt_embedding_table: torch.Tensor | None = None,
        pad_id: int = 0,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        if not batch_input_ids:
            raise ValueError("batch_input_ids must not be empty")
        if len(batch_input_ids) != 1:
            raise NotImplementedError("Only batch size 1 is supported")

        prepared_batch_input_ids = [
            input_ids.to(dtype=torch.int32, device=self.device).contiguous() for input_ids in batch_input_ids
        ]
        pad_id = int(pad_id)
        eos_id = pad_id if eos_id is None else int(eos_id)

        if self.decoder.remove_input_padding:
            input_ids, context_lengths = _prepare_input_ids(prepared_batch_input_ids)
            host_context_lengths = context_lengths.cpu()
            max_context_length = int(context_lengths.max().item())
        else:
            if attention_mask is None:
                raise ValueError("attention_mask is required when remove_input_padding is disabled")
            input_ids = torch.stack(prepared_batch_input_ids, dim=0)
            attention_mask = attention_mask.to(dtype=torch.int32, device=self.device).contiguous()
            context_lengths = attention_mask.sum(dim=-1).to(dtype=torch.int32)
            host_context_lengths = context_lengths.cpu()
            max_context_length = int(input_ids.shape[1])

        self.decoder.setup(
            batch_size=len(prepared_batch_input_ids),
            max_context_length=max_context_length,
            max_new_tokens=1,
            beam_width=1,
        )
        sampling_config = RuntimeSamplingConfig(
            end_id=eos_id,
            pad_id=pad_id,
            max_new_tokens=1,
        )
        self.decoder._GenerationSession__setup_decoder(input_ids, sampling_config, host_context_lengths)
        model_inputs = self.decoder._prepare_context_inputs(
            batch_size=len(prepared_batch_input_ids),
            context_lengths=context_lengths,
            host_context_lengths=host_context_lengths,
            use_gpt_attention_plugin=self.decoder.use_gpt_attention_plugin,
            remove_input_padding=self.decoder.remove_input_padding,
            max_context_length=max_context_length,
            input_ids=input_ids,
            pad_id=sampling_config.pad_id,
            eos_id=sampling_config.end_id,
        )

        prompt_embedding_table, tasks, prompt_vocab_size = self.ptuning_setup(
            batch_size=len(prepared_batch_input_ids),
            prompt_embedding_table=prompt_embedding_table,
        )
        dummy_cache_indirection = torch.zeros(
            (len(prepared_batch_input_ids), 1, self.decoder.max_seq_length),
            dtype=torch.int32,
            device=self.device,
        )
        context_tensors = self.decoder._get_context_shape_buffer(
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

        output_tensor_names = getattr(self.decoder.runtime, "output_tensor_names", [])
        total_tokens = int(context_lengths.sum().item())
        last_hidden_state_shape = (
            (total_tokens, self.hidden_size)
            if self.decoder.remove_input_padding
            else (len(prepared_batch_input_ids), max_context_length, self.hidden_size)
        )
        if "last_hidden_state_output" in output_tensor_names and "last_hidden_state_output" not in context_tensors:
            context_tensors["last_hidden_state_output"] = RuntimeTensor.from_torch(
                "last_hidden_state_output",
                torch.empty(
                    last_hidden_state_shape,
                    dtype=self.model_dtype,
                    device=self.device,
                ),
            )
        if "last_hidden_state_output" not in context_tensors:
            raise RuntimeError(
                f"The TensorRT-LLM engine at {self.qwen_engine_dir} does not expose last_hidden_state_output."
            )

        context = self.decoder.runtime.ctx_context
        self.decoder.runtime._set_tensors(context, context_tensors)
        engine_io_names = {
            self.decoder.runtime.engine.get_tensor_name(index)
            for index in range(self.decoder.runtime.engine.num_io_tensors)
        }
        for name, tensor in context_tensors.items():
            if name not in engine_io_names:
                continue
            if context.get_tensor_address(name) != tensor.data:
                try:
                    if list(context.get_tensor_shape(name)) != list(tensor.shape):
                        context.set_input_shape(name, tensor.shape)
                except Exception:
                    pass
                context.set_tensor_address(name, tensor.data)
        ok = self.decoder.runtime._run(context, torch.cuda.current_stream().cuda_stream)
        assert ok, "Runtime execution failed for qwen hidden-state session"
        torch.cuda.synchronize()

        last_hidden_state = context_tensors["last_hidden_state_output"].to_torch()
        return last_hidden_state.reshape(len(prepared_batch_input_ids), -1, self.hidden_size)

    def forward(
        self,
        *,
        input_ids: torch.Tensor | Sequence[torch.Tensor],
        attention_mask: torch.Tensor | None = None,
        prompt_embedding_table: torch.Tensor | None = None,
        pad_id: int = 0,
        eos_id: int | None = None,
    ) -> torch.Tensor:
        if isinstance(input_ids, torch.Tensor):
            if input_ids.dim() == 1:
                batch_input_ids = [input_ids]
            elif input_ids.dim() == 2:
                batch_input_ids = [row for row in input_ids]
            else:
                raise ValueError("input_ids tensor must have rank 1 or 2")
        else:
            batch_input_ids = list(input_ids)

        return self.get_last_hidden_state_from_input_ids(
            batch_input_ids,
            attention_mask=attention_mask,
            prompt_embedding_table=prompt_embedding_table,
            pad_id=pad_id,
            eos_id=eos_id,
        )


class InternVLChatTRT(torch.nn.Module):
    def __init__(
        self,
        *,
        device: str = "cuda:0",
        tokenizer_dir: str | Path | None = None,
        vision_engine_path: str | Path = VISION_ENGINE_PATH,
        qwen_engine_dir: str | Path = QWEN_ENGINE_DIR,
        max_input_len: int | None = None,
        image_size: int = DEFAULT_IMAGE_SIZE,
        num_image_token: int = DEFAULT_NUM_IMAGE_TOKEN,
        template: str = DEFAULT_TEMPLATE,
        system_message: str = SYSTEM_MESSAGE,
    ):
        super().__init__()
        self.device = torch.device(device)
        self.image_size = image_size
        self.num_image_token = num_image_token
        self.template = template
        self.system_message = system_message
        self.vision_engine_path = Path(vision_engine_path)
        self.qwen_engine_dir = Path(qwen_engine_dir)

        self.tokenizer = _load_tokenizer(tokenizer_dir)
        self.qwen_model = QWenInfer(self.qwen_engine_dir, self.device)
        self.hidden_size = self.qwen_model.hidden_size
        self.vocab_size = self.qwen_model.vocab_size
        self.max_prompt_embedding_table_size = self.qwen_model.max_prompt_embedding_table_size
        if max_input_len is None:
            self.max_input_len = self.qwen_model.max_input_len
        else:
            if max_input_len > self.qwen_model.max_input_len:
                raise ValueError(
                    f"Requested max_input_len={max_input_len}, but the engine only supports "
                    f"{self.qwen_model.max_input_len} tokens."
                )
            self.max_input_len = max_input_len

        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self._vision_model = None

    def _get_vision_model(self) -> InternVisionModel:
        if self._vision_model is None:
            self._vision_model = InternVisionModel(
                self.vision_engine_path,
                torch.cuda.current_stream().cuda_stream,
                self.device,
            )
        return self._vision_model

    def close(self):
        self.qwen_model.close()
        self._vision_model = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._get_vision_model().infer(pixel_values)

    def make_context(self, question: str, num_patches: int) -> str:
        if "<image>" not in question:
            question = "<image>\n" + question

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()
        image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
        return query.replace("<image>", image_tokens, 1)

    def _build_query(self, question: str, num_patches: int) -> str:
        return self.make_context(question, num_patches)

    def _inject_visual_features(
        self,
        input_ids: torch.Tensor,
        prompt_embedding_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.max_prompt_embedding_table_size <= 0:
            raise RuntimeError(
                f"The Qwen engine at {self.qwen_engine_dir} was built with max_prompt_embedding_table_size=0. "
                "This runtime can return text last hidden states, but multimodal last hidden states require "
                "a Qwen hidden-state engine rebuilt with prompt-embedding capacity."
            )

        selected = input_ids == self.img_context_token_id
        num_virtual_tokens = int(selected.sum().item())
        if num_virtual_tokens == 0:
            raise ValueError("No <IMG_CONTEXT> tokens were found in the query.")
        if num_virtual_tokens != int(prompt_embedding_table.shape[0]):
            raise ValueError(
                "The number of <IMG_CONTEXT> tokens does not match the number of vision embeddings: "
                f"{num_virtual_tokens} vs {prompt_embedding_table.shape[0]}."
            )
        if num_virtual_tokens > self.max_prompt_embedding_table_size:
            raise ValueError(
                "The engine does not have enough prompt-embedding capacity for this image. "
                f"Required {num_virtual_tokens}, built with {self.max_prompt_embedding_table_size}."
            )

        remapped_input_ids = input_ids.clone()
        remapped_input_ids[selected] = torch.arange(
            self.vocab_size,
            self.vocab_size + num_virtual_tokens,
            dtype=remapped_input_ids.dtype,
        )
        return remapped_input_ids, prompt_embedding_table

    def _prepare_virtual_input_ids(
        self,
        input_ids: torch.Tensor,
        prompt_embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        remapped_input_ids, _ = self._inject_visual_features(input_ids, prompt_embedding_table)
        return remapped_input_ids

    def get_text_last_hidden_state(self, prompt: str) -> torch.Tensor:
        tokenizer_padding = False if self.qwen_model.decoder.remove_input_padding else "max_length"
        model_inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding=tokenizer_padding,
            truncation=True,
            max_length=self.max_input_len,
        )
        pad_id, eos_id = _pad_and_eos_ids(self.tokenizer)
        return self.qwen_model.forward(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            pad_id=pad_id,
            eos_id=eos_id,
        )

    def get_multimodal_last_hidden_state(
        self,
        *,
        question: str,
        pixel_values: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if visual_features is None:
            if pixel_values is None:
                raise ValueError("Either pixel_values or visual_features must be provided.")
            visual_features = self.extract_feature(pixel_values)

        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(0)
        if visual_features.dim() != 3:
            raise ValueError("visual_features must have shape [patches, tokens, hidden] or [tokens, hidden]")

        query = self.make_context(question, num_patches=int(visual_features.shape[0]))
        tokenizer_padding = False if self.qwen_model.decoder.remove_input_padding else "max_length"
        model_inputs = self.tokenizer(
            [query],
            return_tensors="pt",
            padding=tokenizer_padding,
            truncation=True,
            max_length=self.max_input_len,
        )
        prompt_embedding_table = visual_features.reshape(-1, visual_features.shape[-1]).contiguous()
        remapped_input_ids, prompt_embedding_table = self._inject_visual_features(
            model_inputs["input_ids"],
            prompt_embedding_table,
        )
        pad_id, eos_id = _pad_and_eos_ids(self.tokenizer)
        return self.qwen_model.forward(
            input_ids=remapped_input_ids[0],
            attention_mask=model_inputs["attention_mask"],
            prompt_embedding_table=prompt_embedding_table,
            pad_id=pad_id,
            eos_id=eos_id,
        )

    def forward(
        self,
        pixel_values: torch.Tensor | None,
        questions: list[str],
        num_patches_list: list[int] | None = None,
    ) -> SimpleNamespace:
        if pixel_values is None:
            outputs = [self.get_text_last_hidden_state(question) for question in questions]
            return SimpleNamespace(hidden_states=(torch.cat(outputs, dim=0),))

        if num_patches_list is None:
            raise ValueError("num_patches_list is required when pixel_values is provided")

        outputs = []
        patch_offset = 0
        for question, num_patches in zip(questions, num_patches_list):
            sample_pixel_values = pixel_values[patch_offset : patch_offset + num_patches]
            patch_offset += num_patches
            outputs.append(
                self.get_multimodal_last_hidden_state(
                    question=question,
                    pixel_values=sample_pixel_values,
                )
            )

        return SimpleNamespace(hidden_states=(torch.cat(outputs, dim=0),))


ReCogDriveVLMTRT = InternVLChatTRT
QWenHiddenStateInfer = QWenInfer

__all__ = [
    "IMG_CONTEXT_TOKEN",
    "IMG_END_TOKEN",
    "IMG_START_TOKEN",
    "InternVLChatTRT",
    "Preprocess",
    "QWenHiddenStateInfer",
    "QWenInfer",
    "ReCogDriveVLMTRT",
]
