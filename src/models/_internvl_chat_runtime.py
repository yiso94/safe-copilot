from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace

import tensorrt as trt
import torch
from PIL import Image
from tensorrt_llm.builder import Engine
from tensorrt_llm.runtime import GenerationSession
from tensorrt_llm.runtime.generation import _prepare_input_ids
from tensorrt_llm.runtime.model_runner import _engine_config_to_model_config
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoConfig, AutoTokenizer

from .conversation import get_conv_template

MODEL_ID = "owl10/ReCogDrive-VLM-2B"
MODEL_OUTPUT_ROOT = Path(
    os.getenv(
        "INTERNVL_MODEL_OUTPUT_ROOT",
        "/workspaces/safe-copilot/models",
    )
)
ROOT = MODEL_OUTPUT_ROOT / "internvl"
SNAPSHOT = Path(
    "/root/.cache/huggingface/hub/models--owl10--ReCogDrive-VLM-2B/snapshots/16873acca08e3c04ab229b3d973f39aeba9db68d"
)
VISION_PLAN = ROOT / "vision_projector_fp16.plan"
LLM_DIR = ROOT / "qwen_hf_model"
HIDDEN_STATE_ENGINE_DIR = ROOT / "qwen_hidden_state_engine"

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


def _local_files_only() -> bool:
    return SNAPSHOT.exists()


def resolve_model_source() -> str:
    if SNAPSHOT.exists():
        return str(SNAPSHOT)
    return MODEL_ID


class Preprocess:
    def __init__(self, image_size: int = DEFAULT_IMAGE_SIZE):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def encode(self, image_paths: list[str]) -> torch.Tensor:
        images = []
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            images.append(self.image_transform(image))
        return torch.stack(images, dim=0)


def _load_tokenizer(llm_dir: Path) -> AutoTokenizer:
    kwargs = dict(
        trust_remote_code=True,
        local_files_only=_local_files_only(),
        use_fast=False,
    )
    try:
        tokenizer = AutoTokenizer.from_pretrained(str(llm_dir), **kwargs)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(resolve_model_source(), **kwargs)
    if tokenizer.pad_token_id is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


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
        task_vocab_size, _ = prompt_embedding_table.shape
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


class InternVLChatTRT(torch.nn.Module):
    def __init__(
        self,
        *,
        device: str = "cuda:0",
        llm_dir: str | Path | None = None,
        vision_engine_path: str | Path | None = None,
        hidden_state_engine_dir: str | Path | None = None,
        hidden_state_max_input_len: int = 2800,
        hidden_state_max_prompt_embedding_table_size: int = 3328,
        image_size: int = DEFAULT_IMAGE_SIZE,
        num_image_token: int = DEFAULT_NUM_IMAGE_TOKEN,
        template: str = DEFAULT_TEMPLATE,
        system_message: str = SYSTEM_MESSAGE,
    ) -> None:
        super().__init__()

        self.device = torch.device(device)
        self.image_size = image_size
        self.num_image_token = num_image_token
        self.template = template
        self.system_message = system_message
        self.hidden_state_max_input_len = hidden_state_max_input_len
        self.hidden_state_max_prompt_embedding_table_size = hidden_state_max_prompt_embedding_table_size
        self.llm_dir = Path(llm_dir or LLM_DIR)
        self.vision_engine_path = Path(vision_engine_path or VISION_PLAN)
        self.hidden_state_engine_dir = Path(hidden_state_engine_dir or HIDDEN_STATE_ENGINE_DIR)

        if not self.vision_engine_path.exists():
            raise FileNotFoundError(
                f"Vision TensorRT engine not found at {self.vision_engine_path}. "
                "Build it with examples/models/internvl/vit_onnx_trt.py first."
            )
        if not self.hidden_state_engine_dir.exists():
            raise FileNotFoundError(
                f"Hidden-state TensorRT-LLM engine not found at {self.hidden_state_engine_dir}. "
                "Build it with examples/models/qwen/build_trtllm.py first."
            )
        if not self.llm_dir.exists():
            raise FileNotFoundError(
                f"Extracted Qwen submodel directory not found at {self.llm_dir}. "
                "Build it with examples/models/qwen/convert_checkpoint.py first."
            )

        self.tokenizer = _load_tokenizer(self.llm_dir)
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
        if self._vision_engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT engine from {self.vision_engine_path}")
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
        self.hidden_size = int(self._hidden_state_session.hidden_size)

    def close(self) -> None:
        self._hidden_state_session = None
        self._hidden_state_engine = None
        self._vision_context = None
        self._vision_engine = None
        self._vision_runtime = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def _build_query(self, question: str, num_patches: int) -> str:
        if "<image>" not in question:
            question = "<image>\n" + question

        template = get_conv_template(self.template)
        template.system_message = self.system_message
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

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        if pixel_values.dim() != 4:
            raise ValueError("pixel_values must have shape [patches, 3, H, W] or [3, H, W]")

        image_embeds = []
        for patch in pixel_values:
            image_embeds.append(self._run_single_vision_patch(patch.unsqueeze(0)))
        return torch.cat(image_embeds, dim=0)

    def _prepare_virtual_input_ids(
        self,
        input_ids: torch.Tensor,
        *,
        prompt_embedding_table: torch.Tensor,
    ) -> torch.Tensor:
        selected = input_ids == self.img_context_token_id
        num_virtual_tokens = int(selected.sum().item())
        if num_virtual_tokens == 0:
            raise ValueError("The prompt does not contain any <IMG_CONTEXT> tokens.")
        if num_virtual_tokens != int(prompt_embedding_table.shape[0]):
            raise ValueError(
                "The number of <IMG_CONTEXT> tokens does not match the number of vision embeddings: "
                f"{num_virtual_tokens} vs {prompt_embedding_table.shape[0]}."
            )
        if num_virtual_tokens > self.hidden_state_max_prompt_embedding_table_size:
            raise ValueError(
                "The hidden-state engine does not have enough prompt-embedding capacity for this image. "
                f"Required {num_virtual_tokens}, built with {self.hidden_state_max_prompt_embedding_table_size}."
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
        prompt_embedding_table: torch.Tensor | None = None,
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

    def get_text_last_hidden_state(self, prompt: str) -> torch.Tensor:
        model_inputs = self.tokenizer(
            [prompt],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hidden_state_max_input_len,
        )
        return self._run_hidden_state_engine(
            [model_inputs["input_ids"][0]],
            attention_mask=model_inputs["attention_mask"],
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

        num_patches = int(visual_features.shape[0])
        query = self._build_query(question, num_patches=num_patches)
        model_inputs = self.tokenizer(
            [query],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.hidden_state_max_input_len,
        )
        prompt_embedding_table = visual_features.reshape(-1, visual_features.shape[-1]).contiguous()
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
            outputs.append(
                self.get_multimodal_last_hidden_state(
                    question=question,
                    pixel_values=sample_pixel_values,
                )
            )

        if patch_offset != pixel_values.shape[0]:
            raise ValueError(
                "The provided num_patches_list does not match the number of image patches. "
                f"Consumed {patch_offset}, but received {pixel_values.shape[0]}."
            )

        return SimpleNamespace(hidden_states=(torch.cat(outputs, dim=0),))


ReCogDriveVLMTRT = InternVLChatTRT

__all__ = [
    "IMG_CONTEXT_TOKEN",
    "IMG_END_TOKEN",
    "IMG_START_TOKEN",
    "InternVLChatTRT",
    "Preprocess",
    "ReCogDriveVLMTRT",
]
