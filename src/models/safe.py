import json
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer, GenerationConfig

from examples.models.recogdrive.recogdrive_vlm_engines import (
    SNAPSHOT,
    build_language_generation_trtllm_engine,
    build_language_trtllm_engine,
    build_vision_trt_engine,
    get_language_last_hidden_state,
    get_language_last_hidden_state_from_input_ids,
    load_language_generation_runner,
    resolve_model_source,
    run_vision_trt_engine,
)
from models.conversation import get_conv_template

MODEL_DIR = Path("/workspaces/test/src/models/recogdrive_vlm_2b")
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"


class ReCogDriveVLMTRT:
    def __init__(
        self,
        model_dir: str | Path = MODEL_DIR,
        device: str = "cuda:0",
        max_input_len: int = 1024,
        max_output_len: int = 128,
        build_engines: bool = True,
    ):
        self.model_dir = Path(model_dir)
        self.device = torch.device(device)
        self.max_input_len = max_input_len
        self.max_output_len = max_output_len

        config = json.loads((self.model_dir / "config.json").read_text())
        image_size = config.get("force_image_size") or config["vision_config"]["image_size"]
        patch_size = config["vision_config"]["patch_size"]
        self.patch_size = patch_size
        self.select_layer = config["select_layer"]
        self.template = config["template"]
        self.downsample_ratio = config["downsample_ratio"]
        self.ps_version = config["ps_version"]
        self.max_dynamic_patch = config.get("max_dynamic_patch", 1)
        self.hidden_size = config["llm_config"]["hidden_size"]
        self.vocab_size = config["llm_config"]["vocab_size"]
        self.num_image_token = int((image_size // patch_size) ** 2 * (self.downsample_ratio**2))
        self.max_prompt_embedding_table_size = self.num_image_token * self.max_dynamic_patch

        self.conv_template = get_conv_template(self.template)
        self.system_message = self.conv_template.system_message
        self.tokenizer = self._load_tokenizer()
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        self._generation_runner = None
        self._generation_engine_dir = None
        self._hidden_state_engine_dir = None
        if build_engines:
            build_vision_trt_engine()
            self._generation_engine_dir = build_language_generation_trtllm_engine(
                max_input_len=self.max_input_len,
                max_output_len=self.max_output_len,
                max_prompt_embedding_table_size=self.max_prompt_embedding_table_size,
            )

    def _load_tokenizer(self):
        tokenizer_source = resolve_model_source()
        kwargs = dict(
            trust_remote_code=True,
            local_files_only=SNAPSHOT.exists(),
        )
        try:
            return AutoTokenizer.from_pretrained(
                tokenizer_source,
                fix_mistral_regex=True,
                **kwargs,
            )
        except TypeError:
            return AutoTokenizer.from_pretrained(tokenizer_source, **kwargs)

    def _get_generation_runner(self):
        if self._generation_runner is None:
            self._generation_runner = load_language_generation_runner(
                max_input_len=self.max_input_len,
                max_output_len=self.max_output_len,
                max_prompt_embedding_table_size=self.max_prompt_embedding_table_size,
            )
        return self._generation_runner

    def extract_feature(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values is None:
            raise ValueError("pixel_values must not be None")
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        pixel_values = pixel_values.to(device=self.device, dtype=torch.float16)
        return run_vision_trt_engine(pixel_values)

    def _prepare_input_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.dim() == 2:
            if input_ids.shape[0] != 1:
                raise ValueError("generate() currently supports batch size 1; use batch_chat() for multiple prompts")
            input_ids = input_ids[0]
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attention_mask = attention_mask[0]
            input_ids = input_ids[attention_mask.bool()]
        return input_ids.to(dtype=torch.int32)

    def _inject_visual_features(
        self,
        input_ids: torch.Tensor,
        visual_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if visual_features.dim() == 2:
            visual_features = visual_features.unsqueeze(0)
        if visual_features.dim() != 3:
            raise ValueError("visual_features must have shape [batch, tokens, hidden] or [tokens, hidden]")

        flat_visual_features = visual_features.reshape(-1, visual_features.shape[-1])
        input_ids = input_ids.clone()
        selected = input_ids == self.img_context_token_id
        num_selected = int(selected.sum().item())
        if num_selected == 0:
            raise ValueError("No <IMG_CONTEXT> tokens were found in input_ids")
        if num_selected != flat_visual_features.shape[0]:
            raise ValueError(
                "The number of image placeholder tokens does not match the visual feature length: "
                f"{num_selected} vs {flat_visual_features.shape[0]}"
            )

        fake_prompt_ids = torch.arange(
            self.vocab_size,
            self.vocab_size + num_selected,
            device=input_ids.device,
            dtype=torch.int32,
        )
        input_ids[selected] = fake_prompt_ids
        prompt_table = flat_visual_features.unsqueeze(0).contiguous().to(dtype=torch.float16)
        return input_ids, prompt_table

    @staticmethod
    def _coerce_generation_config(generation_config: GenerationConfig | dict | None) -> dict[str, Any]:
        if generation_config is None:
            return {}
        if isinstance(generation_config, GenerationConfig):
            return generation_config.to_dict()
        if isinstance(generation_config, dict):
            return dict(generation_config)
        raise TypeError(f"Unsupported generation_config type: {type(generation_config)}")

    def _runner_kwargs(
        self, generation_config: GenerationConfig | dict | None, generate_kwargs: dict[str, Any]
    ) -> dict[str, Any]:
        config_dict = self._coerce_generation_config(generation_config)
        merged = {**config_dict, **generate_kwargs}

        do_sample = bool(merged.get("do_sample", False))
        temperature = float(merged.get("temperature", 1.0))
        top_k = int(merged.get("top_k", 50 if do_sample else 1))
        top_p = float(merged.get("top_p", 1.0))
        if not do_sample:
            temperature = 1.0
            top_k = 1
            top_p = 1.0

        end_id = merged.get("eos_token_id", self.tokenizer.eos_token_id)
        if isinstance(end_id, list):
            end_id = end_id[0]
        pad_id = merged.get("pad_token_id", self.tokenizer.pad_token_id)
        if pad_id is None:
            pad_id = end_id

        return {
            "max_new_tokens": int(merged.get("max_new_tokens", self.max_output_len)),
            "num_beams": int(merged.get("num_beams", 1)),
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "repetition_penalty": float(merged.get("repetition_penalty", 1.0)),
            "end_id": int(end_id),
            "pad_id": int(pad_id),
            "return_dict": False,
        }

    @torch.no_grad()
    def generate(
        self,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
        generation_config: GenerationConfig | dict | None = None,
        output_hidden_states: bool | None = None,
        **generate_kwargs,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("input_ids must not be None")
        if output_hidden_states:
            raise NotImplementedError(
                "TensorRT generation does not return hidden states here. "
                "Use get_text_last_hidden_state() for text-only hidden states."
            )

        runner = self._get_generation_runner()
        prepared_input_ids = self._prepare_input_ids(input_ids, attention_mask).to(self.device)
        prompt_table = None
        prompt_tasks = None

        if pixel_values is not None or visual_features is not None:
            if visual_features is None:
                visual_features = self.extract_feature(pixel_values)
            prepared_input_ids, prompt_table = self._inject_visual_features(prepared_input_ids, visual_features)
            prompt_tasks = "0"

        outputs = runner.generate(
            batch_input_ids=[prepared_input_ids],
            prompt_table=prompt_table,
            prompt_tasks=prompt_tasks,
            **self._runner_kwargs(generation_config, generate_kwargs),
        )
        if outputs.dim() == 3:
            return outputs[:, 0, :]
        return outputs

    def get_text_last_hidden_state(self, prompt: str) -> torch.Tensor:
        self._hidden_state_engine_dir = build_language_trtllm_engine(self.max_input_len)
        return get_language_last_hidden_state(prompt, self.max_input_len)

    def get_multimodal_last_hidden_state(
        self,
        *,
        pixel_values: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        visual_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids is None:
            raise ValueError("input_ids must not be None")

        prepared_input_ids = self._prepare_input_ids(input_ids, attention_mask).to(self.device)
        prompt_table = None
        prompt_tasks = None
        max_prompt_embedding_table_size = 0

        if pixel_values is not None or visual_features is not None:
            if visual_features is None:
                visual_features = self.extract_feature(pixel_values)
            prepared_input_ids, prompt_table = self._inject_visual_features(prepared_input_ids, visual_features)
            prompt_tasks = "0"
            max_prompt_embedding_table_size = self.max_prompt_embedding_table_size

        self._hidden_state_engine_dir = build_language_trtllm_engine(
            self.max_input_len,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        )
        return get_language_last_hidden_state_from_input_ids(
            [prepared_input_ids],
            max_input_len=self.max_input_len,
            prompt_embedding_table=prompt_table,
            prompt_tasks=prompt_tasks,
            max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        )

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN=IMG_START_TOKEN,
        IMG_END_TOKEN=IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN=IMG_CONTEXT_TOKEN,
        verbose=False,
    ):
        if history is None and pixel_values is not None and "<image>" not in question:
            question = "<image>\n" + question

        if num_patches_list is None:
            num_patches_list = [pixel_values.shape[0]] if pixel_values is not None else []
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        self.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            print(f"dynamic ViT batch size: {pixel_values.shape[0]}")

        for num_patches in num_patches_list:
            image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches + IMG_END_TOKEN
            query = query.replace("<image>", image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors="pt")
        input_ids = model_inputs["input_ids"].to(self.device)
        attention_mask = model_inputs.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )
        prompt_length = int(attention_mask[0].sum().item()) if attention_mask is not None else input_ids.shape[1]
        response_tokens = generation_output[:, prompt_length:]
        response = tokenizer.batch_decode(response_tokens, skip_special_tokens=True)[0]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        if verbose:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, "")
            query_to_print = query_to_print.replace(f"{IMG_START_TOKEN}{IMG_END_TOKEN}", "<image>")
            print(query_to_print, response)
        return response

    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN=IMG_START_TOKEN,
        IMG_END_TOKEN=IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN=IMG_CONTEXT_TOKEN,
        verbose=False,
        image_counts=None,
    ):
        if history is not None or return_history:
            raise NotImplementedError("Now multi-turn chat is not supported in batch_chat.")
        if image_counts is not None:
            num_patches_list = image_counts
        if num_patches_list is None:
            if pixel_values is None:
                num_patches_list = [0 for _ in questions]
            else:
                raise ValueError("num_patches_list is required when pixel_values is provided")

        responses = []
        offset = 0
        for question, num_patches in zip(questions, num_patches_list):
            sample_pixels = None
            if pixel_values is not None:
                sample_pixels = pixel_values[offset : offset + num_patches]
                offset += num_patches
            response = self.chat(
                tokenizer=tokenizer,
                pixel_values=sample_pixels,
                question=question,
                generation_config=generation_config,
                history=None,
                return_history=False,
                num_patches_list=[num_patches] if sample_pixels is not None else [],
                IMG_START_TOKEN=IMG_START_TOKEN,
                IMG_END_TOKEN=IMG_END_TOKEN,
                IMG_CONTEXT_TOKEN=IMG_CONTEXT_TOKEN,
                verbose=verbose,
            )
            responses.append(response)
        return responses

    def forward(self, *args, **kwargs):
        raise NotImplementedError(
            "The TensorRT runtime wrapper is intended for inference helpers such as "
            "extract_feature(), generate(), chat(), batch_chat(), and get_text_last_hidden_state(). "
            "Training-style forward/logits output is not implemented."
        )


if __name__ == "__main__":
    model = ReCogDriveVLMTRT()
    print(model._generation_engine_dir)
