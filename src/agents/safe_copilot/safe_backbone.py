from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import torch

from models.internvl_chat import InternVLChatTRT, resolve_model_source


class SAFeCopilotBackbone(torch.nn.Module):
    def __init__(
        self,
        *,
        vit_engine_path: str | Path,
        qwen_engine_dir: str | Path,
        checkpoint_path: str | None = None,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()
        model_source = checkpoint_path or resolve_model_source()
        self.device = device
        self.vit_engine_path = Path(vit_engine_path)
        self.qwen_engine_dir = Path(qwen_engine_dir)
        self.runtime = InternVLChatTRT(
            device=device,
            tokenizer_dir=model_source,
            vision_engine_path=self.vit_engine_path,
            qwen_engine_dir=self.qwen_engine_dir,
        )
        self.tokenizer = self.runtime.tokenizer
        self.num_image_token = self.runtime.num_image_token

    def close(self) -> None:
        self.runtime.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def forward(
        self,
        pixel_values: torch.Tensor | None,
        questions: list[str],
        num_patches_list: list[int] | None,
    ):
        if pixel_values is not None and num_patches_list is None:
            raise ValueError("num_patches_list is required when pixel_values is provided")
        if num_patches_list is not None and len(questions) != len(num_patches_list):
            raise ValueError(
                "questions and num_patches_list must have the same length, "
                f"but received {len(questions)} and {len(num_patches_list)}."
            )
        if pixel_values is None:
            return self.runtime(pixel_values, questions, num_patches_list)

        outputs = []
        patch_offset = 0
        for question, num_patches in zip(questions, num_patches_list):
            sample_pixel_values = pixel_values[patch_offset : patch_offset + num_patches]
            patch_offset += num_patches
            hidden_state = self.runtime.get_multimodal_last_hidden_state(
                question=question,
                pixel_values=sample_pixel_values,
            )
            if self.runtime.qwen_model.decoder.remove_input_padding:
                hidden_state = self._restore_padded_layout(
                    hidden_state,
                    question=question,
                    num_patches=num_patches,
                )
            outputs.append(hidden_state)

        if patch_offset != pixel_values.shape[0]:
            raise ValueError(
                "The provided num_patches_list does not match the number of image patches. "
                f"Consumed {patch_offset}, but received {pixel_values.shape[0]}."
            )

        return SimpleNamespace(hidden_states=(torch.cat(outputs, dim=0),))

    def _restore_padded_layout(
        self,
        hidden_state: torch.Tensor,
        *,
        question: str,
        num_patches: int,
    ) -> torch.Tensor:
        query = self.runtime.make_context(question, num_patches)
        model_inputs = self.tokenizer(
            [query],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=self.runtime.max_input_len,
        )
        attention_mask = model_inputs["attention_mask"].to(device=hidden_state.device, dtype=torch.bool)
        num_valid_tokens = int(attention_mask.sum().item())
        if hidden_state.shape[1] != num_valid_tokens:
            raise ValueError(
                "Packed Qwen TRT output length does not match the number of valid prompt tokens: "
                f"{hidden_state.shape[1]} vs {num_valid_tokens}."
            )

        padded_hidden_state = torch.zeros(
            (hidden_state.shape[0], self.runtime.max_input_len, hidden_state.shape[-1]),
            dtype=hidden_state.dtype,
            device=hidden_state.device,
        )
        padded_hidden_state[0, attention_mask[0]] = hidden_state[0]
        return padded_hidden_state
