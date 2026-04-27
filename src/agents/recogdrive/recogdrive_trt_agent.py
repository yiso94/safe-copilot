from __future__ import annotations

from pathlib import Path
from typing import Optional

from transformers.feature_extraction_utils import BatchFeature
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from .recogdrive_agent import ReCogDriveAgent
from .recogdrive_diffusion_planner_trt import ReCogDriveDiffusionPlannerTRT
from .recogdrive_trt_backbone import RecogDriveTRTBackbone


class ReCogDriveTRTAgent(ReCogDriveAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        vlm_path: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        cam_type: Optional[str] = 'single',
        vlm_type: Optional[str] = 'internvl',
        dit_type: Optional[str] = 'small',
        sampling_method: Optional[str] = 'ddim',
        cache_mode: bool = False,
        cache_hidden_state: bool = False,
        lr: float = 1e-4,
        grpo: bool = False,
        metric_cache_path: Optional[str] = '',
        reference_policy_checkpoint: Optional[str] = '',
        vlm_size: Optional[str] = 'small',
        train_backbone: bool = False,
        vision_engine_path: str | Path | None = None,
        hidden_state_engine_dir: str | Path | None = None,
        llm_dir: str | Path | None = None,
        hidden_state_max_input_len: int = 2800,
        hidden_state_max_prompt_embedding_table_size: int = 3328,
        hidden_state_remove_input_padding: bool = False,
        hidden_state_gpt_attention_plugin: str | None = None,
        diffusion_engine_path: str | Path | None = None,
        diffusion_metadata_path: str | Path | None = None,
    ):
        self.trt_vision_engine_path = vision_engine_path
        self.trt_hidden_state_engine_dir = hidden_state_engine_dir
        self.trt_llm_dir = llm_dir
        self.trt_hidden_state_max_input_len = hidden_state_max_input_len
        self.trt_hidden_state_max_prompt_embedding_table_size = hidden_state_max_prompt_embedding_table_size
        self.trt_hidden_state_remove_input_padding = hidden_state_remove_input_padding
        self.trt_hidden_state_gpt_attention_plugin = hidden_state_gpt_attention_plugin
        self.trt_diffusion_engine_path = diffusion_engine_path
        self.trt_diffusion_metadata_path = diffusion_metadata_path
        self._action_head_trt: ReCogDriveDiffusionPlannerTRT | None = None
        super().__init__(
            trajectory_sampling=trajectory_sampling,
            vlm_path=vlm_path,
            checkpoint_path=checkpoint_path,
            cam_type=cam_type,
            vlm_type=vlm_type,
            dit_type=dit_type,
            sampling_method=sampling_method,
            cache_mode=cache_mode,
            cache_hidden_state=cache_hidden_state,
            lr=lr,
            grpo=grpo,
            metric_cache_path=metric_cache_path,
            reference_policy_checkpoint=reference_policy_checkpoint,
            vlm_size=vlm_size,
            train_backbone=train_backbone,
        )

    def _validate_backbone_configuration(self) -> None:
        if self.vlm_type and self.vlm_type.lower() != "internvl":
            raise ValueError(
                "ReCogDriveTRTAgent currently supports only the InternVL-style ReCogDrive backbone, "
                f"but received vlm_type={self.vlm_type!r}."
            )

    def _build_backbone(self, device: str) -> RecogDriveTRTBackbone:
        return RecogDriveTRTBackbone(
            device=device,
            llm_dir=self.trt_llm_dir,
            vision_engine_path=self.trt_vision_engine_path,
            hidden_state_engine_dir=self.trt_hidden_state_engine_dir,
            hidden_state_max_input_len=self.trt_hidden_state_max_input_len,
            hidden_state_max_prompt_embedding_table_size=self.trt_hidden_state_max_prompt_embedding_table_size,
            hidden_state_remove_input_padding=self.trt_hidden_state_remove_input_padding,
            hidden_state_gpt_attention_plugin=self.trt_hidden_state_gpt_attention_plugin,
        )

    def forward(
        self,
        features,
        targets=None,
        tokens_list=None,
        *,
        deterministic: bool = True,
        init_actions=None,
    ):
        return super().forward(
            features,
            targets=targets,
            tokens_list=tokens_list,
            deterministic=deterministic,
            init_actions=init_actions,
        )

    def _get_action_head_trt(self) -> ReCogDriveDiffusionPlannerTRT:
        if self._action_head_trt is None:
            self._action_head_trt = ReCogDriveDiffusionPlannerTRT(
                device=self.device,
                engine_path=self.trt_diffusion_engine_path,
                metadata_path=self.trt_diffusion_metadata_path,
            )
        return self._action_head_trt

    def _run_inference_action_head(
        self,
        last_hidden_state,
        action_inputs: BatchFeature,
        *,
        deterministic: bool = True,
        init_actions=None,
    ) -> BatchFeature:
        return self._get_action_head_trt().get_action(
            last_hidden_state,
            action_inputs,
            init_actions=init_actions,
            deterministic=deterministic,
        )
