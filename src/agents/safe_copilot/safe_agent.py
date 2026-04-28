import os
from pathlib import Path

import torch
import torch.optim as optim
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from omegaconf import DictConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.feature_extraction_utils import BatchFeature

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import AgentInput, SensorConfig, Trajectory
from navsim.planning.training.abstract_feature_target_builder import AbstractFeatureBuilder, AbstractTargetBuilder

from ..recogdrive.recogdrive_diffusion_planner_trt import ReCogDriveDiffusionPlannerTRT
from .recogdrive_diffusion_planner import (
    ReCogDriveDiffusionPlanner,
    ReCogDriveDiffusionPlannerConfig,
)
from .recogdrive_features import ReCogDriveFeatureBuilder, TrajectoryTargetBuilder
from .safe_backbone import SAFeCopilotBackbone
from .utils.internvl_preprocess import load_image
from .utils.lr_scheduler import WarmupCosLR
from .utils.utils import build_from_configs, format_number

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_VISION_ENGINE_PATH = REPO_ROOT / "models" / "vision_projector_bf16.plan"
DEFAULT_QWEN_ENGINE_DIR = REPO_ROOT / "models" / "qwen_bf16"
DEFAULT_DIFFUSION_STEP_ENGINE_PATH = REPO_ROOT / "models" / "diffusion_denoising_step_fp16.plan"
DEFAULT_DIFFUSION_FULL_ENGINE_PATH = REPO_ROOT / "models" / "diffusion_full_planner_fp16.plan"
DEFAULT_DIFFUSION_ENGINE_PATH = Path(
    os.getenv("SAFE_COPILOT_DIFFUSION_ENGINE_PATH", str(DEFAULT_DIFFUSION_FULL_ENGINE_PATH))
)
DEFAULT_DIFFUSION_METADATA_PATH = DEFAULT_DIFFUSION_ENGINE_PATH.with_suffix(".metadata.json")
DEFAULT_VLM_MODEL_SOURCE = "owl10/ReCogDrive-VLM-2B"


class ReCogDriveAgent(AbstractAgent):
    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        vlm_path: str | None = None,
        checkpoint_path: str | None = None,
        cam_type: str | None = "single",
        vlm_type: str | None = "internvl",
        dit_type: str | None = "small",
        sampling_method: str | None = "ddim",
        cache_mode: bool = False,
        cache_hidden_state: bool = True,
        lr: float = 1e-4,
        grpo: bool = False,
        metric_cache_path: str | None = "",
        reference_policy_checkpoint: str | None = "",
        vlm_size: str | None = "small",
        train_backbone: bool = False,
        vision_engine_path: str | Path | None = None,
        hidden_state_engine_dir: str | Path | None = None,
        qwen_engine_dir: str | Path | None = None,
        llm_dir: str | Path | None = None,
        hidden_state_max_input_len: int = 2800,
        hidden_state_max_prompt_embedding_table_size: int = 3328,
        hidden_state_remove_input_padding: bool = False,
        hidden_state_gpt_attention_plugin: str | None = None,
        diffusion_engine_path: str | Path | None = DEFAULT_DIFFUSION_ENGINE_PATH,
        diffusion_metadata_path: str | Path | None = DEFAULT_DIFFUSION_METADATA_PATH,
        use_diffusion_trt: bool | None = None,
    ):
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self.vlm_path = vlm_path
        self.checkpoint_path = checkpoint_path
        self.vlm_type = vlm_type
        self.dit_type = dit_type
        self.cache_mode = cache_mode
        self.cache_hidden_state = cache_hidden_state
        self._lr = lr
        self.grpo = grpo
        self.backbone = None
        self.metric_cache_path = metric_cache_path
        self.reference_policy_checkpoint = reference_policy_checkpoint
        self.vlm_size = vlm_size
        self.train_backbone = train_backbone
        self.trt_vision_engine_path = (
            Path(vision_engine_path) if vision_engine_path is not None else DEFAULT_VISION_ENGINE_PATH
        )
        hidden_state_engine_dir = hidden_state_engine_dir if hidden_state_engine_dir is not None else qwen_engine_dir
        self.trt_hidden_state_engine_dir = (
            Path(hidden_state_engine_dir) if hidden_state_engine_dir is not None else DEFAULT_QWEN_ENGINE_DIR
        )
        self.trt_llm_dir = Path(llm_dir) if llm_dir is not None else None
        self.trt_hidden_state_max_input_len = hidden_state_max_input_len
        self.trt_hidden_state_max_prompt_embedding_table_size = hidden_state_max_prompt_embedding_table_size
        self.trt_hidden_state_remove_input_padding = hidden_state_remove_input_padding
        self.trt_hidden_state_gpt_attention_plugin = hidden_state_gpt_attention_plugin
        self.trt_diffusion_engine_path = Path(diffusion_engine_path) if diffusion_engine_path is not None else None
        self.trt_diffusion_metadata_path = (
            Path(diffusion_metadata_path) if diffusion_metadata_path is not None else None
        )
        self.use_diffusion_trt = use_diffusion_trt
        self._action_head_trt: ReCogDriveDiffusionPlannerTRT | None = None

        local_rank = int(os.getenv("LOCAL_RANK", "0"))
        device = f"cuda:{local_rank}"
        self.device = device
        if not self.cache_hidden_state and not self.cache_mode:
            print("Agent running in 'no-cache' mode. Initializing internal backbone.")
            if not self.vlm_type:
                raise ValueError("In 'no-cache' mode, vlm_type is required.")
            model_source = str(self.trt_llm_dir or self.vlm_path or DEFAULT_VLM_MODEL_SOURCE)
            self.backbone = SAFeCopilotBackbone(
                vit_engine_path=self.trt_vision_engine_path,
                qwen_engine_dir=self.trt_hidden_state_engine_dir,
                checkpoint_path=model_source,
                device=device,
            )

        if self.dit_type == "large":
            cfg = make_recogdrive_config(
                self.dit_type,
                action_dim=3,
                action_horizon=8,
                grpo=self.grpo,
                input_embedding_dim=1536,
                sampling_method=sampling_method,
            )
        elif self.dit_type == "small":
            cfg = make_recogdrive_config(
                self.dit_type,
                action_dim=3,
                action_horizon=8,
                grpo=self.grpo,
                input_embedding_dim=384,
                sampling_method=sampling_method,
            )

        cfg.vlm_size = self.vlm_size

        if self.grpo:
            cfg.grpo_cfg.metric_cache_path = self.metric_cache_path
            cfg.grpo_cfg.reference_policy_checkpoint = self.reference_policy_checkpoint

        self.action_head = ReCogDriveDiffusionPlanner(cfg).cuda()
        self.num_inference_samples = 1
        self.inference_selection_mode = "median"

    def name(self) -> str:
        return self.__class__.__name__

    def initialize(self) -> None:
        if self.checkpoint_path:
            ckpt = torch.load(self.checkpoint_path, map_location="cpu", weights_only=False)["state_dict"]
            model_dict = self.state_dict()
            filtered_ckpt = {}
            for k, v in ckpt.items():
                k2 = k[len("agent.") :] if k.startswith("agent.") else k
                if k2 in model_dict and v.shape == model_dict[k2].shape:
                    filtered_ckpt[k2] = v
            self.load_state_dict(filtered_ckpt, strict=False)

    def get_sensor_config(self) -> SensorConfig:
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> list[AbstractTargetBuilder]:
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> list[AbstractFeatureBuilder]:
        return [
            ReCogDriveFeatureBuilder(
                cache_hidden_state=self.cache_hidden_state,
                model_type=self.vlm_type,
                checkpoint_path=self.vlm_path,
                device=self.device,
                cache_mode=self.cache_mode,
            )
        ]

    def forward(
        self,
        features: dict[str, torch.Tensor],
        targets=None,
        tokens_list=None,
        *,
        deterministic: bool = False,
        init_actions: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        for key, tensor in features.items():
            if isinstance(tensor, torch.Tensor):
                features[key] = tensor.cuda()

        model_dtype = next(self.action_head.parameters()).dtype

        history_trajectory = features["history_trajectory"].cuda()
        high_command_one_hot = features["high_command_one_hot"].cuda()

        if history_trajectory.ndim == 2:
            history_trajectory = history_trajectory.unsqueeze(0)
        if high_command_one_hot.ndim == 1:
            high_command_one_hot = high_command_one_hot.unsqueeze(0)

        if self.cache_hidden_state:
            last_hidden_state = features["last_hidden_state"].cuda()
        else:
            if self.backbone is None:
                raise RuntimeError("Agent is in 'no-cache' mode, but backbone is not initialized.")
            image_path_tensor = features["image_path_tensor"]
            if image_path_tensor.ndim == 1:
                image_path_tensor = image_path_tensor.unsqueeze(0)
            image_paths = self._decode_paths_from_tensor(image_path_tensor)

            pixel_values_list = [load_image(path) for path in image_paths]

            num_patches_list = [p.shape[0] for p in pixel_values_list]
            pixel_values_cat = torch.cat(pixel_values_list, dim=0).cuda()

            navigation_commands = ["turn left", "go straight", "turn right"]
            command_indices = torch.argmax(high_command_one_hot, dim=-1)
            command_str_list = [navigation_commands[idx.item()] for idx in command_indices]

            questions = []
            batch_size = high_command_one_hot.shape[0]
            for i in range(batch_size):
                history_trajectory_sample = history_trajectory[i]
                command_str_sample = command_str_list[i]

                history_str = " ".join(
                    [
                        f"   - t-{3 - j}: ({format_number(history_trajectory_sample[j, 0].item())}, "
                        f"{format_number(history_trajectory_sample[j, 1].item())}, "
                        f"{format_number(history_trajectory_sample[j, 2].item())})"
                        for j in range(history_trajectory_sample.shape[0])
                    ]
                )

                prompt = (
                    "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
                    "1. Visual perception from front camera view\n"
                    f"2. Historical motion context (last 4 timesteps):{history_str}\n"
                    f"3. Active navigation command: [{command_str_sample.upper()}]"
                )
                output_requirements = (
                    "\nOutput requirements:\n- Predict 8 future trajectory points\n"
                    "- Each point format: (x:float, y:float, heading:float)\n"
                    "- Use [PT, ...] to encapsulate the trajectory\n"
                    "- Maintain numerical precision to 2 decimal places"
                )
                questions.append(f"{prompt}{output_requirements}")

            outputs = self.backbone(pixel_values_cat, questions, num_patches_list=num_patches_list)
            last_hidden_state = outputs.hidden_states[-1]

        status_feature = features["status_feature"].cuda()
        if status_feature.ndim == 1:
            status_feature = status_feature.unsqueeze(0)
        if last_hidden_state.ndim == 2:
            last_hidden_state = last_hidden_state.unsqueeze(0)

        last_hidden_state = last_hidden_state.to(model_dtype)
        history_trajectory_reshaped = history_trajectory.view(history_trajectory.size(0), -1)
        input_state = torch.cat([status_feature, history_trajectory_reshaped], dim=1)

        if self.training and not self.grpo:
            action_inputs = BatchFeature(
                data={
                    "state": input_state.to(model_dtype),
                    "his_traj": history_trajectory_reshaped.to(model_dtype),
                    "status_feature": status_feature.to(model_dtype),
                    "action": targets["trajectory"].to(model_dtype),
                }
            )
            return self.action_head(last_hidden_state, action_inputs)
        elif self.training and self.grpo:
            action_inputs = BatchFeature(
                data={
                    "state": input_state.to(model_dtype),
                    "his_traj": history_trajectory_reshaped.to(model_dtype),
                    "status_feature": status_feature.to(model_dtype),
                    "action": targets["trajectory"].to(model_dtype),
                }
            )
            return self.action_head.forward_grpo(last_hidden_state, action_inputs, tokens_list)
        else:
            action_inputs = BatchFeature(
                {
                    "state": input_state.to(model_dtype),
                    "his_traj": history_trajectory_reshaped.to(model_dtype),
                    "status_feature": status_feature.to(model_dtype),
                }
            )
            return self._run_inference_action_head(
                last_hidden_state.to(model_dtype),
                action_inputs,
                deterministic=deterministic,
                init_actions=init_actions,
            )

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))
        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)

        return Trajectory(poses)

    def compute_trajectory_vis(self, agent_input: AgentInput) -> Trajectory:
        self.eval()

        features: dict[str, torch.Tensor] = {}
        # build features
        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # add batch dimension
        features = {k: v.unsqueeze(0) for k, v in features.items()}

        with torch.no_grad():
            predictions = self.forward(features)
            poses = predictions["pred_traj"].float().cpu().squeeze(0)
        return Trajectory(poses)

    def compute_loss(
        self, features: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], predictions: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        if self.training and self.grpo:
            return predictions
        elif self.training:
            return predictions.loss
        else:
            return torch.nn.functional.l1_loss(predictions["pred_traj"], targets["trajectory"])

    def get_optimizers(self) -> Optimizer | dict[str, LRScheduler]:
        optimizer_cfg = DictConfig({"type": "AdamW", "lr": self._lr, "weight_decay": 1e-4, "betas": (0.9, 0.95)})

        params = list(self.action_head.parameters())
        if self.backbone is not None and self.train_backbone:
            params += list(self.backbone.parameters())

        optimizer = build_from_configs(optim, optimizer_cfg, params=params)

        if self.grpo:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=0.0, epochs=10, warmup_epochs=0)
        else:
            scheduler = WarmupCosLR(optimizer=optimizer, lr=self._lr, min_lr=1e-6, epochs=200, warmup_epochs=3)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def _decode_paths_from_tensor(path_tensor: torch.Tensor) -> list[str]:
        """Decodes a batch of path tensors back into a list of file path strings.

        Args:
            path_tensor (torch.Tensor): A 2D tensor of shape
                (batch_size, max_path_length) from the collate_fn.

        Returns:
            List[str]: A list of decoded file path strings.
        """
        decoded_paths = []
        for single_path_tensor in path_tensor:
            chars = []
            for code in single_path_tensor:
                code_item = code.item()
                if code_item == 0:
                    break
                chars.append(chr(code_item))
            decoded_paths.append("".join(chars))
        return decoded_paths

    def _should_use_diffusion_trt(self) -> bool:
        if self.use_diffusion_trt is not None:
            return self.use_diffusion_trt
        return self.trt_diffusion_engine_path is not None and self.trt_diffusion_engine_path.exists()

    def _get_action_head_trt(self) -> ReCogDriveDiffusionPlannerTRT:
        if self.trt_diffusion_engine_path is None:
            raise FileNotFoundError("Diffusion TRT engine path is not configured.")
        if self._action_head_trt is None:
            self._action_head_trt = ReCogDriveDiffusionPlannerTRT(
                device=self.device,
                engine_path=self.trt_diffusion_engine_path,
                metadata_path=self.trt_diffusion_metadata_path,
                planner=self.action_head,
            )
        return self._action_head_trt

    def _run_inference_action_head(
        self,
        last_hidden_state: torch.Tensor,
        action_inputs: BatchFeature,
        *,
        deterministic: bool = False,
        init_actions: torch.Tensor | None = None,
    ) -> BatchFeature:
        if init_actions is not None:
            init_actions = init_actions.to(device=last_hidden_state.device, dtype=last_hidden_state.dtype)

        if self._should_use_diffusion_trt():
            return self._get_action_head_trt().get_action(
                last_hidden_state,
                action_inputs,
                init_actions=init_actions,
                deterministic=True,
            )

        if self.use_diffusion_trt:
            raise FileNotFoundError(
                f"Diffusion TRT engine was requested but was not found at {self.trt_diffusion_engine_path}."
            )
        return self.action_head.get_action(
            last_hidden_state,
            action_inputs,
            init_actions=init_actions,
            deterministic=deterministic,
        )


class SAFeCopilotAgent(ReCogDriveAgent):
    """Hydra-facing alias that preserves the expected SAFE agent target path."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("cache_hidden_state", False)
        kwargs.setdefault("use_diffusion_trt", True)
        kwargs.setdefault("vision_engine_path", DEFAULT_VISION_ENGINE_PATH)
        kwargs.setdefault("hidden_state_engine_dir", DEFAULT_QWEN_ENGINE_DIR)
        kwargs.setdefault("diffusion_engine_path", DEFAULT_DIFFUSION_ENGINE_PATH)
        kwargs.setdefault("diffusion_metadata_path", DEFAULT_DIFFUSION_METADATA_PATH)
        super().__init__(*args, **kwargs)


class ReCogDriveAgentTRT(ReCogDriveAgent):
    """SAFE agent variant that uses TRT backbones and the diffusion step engine."""

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling | None = None,
        **kwargs,
    ):
        if trajectory_sampling is None:
            trajectory_sampling = TrajectorySampling(num_poses=8)
        kwargs.setdefault("cache_hidden_state", False)
        kwargs.setdefault("use_diffusion_trt", True)
        kwargs.setdefault("vision_engine_path", DEFAULT_VISION_ENGINE_PATH)
        kwargs.setdefault("hidden_state_engine_dir", DEFAULT_QWEN_ENGINE_DIR)
        kwargs.setdefault("diffusion_engine_path", DEFAULT_DIFFUSION_ENGINE_PATH)
        kwargs.setdefault("diffusion_metadata_path", DEFAULT_DIFFUSION_METADATA_PATH)
        kwargs.setdefault("vlm_path", str(kwargs.get("llm_dir") or DEFAULT_VLM_MODEL_SOURCE))
        super().__init__(trajectory_sampling=trajectory_sampling, **kwargs)

    def predict_pred_traj(
        self,
        features: dict[str, torch.Tensor],
        *,
        deterministic: bool = True,
        init_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.forward(
            features,
            deterministic=deterministic,
            init_actions=init_actions,
        )["pred_traj"]


def make_recogdrive_config(
    size: str,
    *,
    action_dim: int,
    action_horizon: int,
    input_embedding_dim: int,
    sampling_method: str = "ddim",
    num_inference_steps: int = 5,
    grpo: bool = False,
    model_dtype: str = "float16",
) -> ReCogDriveDiffusionPlannerConfig:
    """A factory function to create a ReCogDriveDiffusionPlannerConfig object.

    This function simplifies configuration by using a size preset ("small",
    "large", "large_new") to define the core DiT architecture, while allowing
    other important planner settings to be specified.

    Args:
        size (str): The size preset for the DiT backbone.
        action_dim (int): The dimension of the action space.
        action_horizon (int): The number of future action steps to predict.
        input_embedding_dim (int): Dimension of the input embeddings to the DiT.
        sampling_method (str): The core training and sampling methodology.
        num_inference_steps (int): Number of steps for inference sampling.
        grpo (bool): If True, enables GRPO-specific logic.
        model_dtype (str): The data type for model computations.

    Returns:
        ReCogDriveDiffusionPlannerConfig: An instantiated and configured planner config object.
    """
    size = size.lower()
    if size == "small":
        diffusion_model_cfg = {"num_heads": 8, "head_dim": 48, "num_layers": 16, "output_dim": 512}
    elif size == "large":
        diffusion_model_cfg = {"num_heads": 32, "head_dim": 48, "num_layers": 16, "output_dim": 1536}
    else:
        raise ValueError(f"Unknown model size: {size!r}")

    common_params: dict[str, any] = {
        "dropout": 0.0,
        "attention_bias": True,
        "norm_eps": 1e-5,
        "interleave_attention": True,
    }
    diffusion_model_cfg.update(common_params)

    config = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg=diffusion_model_cfg,
        action_dim=action_dim,
        action_horizon=action_horizon,
        input_embedding_dim=input_embedding_dim,
        sampling_method=sampling_method,
        num_inference_steps=num_inference_steps,
        grpo=grpo,
        model_dtype=model_dtype,
    )

    return config
