from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import tensorrt as trt
import torch
from transformers.feature_extraction_utils import BatchFeature

MODEL_OUTPUT_ROOT = Path(
    os.getenv(
        "RECOGDRIVE_MODEL_OUTPUT_ROOT",
        "/workspaces/safe-copilot/models",
    )
)
ROOT = MODEL_OUTPUT_ROOT / "recogdrive"
DIFFUSION_DIR = ROOT / "diffusion_trt"
DIFFUSION_PLAN = DIFFUSION_DIR / "diffusion_planner.plan"
DIFFUSION_METADATA = DIFFUSION_DIR / "metadata.json"
DIFFUSION_ENGINE_INTERFACE_VERSION = 3
REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_MODELS_ROOT = REPO_ROOT / "examples" / "models"
DEFAULT_DIFFUSION_CHECKPOINT = (
    REPO_ROOT
    / "examples"
    / "models"
    / "diffusion_planner"
    / "models"
    / "ReCogDrive_Diffusion_Planner_2B_RL.ckpt"
)

if str(EXAMPLES_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_MODELS_ROOT))


def _torch_dtype_from_trt(dtype: trt.DataType) -> torch.dtype:
    mapping = {
        trt.DataType.FLOAT: torch.float32,
        trt.DataType.HALF: torch.float16,
        trt.DataType.INT32: torch.int32,
        trt.DataType.INT8: torch.int8,
        trt.DataType.BOOL: torch.bool,
    }
    if hasattr(trt.DataType, "BF16"):
        mapping[trt.DataType.BF16] = torch.bfloat16
    try:
        return mapping[dtype]
    except KeyError as exc:
        raise TypeError(f"Unsupported TensorRT dtype for diffusion runtime: {dtype}") from exc


class ReCogDriveDiffusionPlannerTRT:
    def __init__(
        self,
        *,
        device: str = "cuda:0",
        engine_path: str | Path | None = None,
        metadata_path: str | Path | None = None,
        planner: torch.nn.Module | None = None,
    ) -> None:
        self.device = torch.device(device)
        self.engine_path = Path(engine_path or DIFFUSION_PLAN)
        self.metadata_path = Path(metadata_path or DIFFUSION_METADATA)

        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"Diffusion TRT engine was not found at {self.engine_path}. "
                "Build it with examples/models/recogdrive/convert_checkpoint.py first."
            )

        if metadata_path is None and not self.metadata_path.exists():
            adjacent_metadata_path = self.engine_path.with_suffix(".metadata.json")
            if adjacent_metadata_path.exists():
                self.metadata_path = adjacent_metadata_path

        metadata = {}
        if self.metadata_path.exists():
            metadata = json.loads(self.metadata_path.read_text())
        if metadata.get("engine_interface_version") != DIFFUSION_ENGINE_INTERFACE_VERSION:
            raise RuntimeError(
                "Diffusion TRT engine interface mismatch. "
                f"Expected version {DIFFUSION_ENGINE_INTERFACE_VERSION}, got "
                f"{metadata.get('engine_interface_version')!r}. Rebuild the engine with "
                "examples/models/recogdrive/convert_checkpoint.py."
            )

        self.action_horizon = int(metadata.get("action_horizon", 8))
        self.action_dim = int(metadata.get("action_dim", 3))
        self.his_traj_dim = int(metadata.get("his_traj_dim", 12))
        self.status_feature_dim = int(metadata.get("status_feature_dim", 8))
        self.feature_dim = int(metadata.get("feature_dim", 1536))
        self.input_embedding_dim = int(metadata.get("input_embedding_dim", 384))
        self.sampling_method = str(metadata.get("sampling_method", "ddim"))
        self.checkpoint_path = Path(
            metadata.get(
                "checkpoint_path",
                os.getenv("RECOGDRIVE_DIFFUSION_CHECKPOINT", str(DEFAULT_DIFFUSION_CHECKPOINT)),
            )
        ).expanduser()
        self._source_planner = planner
        self._planner = self._load_planner()

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT diffusion engine from {self.engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context for diffusion planner")

    def _load_planner(self) -> torch.nn.Module:
        if self._source_planner is None:
            from recogdrive.recogdrive_vlm_engines import _load_diffusion_planner_from_checkpoint

            planner, _ = _load_diffusion_planner_from_checkpoint(
                self.checkpoint_path,
                sampling_method=self.sampling_method,
            )
        else:
            planner = self._source_planner.to(self.device)
            planner_sampling_method = getattr(getattr(planner, "config", None), "sampling_method", None)
            if planner_sampling_method is not None and planner_sampling_method != self.sampling_method:
                raise ValueError(
                    "Diffusion TRT metadata sampling_method does not match the provided planner: "
                    f"{self.sampling_method!r} != {planner_sampling_method!r}"
                )

        planner.eval().requires_grad_(False)
        if next(planner.parameters()).dtype != torch.float16:
            planner = planner.half()

        # These modules live inside the TRT denoising-step engine. Dropping the
        # PyTorch copies keeps the runtime from carrying two full DiT graphs.
        planner.action_encoder = torch.nn.Identity()
        planner.action_decoder = torch.nn.Identity()
        planner.fusion_projector = torch.nn.Identity()
        planner.model = torch.nn.Identity()
        if hasattr(planner, "position_embedding"):
            planner.position_embedding = torch.nn.Identity()
        torch.cuda.empty_cache()
        return planner

    def _prepare_inputs(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        init_actions: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if vl_features.dim() == 2:
            vl_features = vl_features.unsqueeze(0)
        if vl_features.dim() != 3:
            raise ValueError("vl_features must have shape [batch, seq, hidden] or [seq, hidden]")
        if vl_features.shape[-1] != self.feature_dim:
            raise ValueError(
                f"vl_features hidden size mismatch: expected {self.feature_dim}, got {vl_features.shape[-1]}"
            )
        vl_features = vl_features.to(device=self.device, dtype=torch.float16).contiguous()

        his_traj = action_input.his_traj
        if his_traj.dim() == 1:
            his_traj = his_traj.unsqueeze(0)
        if his_traj.dim() != 2 or his_traj.shape[-1] != self.his_traj_dim:
            raise ValueError(
                f"his_traj must have shape [batch, {self.his_traj_dim}], got {tuple(his_traj.shape)}"
            )
        his_traj = his_traj.to(device=self.device, dtype=torch.float16).contiguous()

        status_feature = action_input.status_feature
        if status_feature.dim() == 1:
            status_feature = status_feature.unsqueeze(0)
        if status_feature.dim() != 2 or status_feature.shape[-1] != self.status_feature_dim:
            raise ValueError(
                "status_feature must have shape "
                f"[batch, {self.status_feature_dim}], got {tuple(status_feature.shape)}"
            )
        status_feature = status_feature.to(device=self.device, dtype=torch.float16).contiguous()

        batch_size = vl_features.shape[0]
        if his_traj.shape[0] != batch_size or status_feature.shape[0] != batch_size:
            raise ValueError("vl_features, his_traj, and status_feature must have the same batch size")

        if init_actions is None:
            init_actions = torch.randn(
                batch_size,
                self.action_horizon,
                self.action_dim,
                device=self.device,
                dtype=torch.float16,
            )
        else:
            if init_actions.dim() == 2:
                init_actions = init_actions.unsqueeze(0)
            if init_actions.shape != (batch_size, self.action_horizon, self.action_dim):
                raise ValueError(
                    "init_actions must have shape "
                    f"({batch_size}, {self.action_horizon}, {self.action_dim}), got {tuple(init_actions.shape)}"
                )
            init_actions = init_actions.to(device=self.device, dtype=torch.float16).contiguous()

        return vl_features, his_traj, status_feature, init_actions

    def _encode_conditioning(
        self,
        vl_features: torch.Tensor,
        his_traj: torch.Tensor,
        status_feature: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        planner = self._planner
        vl_embeds = planner.feature_encoder(vl_features).contiguous()
        history_embeds = planner.his_traj_encoder(his_traj.unsqueeze(1)).repeat(
            1,
            self.action_horizon,
            1,
        ).contiguous()
        ego_embeds = planner.ego_status_encoder(status_feature).contiguous()
        return vl_embeds, history_embeds, ego_embeds

    def _run_denoising_step(
        self,
        current_actions: torch.Tensor,
        vl_embeds: torch.Tensor,
        history_embeds: torch.Tensor,
        ego_embeds: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        current_actions = current_actions.to(device=self.device, dtype=torch.float16).contiguous()
        vl_embeds = vl_embeds.to(device=self.device, dtype=torch.float16).contiguous()
        history_embeds = history_embeds.to(device=self.device, dtype=torch.float16).contiguous()
        ego_embeds = ego_embeds.to(device=self.device, dtype=torch.float16).contiguous()
        timesteps = timesteps.to(device=self.device, dtype=torch.int32).contiguous()

        self._context.set_input_shape("current_actions", tuple(current_actions.shape))
        self._context.set_input_shape("vl_embeds", tuple(vl_embeds.shape))
        self._context.set_input_shape("history_embeds", tuple(history_embeds.shape))
        self._context.set_input_shape("ego_embeds", tuple(ego_embeds.shape))
        self._context.set_input_shape("timesteps", tuple(timesteps.shape))
        output_shape = tuple(self._context.get_tensor_shape("model_prediction"))
        output_dtype = _torch_dtype_from_trt(self._engine.get_tensor_dtype("model_prediction"))
        model_prediction = torch.empty(output_shape, device=self.device, dtype=output_dtype)

        self._context.set_tensor_address("current_actions", current_actions.data_ptr())
        self._context.set_tensor_address("vl_embeds", vl_embeds.data_ptr())
        self._context.set_tensor_address("history_embeds", history_embeds.data_ptr())
        self._context.set_tensor_address("ego_embeds", ego_embeds.data_ptr())
        self._context.set_tensor_address("timesteps", timesteps.data_ptr())
        self._context.set_tensor_address("model_prediction", model_prediction.data_ptr())

        ok = self._context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT diffusion denoising-step execution failed")
        return model_prediction

    def _ddim_mean(
        self,
        current_actions: torch.Tensor,
        model_prediction: torch.Tensor,
        schedule_index: torch.Tensor,
    ) -> torch.Tensor:
        planner = self._planner
        x = current_actions.to(dtype=torch.float16)
        pred_noise = model_prediction.to(dtype=x.dtype)
        alpha_t = planner.extract(planner.ddim_alphas, schedule_index, x.shape)
        sqrt_one_minus_alpha_t = planner.extract(planner.ddim_sqrt_one_minus_alphas, schedule_index, x.shape)
        x_recon = (x - sqrt_one_minus_alpha_t * pred_noise) / (alpha_t**0.5)

        denoised_clip_value = getattr(planner, "denoised_clip_value", 1.0)
        x_recon.clamp_(-denoised_clip_value, denoised_clip_value)

        alpha_prev = planner.extract(planner.ddim_alphas_prev, schedule_index, x.shape)
        pred_noise = (x - (alpha_t**0.5) * x_recon) / sqrt_one_minus_alpha_t
        eps_clip_value = getattr(planner, "eps_clip_value", None)
        if eps_clip_value is not None:
            pred_noise.clamp_(-eps_clip_value, eps_clip_value)

        pred_dir_xt = (1.0 - alpha_prev).clamp(min=0).sqrt() * pred_noise
        return (alpha_prev**0.5) * x_recon + pred_dir_xt

    def get_action(
        self,
        vl_features: torch.Tensor,
        action_input: BatchFeature,
        *,
        init_actions: torch.Tensor | None = None,
        deterministic: bool = True,
    ) -> BatchFeature:
        if not deterministic:
            raise NotImplementedError(
                "The TensorRT diffusion planner runtime currently supports deterministic inference only. "
                "Pass deterministic=True and provide init_actions if you need a specific starting trajectory."
            )

        vl_features, his_traj, status_feature, init_actions = self._prepare_inputs(
            vl_features,
            action_input,
            init_actions,
        )
        torch.cuda.synchronize(self.device)

        with torch.inference_mode():
            vl_embeds, history_embeds, ego_embeds = self._encode_conditioning(
                vl_features,
                his_traj,
                status_feature,
            )
            current_actions = init_actions
            batch_size = current_actions.shape[0]

            if self.sampling_method == "flow":
                dt = 1.0 / self._planner.config.num_inference_steps
                for step in range(self._planner.config.num_inference_steps):
                    bucket_index = int(
                        step
                        / self._planner.config.num_inference_steps
                        * self._planner.config.flow_cfg.num_timestep_buckets
                    )
                    timesteps = torch.full((batch_size,), bucket_index, device=self.device, dtype=torch.int32)
                    model_prediction = self._run_denoising_step(
                        current_actions,
                        vl_embeds,
                        history_embeds,
                        ego_embeds,
                        timesteps,
                    )
                    if self._planner.config.flow_cfg.mean_variance_net:
                        model_prediction = model_prediction.chunk(2, dim=-1)[0]
                    current_actions = current_actions + dt * model_prediction.to(dtype=current_actions.dtype)
            elif self.sampling_method == "ddim":
                for step_index in range(self._planner.ddim_steps):
                    timestep_value = int(self._planner.ddim_t[step_index].item())
                    timesteps = torch.full((batch_size,), timestep_value, device=self.device, dtype=torch.int32)
                    model_prediction = self._run_denoising_step(
                        current_actions,
                        vl_embeds,
                        history_embeds,
                        ego_embeds,
                        timesteps,
                    )
                    schedule_index = self._planner.make_timesteps(batch_size, step_index, self.device)
                    current_actions = self._ddim_mean(current_actions, model_prediction, schedule_index)
            else:
                raise NotImplementedError(
                    "The TensorRT diffusion runtime supports only 'ddim' and 'flow' sampling."
                )

            final_action_clip_value = getattr(self._planner, "final_action_clip_value", 1.0)
            if final_action_clip_value is not None:
                current_actions.clamp_(-final_action_clip_value, final_action_clip_value)
            pred_traj = self._planner.denorm_odo(current_actions)

        torch.cuda.synchronize(self.device)
        return BatchFeature(data={"pred_traj": pred_traj.float()})
