from __future__ import annotations

import json
import os
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
    ) -> None:
        self.device = torch.device(device)
        self.engine_path = Path(engine_path or DIFFUSION_PLAN)
        self.metadata_path = Path(metadata_path or DIFFUSION_METADATA)

        if not self.engine_path.exists():
            raise FileNotFoundError(
                f"Diffusion TRT engine was not found at {self.engine_path}. "
                "Build it with examples/models/recogdrive/convert_checkpoint.py first."
            )

        metadata = {}
        if self.metadata_path.exists():
            metadata = json.loads(self.metadata_path.read_text())

        self.action_horizon = int(metadata.get("action_horizon", 8))
        self.action_dim = int(metadata.get("action_dim", 3))
        self.his_traj_dim = int(metadata.get("his_traj_dim", 12))
        self.status_feature_dim = int(metadata.get("status_feature_dim", 8))
        self.feature_dim = int(metadata.get("feature_dim", 1536))

        self._logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._logger)
        self._engine = self._runtime.deserialize_cuda_engine(self.engine_path.read_bytes())
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TensorRT diffusion engine from {self.engine_path}")
        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError("Failed to create TensorRT execution context for diffusion planner")

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

        self._context.set_input_shape("vl_features", tuple(vl_features.shape))
        self._context.set_input_shape("his_traj", tuple(his_traj.shape))
        self._context.set_input_shape("status_feature", tuple(status_feature.shape))
        self._context.set_input_shape("init_actions", tuple(init_actions.shape))
        output_shape = tuple(self._context.get_tensor_shape("pred_traj"))
        output_dtype = _torch_dtype_from_trt(self._engine.get_tensor_dtype("pred_traj"))
        output = torch.empty(output_shape, device=self.device, dtype=output_dtype)

        self._context.set_tensor_address("vl_features", vl_features.data_ptr())
        self._context.set_tensor_address("his_traj", his_traj.data_ptr())
        self._context.set_tensor_address("status_feature", status_feature.data_ptr())
        self._context.set_tensor_address("init_actions", init_actions.data_ptr())
        self._context.set_tensor_address("pred_traj", output.data_ptr())

        ok = self._context.execute_async_v3(torch.cuda.current_stream(self.device).cuda_stream)
        if not ok:
            raise RuntimeError("TensorRT diffusion planner execution failed")
        torch.cuda.synchronize(self.device)
        return BatchFeature(data={"pred_traj": output.float()})
