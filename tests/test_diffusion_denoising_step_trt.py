import json
import os
import sys
from pathlib import Path

import pytest
import torch

trt = pytest.importorskip("tensorrt")

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXAMPLES_MODELS_ROOT = REPO_ROOT / "examples" / "models"
if str(EXAMPLES_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_MODELS_ROOT))

from recogdrive import recogdrive_vlm_engines as rv  # noqa: E402

PLAN_PATH = Path(
    os.getenv(
        "DIFFUSION_DENOISING_STEP_PLAN",
        str(REPO_ROOT / "models" / "diffusion_denoising_step_fp16.plan"),
    )
)
METADATA_PATH = Path(
    os.getenv(
        "DIFFUSION_DENOISING_STEP_METADATA",
        str(PLAN_PATH.with_suffix(".metadata.json")),
    )
)
ATOL = float(os.getenv("DIFFUSION_DENOISING_STEP_ATOL", "5e-2"))
RTOL = float(os.getenv("DIFFUSION_DENOISING_STEP_RTOL", "5e-2"))

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA is required to compare the diffusion denoising-step TRT engine",
)


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
    return mapping[dtype]


def _require_denoising_step_engine() -> tuple[Path, dict]:
    if not PLAN_PATH.exists():
        pytest.skip(
            "Diffusion denoising-step TRT plan is missing. Build it with "
            "`examples/models/diffusion_planner/diffusion_onnx_trt.py` first."
        )
    if not METADATA_PATH.exists():
        pytest.skip(f"Diffusion denoising-step metadata is missing: {METADATA_PATH}")

    metadata = json.loads(METADATA_PATH.read_text())
    assert metadata.get("engine_interface_version") == rv.DIFFUSION_ENGINE_INTERFACE_VERSION

    checkpoint_path = Path(metadata["checkpoint_path"])
    if not checkpoint_path.exists():
        pytest.skip(f"Diffusion planner checkpoint is missing: {checkpoint_path}")
    return PLAN_PATH, metadata


def _make_inputs(metadata: dict) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    max_vl_seq_len = int(metadata.get("max_vl_seq_len", 2800))
    vl_seq_len = max(1, min(max_vl_seq_len, 32))
    action_horizon = int(metadata["action_horizon"])
    action_dim = int(metadata["action_dim"])
    input_embedding_dim = int(metadata["input_embedding_dim"])

    return {
        "current_actions": torch.randn(
            1,
            action_horizon,
            action_dim,
            device="cuda",
            dtype=torch.float16,
        ),
        "vl_embeds": torch.randn(
            1,
            vl_seq_len,
            input_embedding_dim,
            device="cuda",
            dtype=torch.float16,
        ),
        "history_embeds": torch.randn(
            1,
            action_horizon,
            input_embedding_dim,
            device="cuda",
            dtype=torch.float16,
        ),
        "ego_embeds": torch.randn(
            1,
            input_embedding_dim,
            device="cuda",
            dtype=torch.float16,
        ),
        "timesteps": torch.zeros(1, device="cuda", dtype=torch.int32),
    }


def _run_trt_engine(plan_path: Path, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(plan_path.read_bytes())
    assert engine is not None, f"Failed to deserialize TensorRT engine: {plan_path}"
    context = engine.create_execution_context()
    assert context is not None, "Failed to create TensorRT execution context"

    for name, tensor in inputs.items():
        context.set_input_shape(name, tuple(tensor.shape))

    output_shape = tuple(context.get_tensor_shape("model_prediction"))
    output_dtype = _torch_dtype_from_trt(engine.get_tensor_dtype("model_prediction"))
    output = torch.empty(output_shape, device="cuda", dtype=output_dtype)

    for name, tensor in inputs.items():
        context.set_tensor_address(name, tensor.contiguous().data_ptr())
    context.set_tensor_address("model_prediction", output.data_ptr())

    ok = context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    assert ok, "TensorRT denoising-step execution failed"
    torch.cuda.synchronize()
    return output


def test_diffusion_denoising_step_trt_matches_pytorch() -> None:
    plan_path, metadata = _require_denoising_step_engine()
    inputs = _make_inputs(metadata)

    planner, _ = rv._load_diffusion_planner_from_checkpoint(
        metadata["checkpoint_path"],
        sampling_method=metadata.get("sampling_method", "ddim"),
    )
    rv._disable_compiled_modulate_methods_for_export(planner)
    wrapper = rv.DiffusionDenoisingStepWrapper(planner).eval().cuda()

    with torch.inference_mode():
        reference = wrapper(
            inputs["current_actions"],
            inputs["vl_embeds"],
            inputs["history_embeds"],
            inputs["ego_embeds"],
            inputs["timesteps"],
        )
        candidate = _run_trt_engine(plan_path, inputs)

    assert candidate.shape == reference.shape
    diff = (reference.float() - candidate.float()).abs()
    assert torch.allclose(reference.float(), candidate.float(), atol=ATOL, rtol=RTOL), (
        f"TRT denoising-step output mismatch: max_abs_diff={diff.max().item():.6f}, "
        f"mean_abs_diff={diff.mean().item():.6f}, atol={ATOL}, rtol={RTOL}"
    )
