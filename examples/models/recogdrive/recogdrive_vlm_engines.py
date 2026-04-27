import json
import os
import sys
import types
import warnings
from dataclasses import dataclass
from pathlib import Path

import tensorrt as trt
import torch
import torch.nn.functional as F
from recogdrive.trt_hidden_state_engine import (
    build_hidden_state_engine,
    get_last_hidden_state_tensorrt_llm,
    get_last_hidden_state_tensorrt_llm_from_input_ids,
)
from transformers import AutoConfig, AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MODEL_ID = "owl10/ReCogDrive-VLM-2B"
PROMPT = os.getenv("PROMPT", "What is the capital of France?")
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
VISION_DIR = ROOT / "vision_trt"
LLM_DIR = ROOT / "qwen2_submodel"
VISION_ONNX = VISION_DIR / "vision_projector.onnx"
VISION_PLAN = VISION_DIR / "vision_projector.plan"
DIFFUSION_DIR = ROOT / "diffusion_trt"
DIFFUSION_ONNX = DIFFUSION_DIR / "diffusion_planner.onnx"
DIFFUSION_PLAN = DIFFUSION_DIR / "diffusion_planner.plan"
DIFFUSION_METADATA = DIFFUSION_DIR / "metadata.json"
DIFFUSION_ENGINE_INTERFACE_VERSION = 2
DIFFUSION_CHECKPOINT = Path(
    os.getenv(
        "RECOGDRIVE_DIFFUSION_CHECKPOINT",
        "/workspaces/safe-copilot/examples/models/diffusion_planner/models/ReCogDrive_Diffusion_Planner_2B_RL.ckpt",
    )
)


class VisionProjectorWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values):
        return self.model.extract_feature(pixel_values)


class DiffusionPlannerWrapper(torch.nn.Module):
    def __init__(self, planner: torch.nn.Module):
        super().__init__()
        self.planner = planner

    def forward(
        self,
        vl_features: torch.Tensor,
        his_traj: torch.Tensor,
        status_feature: torch.Tensor,
        init_actions: torch.Tensor,
    ) -> torch.Tensor:
        planner = self.planner
        vl_embeds = planner.feature_encoder(vl_features)
        history_embeds = planner.his_traj_encoder(his_traj.unsqueeze(1)).repeat(
            1, planner.config.action_horizon, 1
        )
        ego_embeds = planner.ego_status_encoder(status_feature)
        current_actions = init_actions
        batch_size = current_actions.shape[0]
        device = current_actions.device

        if planner.config.sampling_method == "flow":
            dt = 1.0 / planner.config.num_inference_steps
            for step in range(planner.config.num_inference_steps):
                bucket_index = int(
                    step / planner.config.num_inference_steps * planner.config.flow_cfg.num_timestep_buckets
                )
                timestep = torch.full((batch_size,), bucket_index, device=device, dtype=torch.long)
                action_features = planner.action_encoder(current_actions, timestep)
                if hasattr(planner, "position_embedding"):
                    position_ids = torch.arange(planner.config.action_horizon, device=device)
                    action_features = action_features + planner.position_embedding(position_ids)
                vl_embeds_mean = vl_embeds.mean(1).unsqueeze(1).repeat(1, planner.config.action_horizon, 1)
                fused_input = planner.fusion_projector(
                    torch.cat((history_embeds, vl_embeds_mean, action_features), dim=2)
                )
                model_output = planner.model(fused_input, vl_embeds, ego_embeds, timestep)
                predicted_flow = planner.action_decoder(model_output)
                if planner.config.flow_cfg.mean_variance_net:
                    predicted_flow = predicted_flow.chunk(2, dim=-1)[0]
                current_actions = current_actions + dt * predicted_flow
        elif planner.config.sampling_method == "ddim":
            for step_index in range(planner.ddim_steps):
                timestep_value = int(planner.ddim_t[step_index].item())
                timestep = torch.full((batch_size,), timestep_value, device=device, dtype=torch.long)
                schedule_index = torch.full((batch_size,), step_index, device=device, dtype=torch.long)
                mean, _, _ = planner.p_mean_variance(
                    current_actions,
                    timestep,
                    schedule_index,
                    vl_embeds,
                    history_embeds,
                    ego_embeds,
                    deterministic=True,
                )
                current_actions = mean
        else:
            raise NotImplementedError(
                "The TensorRT diffusion export currently supports only 'ddim' and 'flow' sampling."
            )

        final_action_clip_value = getattr(planner, "final_action_clip_value", 1.0)
        if final_action_clip_value is not None:
            current_actions = current_actions.clamp(-final_action_clip_value, final_action_clip_value)
        return planner.denorm_odo(current_actions)


@dataclass
class _DiffusionPlannerSpec:
    checkpoint_path: Path
    feature_dim: int
    input_embedding_dim: int
    hidden_size: int
    his_traj_dim: int
    status_feature_dim: int
    action_dim: int
    action_horizon: int
    num_heads: int
    head_dim: int
    num_layers: int
    output_dim: int
    vlm_size: str
    sampling_method: str


def _register_stub_module(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        sys.modules[name] = module
    return module


def _install_diffusion_builder_stubs() -> None:
    if "decord" not in sys.modules:
        decord = types.ModuleType("decord")

        class VideoReader:
            pass

        def cpu(*args, **kwargs):
            return None

        decord.VideoReader = VideoReader
        decord.cpu = cpu
        sys.modules["decord"] = decord

    if "navsim" in sys.modules and "nuplan" in sys.modules:
        return

    navsim = _register_stub_module("navsim")
    navsim_agents = _register_stub_module("navsim.agents")
    navsim_agents_abstract = _register_stub_module("navsim.agents.abstract_agent")
    navsim_common = _register_stub_module("navsim.common")
    navsim_common_dataclasses = _register_stub_module("navsim.common.dataclasses")
    navsim_common_dataloader = _register_stub_module("navsim.common.dataloader")
    navsim_evaluate = _register_stub_module("navsim.evaluate")
    navsim_evaluate_pdm = _register_stub_module("navsim.evaluate.pdm_score")
    navsim_planning = _register_stub_module("navsim.planning")
    navsim_planning_training = _register_stub_module("navsim.planning.training")
    navsim_planning_simulation = _register_stub_module("navsim.planning.simulation")
    navsim_planning_simulation_planner = _register_stub_module("navsim.planning.simulation.planner")
    navsim_planning_training_abstract = _register_stub_module(
        "navsim.planning.training.abstract_feature_target_builder"
    )
    navsim_pdm = _register_stub_module("navsim.planning.simulation.planner.pdm_planner")
    navsim_pdm_scoring_pkg = _register_stub_module("navsim.planning.simulation.planner.pdm_planner.scoring")
    navsim_pdm_simulation_pkg = _register_stub_module(
        "navsim.planning.simulation.planner.pdm_planner.simulation"
    )
    navsim_pdm_scoring = _register_stub_module(
        "navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer"
    )
    navsim_pdm_simulator = _register_stub_module(
        "navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator"
    )

    nuplan = _register_stub_module("nuplan")
    nuplan_planning = _register_stub_module("nuplan.planning")
    nuplan_planning_simulation = _register_stub_module("nuplan.planning.simulation")
    nuplan_planning_trajectory = _register_stub_module("nuplan.planning.simulation.trajectory")
    nuplan_trajectory_sampling = _register_stub_module(
        "nuplan.planning.simulation.trajectory.trajectory_sampling"
    )

    class AbstractAgent(torch.nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()

    class AbstractFeatureBuilder:
        pass

    class AbstractTargetBuilder:
        pass

    class SensorConfig:
        @staticmethod
        def build_all_sensors(include=None):
            return {"include": include}

    class Trajectory:
        def __init__(self, poses):
            self.poses = poses

    class AgentInput:
        pass

    class Scene:
        pass

    class MetricCacheLoader:
        pass

    @dataclass
    class PDMScorerConfig:
        progress_weight: float = 10.0
        ttc_weight: float = 5.0
        comfortable_weight: float = 2.0

    class PDMScorer:
        def __init__(self, *args, **kwargs):
            pass

    class PDMSimulator:
        def __init__(self, *args, **kwargs):
            pass

    @dataclass
    class TrajectorySampling:
        num_poses: int = 8

    def pdm_score(*args, **kwargs):
        return 0.0

    navsim_agents_abstract.AbstractAgent = AbstractAgent
    navsim_agents_abstract.AgentInput = AgentInput
    navsim_common_dataclasses.AgentInput = AgentInput
    navsim_common_dataclasses.Scene = Scene
    navsim_common_dataclasses.SensorConfig = SensorConfig
    navsim_common_dataclasses.Trajectory = Trajectory
    navsim_common_dataloader.MetricCacheLoader = MetricCacheLoader
    navsim_evaluate_pdm.pdm_score = pdm_score
    navsim_planning_training_abstract.AbstractFeatureBuilder = AbstractFeatureBuilder
    navsim_planning_training_abstract.AbstractTargetBuilder = AbstractTargetBuilder
    navsim_pdm_scoring.PDMScorer = PDMScorer
    navsim_pdm_scoring.PDMScorerConfig = PDMScorerConfig
    navsim_pdm_simulator.PDMSimulator = PDMSimulator
    nuplan_trajectory_sampling.TrajectorySampling = TrajectorySampling

    navsim.agents = navsim_agents
    navsim.common = navsim_common
    navsim.evaluate = navsim_evaluate
    navsim.planning = navsim_planning
    navsim_planning.training = navsim_planning_training
    navsim_planning.simulation = navsim_planning_simulation
    navsim_planning_simulation.planner = navsim_planning_simulation_planner
    navsim_planning_simulation_planner.pdm_planner = navsim_pdm
    navsim_pdm.scoring = navsim_pdm_scoring_pkg
    navsim_pdm.simulation = navsim_pdm_simulation_pkg
    navsim_pdm_scoring_pkg.pdm_scorer = navsim_pdm_scoring
    navsim_pdm_simulation_pkg.pdm_simulator = navsim_pdm_simulator
    nuplan.planning = nuplan_planning
    nuplan_planning.simulation = nuplan_planning_simulation
    nuplan_planning_simulation.trajectory = nuplan_planning_trajectory
    nuplan_planning_trajectory.trajectory_sampling = nuplan_trajectory_sampling


def _get_diffusion_planner_types():
    try:
        from src.agents.recogdrive.recogdrive_diffusion_planner import (
            ReCogDriveDiffusionPlanner,
            ReCogDriveDiffusionPlannerConfig,
        )
    except ModuleNotFoundError as exc:
        if exc.name not in {"decord", "navsim", "nuplan"}:
            raise
        _install_diffusion_builder_stubs()
        from src.agents.recogdrive.recogdrive_diffusion_planner import (
            ReCogDriveDiffusionPlanner,
            ReCogDriveDiffusionPlannerConfig,
        )

    return ReCogDriveDiffusionPlanner, ReCogDriveDiffusionPlannerConfig


def resolve_model_source() -> str:
    if SNAPSHOT.exists():
        return str(SNAPSHOT)
    return MODEL_ID


def _local_files_only() -> bool:
    return SNAPSHOT.exists()


def _device_from_device_map(device_map) -> torch.device:
    if isinstance(device_map, str):
        return torch.device(device_map)
    if isinstance(device_map, int):
        return torch.device(f"cuda:{device_map}")
    if isinstance(device_map, dict) and device_map:
        root_device = device_map.get("", next(iter(device_map.values())))
        if isinstance(root_device, int):
            return torch.device(f"cuda:{root_device}")
        return torch.device(str(root_device))
    return torch.device("cuda:0")


def load_reference_model(device_map="cuda:0", *, use_flash_attn: bool = True):
    source = resolve_model_source()
    base_kwargs = dict(
        dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=use_flash_attn,
        local_files_only=_local_files_only(),
    )

    try:
        return AutoModel.from_pretrained(source, device_map=device_map, **base_kwargs).eval()
    except Exception:
        model = AutoModel.from_pretrained(source, **base_kwargs).eval()
        return model.to(_device_from_device_map(device_map))


def _load_tokenizer_from_source(source: str, llm_config=None):
    kwargs = dict(
        trust_remote_code=True,
        local_files_only=_local_files_only(),
    )
    if llm_config is not None:
        kwargs["config"] = llm_config

    try:
        return AutoTokenizer.from_pretrained(
            source,
            fix_mistral_regex=True,
            **kwargs,
        )
    except TypeError:
        kwargs.pop("fix_mistral_regex", None)
        return AutoTokenizer.from_pretrained(source, **kwargs)


def _resolve_tokenizer_source(llm_config=None) -> str:
    tokenizer_source = os.getenv("RECOGDRIVE_TOKENIZER_SOURCE")
    if tokenizer_source:
        return tokenizer_source

    name_or_path = getattr(llm_config, "_name_or_path", None)
    if name_or_path:
        candidate = Path(name_or_path)
        if candidate.exists():
            return str(candidate)

        source = resolve_model_source()
        source_path = Path(source)
        if source_path.exists():
            relative_candidate = (source_path / name_or_path).resolve()
            if relative_candidate.exists():
                return str(relative_candidate)

    return resolve_model_source()


def load_reference_tokenizer():
    config = AutoConfig.from_pretrained(
        resolve_model_source(),
        trust_remote_code=True,
        local_files_only=_local_files_only(),
    )
    tokenizer_source = _resolve_tokenizer_source(config.llm_config)
    return _load_tokenizer_from_source(tokenizer_source, llm_config=config.llm_config)


def _metric_summary(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference_fp32 = reference.detach().float()
    candidate_fp32 = candidate.detach().float()
    diff = (reference_fp32 - candidate_fp32).abs()
    cosine = F.cosine_similarity(
        reference_fp32.reshape(-1, reference_fp32.shape[-1]),
        candidate_fp32.reshape(-1, candidate_fp32.shape[-1]),
        dim=-1,
    )
    return {
        "reference_shape": tuple(reference.shape),
        "candidate_shape": tuple(candidate.shape),
        "max_abs_diff": diff.max().item(),
        "mean_abs_diff": diff.mean().item(),
        "cosine_mean": cosine.mean().item(),
        "cosine_min": cosine.min().item(),
    }


def _load_full_model(dtype: torch.dtype = torch.float16, *, use_flash_attn: bool = True):
    return AutoModel.from_pretrained(
        resolve_model_source(),
        trust_remote_code=True,
        dtype=dtype,
        use_flash_attn=use_flash_attn,
        local_files_only=_local_files_only(),
    )


def _resolve_diffusion_checkpoint(checkpoint_path: str | Path | None = None) -> Path:
    resolved = Path(checkpoint_path or DIFFUSION_CHECKPOINT).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(
            f"ReCogDrive diffusion planner checkpoint was not found at {resolved}. "
            "Set RECOGDRIVE_DIFFUSION_CHECKPOINT to override it."
        )
    return resolved


def _normalize_diffusion_state_dict(raw_state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for key, value in raw_state_dict.items():
        normalized_key = key
        if normalized_key.startswith("agent.action_head."):
            normalized_key = normalized_key[len("agent.action_head.") :]
        elif normalized_key.startswith("action_head."):
            normalized_key = normalized_key[len("action_head.") :]
        normalized[normalized_key] = value
    return normalized


def _infer_diffusion_planner_spec(
    state_dict: dict[str, torch.Tensor],
    checkpoint_path: Path,
    *,
    sampling_method: str = "ddim",
) -> _DiffusionPlannerSpec:
    feature_encoder_weight = state_dict["feature_encoder.weight"]
    input_embedding_dim, feature_dim = feature_encoder_weight.shape
    hidden_size = state_dict["his_traj_encoder.fc1.weight"].shape[0]
    his_traj_dim = state_dict["his_traj_encoder.fc1.weight"].shape[1]
    status_feature_dim = state_dict["ego_status_encoder.fc1.weight"].shape[1]
    action_dim = int(state_dict["action_decoder.fc2.bias"].numel())
    action_horizon = int(state_dict["position_embedding.weight"].shape[0])
    output_dim, inner_dim = state_dict["model.final_layer.linear.weight"].shape
    head_dim = int(state_dict["model.rotary_embedder.inv_freq"].numel() * 2)
    num_heads = inner_dim // head_dim
    layer_indices = {
        int(key.split(".")[2])
        for key in state_dict
        if key.startswith("model.transformer_blocks.")
    }
    num_layers = max(layer_indices) + 1 if layer_indices else 0
    vlm_size = "small" if feature_dim == 1536 else "large"

    return _DiffusionPlannerSpec(
        checkpoint_path=checkpoint_path,
        feature_dim=feature_dim,
        input_embedding_dim=input_embedding_dim,
        hidden_size=hidden_size,
        his_traj_dim=his_traj_dim,
        status_feature_dim=status_feature_dim,
        action_dim=action_dim,
        action_horizon=action_horizon,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        output_dim=output_dim,
        vlm_size=vlm_size,
        sampling_method=sampling_method,
    )


def _load_diffusion_planner_from_checkpoint(
    checkpoint_path: str | Path | None = None,
    *,
    sampling_method: str = "ddim",
):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to build the ReCogDrive diffusion TensorRT engine")

    resolved_checkpoint = _resolve_diffusion_checkpoint(checkpoint_path)
    raw_checkpoint = torch.load(resolved_checkpoint, map_location="cpu", weights_only=False)
    raw_state_dict = raw_checkpoint["state_dict"] if "state_dict" in raw_checkpoint else raw_checkpoint
    state_dict = _normalize_diffusion_state_dict(raw_state_dict)
    spec = _infer_diffusion_planner_spec(
        state_dict,
        resolved_checkpoint,
        sampling_method=sampling_method,
    )

    ReCogDriveDiffusionPlanner, ReCogDriveDiffusionPlannerConfig = _get_diffusion_planner_types()
    config = ReCogDriveDiffusionPlannerConfig(
        diffusion_model_cfg={
            "num_heads": spec.num_heads,
            "head_dim": spec.head_dim,
            "num_layers": spec.num_layers,
            "output_dim": spec.output_dim,
            "dropout": 0.0,
            "attention_bias": True,
            "norm_eps": 1e-5,
            "interleave_attention": True,
        },
        input_embedding_dim=spec.input_embedding_dim,
        hidden_size=spec.hidden_size,
        action_dim=spec.action_dim,
        action_horizon=spec.action_horizon,
        max_seq_len=spec.action_horizon,
        sampling_method=spec.sampling_method,
        num_inference_steps=5,
        model_dtype="float16",
        grpo=False,
        vlm_size=spec.vlm_size,
    )
    planner = ReCogDriveDiffusionPlanner(config)
    model_state_dict = planner.state_dict()
    shape_mismatched_keys = [
        key
        for key, value in state_dict.items()
        if key in model_state_dict and value.shape != model_state_dict[key].shape
    ]
    if shape_mismatched_keys:
        raise RuntimeError(
            "The ReCogDrive diffusion planner checkpoint does not match the inferred planner shapes. "
            f"Mismatched keys: {shape_mismatched_keys}"
        )
    filtered_state_dict = {
        key: value
        for key, value in state_dict.items()
        if key in model_state_dict and value.shape == model_state_dict[key].shape
    }
    planner.load_state_dict(filtered_state_dict, strict=False)
    parameter_keys = {name for name, _ in planner.named_parameters()}
    missing_parameter_keys = sorted(parameter_keys - filtered_state_dict.keys())
    if missing_parameter_keys:
        raise RuntimeError(
            "Failed to load the ReCogDrive diffusion planner checkpoint cleanly. "
            f"Missing parameter keys: {missing_parameter_keys}"
        )

    planner = planner.eval().cuda()
    if next(planner.parameters()).dtype != torch.float16:
        planner = planner.half()
    return planner, spec


def _diffusion_metadata_matches(
    *,
    checkpoint_path: Path,
    max_vl_seq_len: int,
    sampling_method: str,
) -> bool:
    if not DIFFUSION_PLAN.exists() or not DIFFUSION_METADATA.exists():
        return False

    try:
        metadata = json.loads(DIFFUSION_METADATA.read_text())
    except Exception:
        return False

    return (
        metadata.get("engine_interface_version") == DIFFUSION_ENGINE_INTERFACE_VERSION
        and metadata.get("checkpoint_path") == str(checkpoint_path)
        and metadata.get("max_vl_seq_len") == max_vl_seq_len
        and metadata.get("sampling_method") == sampling_method
    )


def _disable_compiled_modulate_methods_for_export(planner: torch.nn.Module) -> None:
    final_layer = getattr(planner.model, "final_layer", None)
    if final_layer is not None:
        raw_modulate = getattr(getattr(final_layer, "modulate", None), "__wrapped__", None)
        if raw_modulate is not None:
            final_layer.modulate = types.MethodType(raw_modulate, final_layer)

    for block in getattr(planner.model, "transformer_blocks", []):
        raw_modulate = getattr(getattr(block, "modulate", None), "__wrapped__", None)
        if raw_modulate is not None:
            block.modulate = types.MethodType(raw_modulate, block)


def extract_language_submodel() -> Path:
    config_path = LLM_DIR / "config.json"
    if not config_path.exists():
        LLM_DIR.mkdir(parents=True, exist_ok=True)
        model = _load_full_model(dtype=torch.bfloat16)
        model.language_model.save_pretrained(LLM_DIR)
        tokenizer_source = _resolve_tokenizer_source(model.config.llm_config)
        try:
            tokenizer = _load_tokenizer_from_source(tokenizer_source, llm_config=model.config.llm_config)
            tokenizer.save_pretrained(LLM_DIR)
        except Exception as exc:
            warnings.warn(
                "Tokenizer assets were not found locally, so only the language submodel weights/config were saved. "
                "Set RECOGDRIVE_TOKENIZER_SOURCE to a local tokenizer directory if you need tokenizer files in "
                f"{LLM_DIR}. Original error: {exc}"
            )

    config = json.loads(config_path.read_text())
    rope = config.get("rope_scaling")
    if isinstance(rope, dict) and rope.get("type") == "dynamic" and "alpha" not in rope:
        rope["alpha"] = rope.get("factor", 1.0)
        config_path.write_text(json.dumps(config, indent=2))
    return LLM_DIR


def export_vision_onnx() -> Path:
    if VISION_ONNX.exists() and VISION_ONNX.with_suffix(".onnx.data").exists():
        return VISION_ONNX

    VISION_DIR.mkdir(parents=True, exist_ok=True)
    model = _load_full_model(dtype=torch.float16, use_flash_attn=False).eval().cuda()
    wrapper = VisionProjectorWrapper(model).eval().cuda()
    example = torch.zeros(1, 3, 448, 448, device="cuda", dtype=torch.float16)

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (example,),
            str(VISION_ONNX),
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            opset_version=18,
            external_data=True,
            do_constant_folding=True,
        )
    return VISION_ONNX


def build_vision_trt_engine() -> Path:
    if VISION_PLAN.exists():
        return VISION_PLAN

    export_vision_onnx()
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    cwd = os.getcwd()
    try:
        os.chdir(VISION_DIR)
        with open(VISION_ONNX, "rb") as f:
            ok = parser.parse(f.read())
        if not ok:
            errors = [str(parser.get_error(i)) for i in range(parser.num_errors)]
            raise RuntimeError("Failed to parse vision ONNX with TensorRT\n" + "\n".join(errors))

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape("pixel_values", (1, 3, 448, 448), (1, 3, 448, 448), (1, 3, 448, 448))
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
    finally:
        os.chdir(cwd)

    if serialized is None:
        raise RuntimeError("TensorRT failed to build the vision engine")
    VISION_PLAN.write_bytes(bytes(serialized))
    return VISION_PLAN


def run_vision_trt_engine(pixel_values: torch.Tensor) -> torch.Tensor:
    engine_path = build_vision_trt_engine()
    runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()
    context.set_input_shape("pixel_values", tuple(pixel_values.shape))
    output_shape = tuple(context.get_tensor_shape("image_embeds"))
    output = torch.empty(output_shape, device="cuda", dtype=torch.float16)
    context.set_tensor_address("pixel_values", pixel_values.contiguous().data_ptr())
    context.set_tensor_address("image_embeds", output.data_ptr())
    ok = context.execute_async_v3(torch.cuda.current_stream().cuda_stream)
    if not ok:
        raise RuntimeError("TensorRT vision engine execution failed")
    torch.cuda.synchronize()
    return output


def export_diffusion_onnx(
    *,
    max_vl_seq_len: int = 2800,
    checkpoint_path: str | Path | None = None,
    sampling_method: str = "ddim",
) -> Path:
    resolved_checkpoint = _resolve_diffusion_checkpoint(checkpoint_path)
    if _diffusion_metadata_matches(
        checkpoint_path=resolved_checkpoint,
        max_vl_seq_len=max_vl_seq_len,
        sampling_method=sampling_method,
    ) and DIFFUSION_ONNX.exists():
        return DIFFUSION_ONNX

    DIFFUSION_DIR.mkdir(parents=True, exist_ok=True)
    planner, spec = _load_diffusion_planner_from_checkpoint(
        resolved_checkpoint,
        sampling_method=sampling_method,
    )
    _disable_compiled_modulate_methods_for_export(planner)
    wrapper = DiffusionPlannerWrapper(planner).eval().cuda()
    example_vl_features = torch.zeros(
        1,
        min(max_vl_seq_len, max(spec.action_horizon, 256)),
        spec.feature_dim,
        device="cuda",
        dtype=torch.float16,
    )
    example_his_traj = torch.zeros(1, spec.his_traj_dim, device="cuda", dtype=torch.float16)
    example_status_feature = torch.zeros(
        1,
        spec.status_feature_dim,
        device="cuda",
        dtype=torch.float16,
    )
    example_init_actions = torch.zeros(
        1,
        spec.action_horizon,
        spec.action_dim,
        device="cuda",
        dtype=torch.float16,
    )

    with torch.inference_mode():
        torch.onnx.export(
            wrapper,
            (example_vl_features, example_his_traj, example_status_feature, example_init_actions),
            str(DIFFUSION_ONNX),
            input_names=["vl_features", "his_traj", "status_feature", "init_actions"],
            output_names=["pred_traj"],
            opset_version=18,
            dynamo=False,
            external_data=True,
            do_constant_folding=False,
            dynamic_axes={
                "vl_features": {1: "vl_seq_len"},
            },
        )

    DIFFUSION_METADATA.write_text(
        json.dumps(
            {
                "engine_interface_version": DIFFUSION_ENGINE_INTERFACE_VERSION,
                "checkpoint_path": str(resolved_checkpoint),
                "max_vl_seq_len": max_vl_seq_len,
                "sampling_method": sampling_method,
                "feature_dim": spec.feature_dim,
                "his_traj_dim": spec.his_traj_dim,
                "status_feature_dim": spec.status_feature_dim,
                "action_horizon": spec.action_horizon,
                "action_dim": spec.action_dim,
            },
            indent=2,
        )
    )
    return DIFFUSION_ONNX


def build_diffusion_trt_engine(
    *,
    max_vl_seq_len: int = 2800,
    checkpoint_path: str | Path | None = None,
    sampling_method: str = "ddim",
) -> Path:
    resolved_checkpoint = _resolve_diffusion_checkpoint(checkpoint_path)
    if _diffusion_metadata_matches(
        checkpoint_path=resolved_checkpoint,
        max_vl_seq_len=max_vl_seq_len,
        sampling_method=sampling_method,
    ) and DIFFUSION_PLAN.exists():
        return DIFFUSION_PLAN

    export_diffusion_onnx(
        max_vl_seq_len=max_vl_seq_len,
        checkpoint_path=resolved_checkpoint,
        sampling_method=sampling_method,
    )
    _, spec = _load_diffusion_planner_from_checkpoint(
        resolved_checkpoint,
        sampling_method=sampling_method,
    )

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    cwd = os.getcwd()
    try:
        os.chdir(DIFFUSION_DIR)
        with open(DIFFUSION_ONNX, "rb") as handle:
            ok = parser.parse(handle.read())
        if not ok:
            errors = [str(parser.get_error(index)) for index in range(parser.num_errors)]
            raise RuntimeError("Failed to parse diffusion ONNX with TensorRT\n" + "\n".join(errors))

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        profile = builder.create_optimization_profile()
        profile.set_shape(
            "vl_features",
            (1, 1, spec.feature_dim),
            (1, min(max_vl_seq_len, max(spec.action_horizon, 256)), spec.feature_dim),
            (1, max_vl_seq_len, spec.feature_dim),
        )
        profile.set_shape(
            "his_traj",
            (1, spec.his_traj_dim),
            (1, spec.his_traj_dim),
            (1, spec.his_traj_dim),
        )
        profile.set_shape(
            "status_feature",
            (1, spec.status_feature_dim),
            (1, spec.status_feature_dim),
            (1, spec.status_feature_dim),
        )
        profile.set_shape(
            "init_actions",
            (1, spec.action_horizon, spec.action_dim),
            (1, spec.action_horizon, spec.action_dim),
            (1, spec.action_horizon, spec.action_dim),
        )
        config.add_optimization_profile(profile)
        serialized = builder.build_serialized_network(network, config)
    finally:
        os.chdir(cwd)

    if serialized is None:
        raise RuntimeError("TensorRT failed to build the diffusion planner engine")
    DIFFUSION_PLAN.write_bytes(bytes(serialized))
    return DIFFUSION_PLAN


def build_language_trtllm_engine(
    max_input_len: int = 64,
    max_prompt_embedding_table_size: int = 0,
    *,
    remove_input_padding: bool = True,
    gpt_attention_plugin: str | None = "auto",
) -> Path:
    llm_dir = extract_language_submodel()
    return build_hidden_state_engine(
        str(llm_dir),
        max_input_len,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        remove_input_padding=remove_input_padding,
        gpt_attention_plugin=gpt_attention_plugin,
    )


def get_language_last_hidden_state(
    prompt: str,
    max_input_len: int = 64,
    max_prompt_embedding_table_size: int = 0,
) -> torch.Tensor:
    llm_dir = extract_language_submodel()
    return get_last_hidden_state_tensorrt_llm(
        str(llm_dir),
        prompt,
        max_input_len,
    )


def get_language_last_hidden_state_from_input_ids(
    batch_input_ids: list[torch.Tensor],
    *,
    max_input_len: int = 64,
    prompt_embedding_table: torch.Tensor | None = None,
    prompt_tasks: str | torch.Tensor | None = None,
    max_prompt_embedding_table_size: int = 0,
) -> torch.Tensor:
    llm_dir = extract_language_submodel()
    return get_last_hidden_state_tensorrt_llm_from_input_ids(
        str(llm_dir),
        batch_input_ids,
        max_input_len,
        prompt_embedding_table=prompt_embedding_table,
        prompt_tasks=prompt_tasks,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
    )


def compare_reference_model_to_converted_engines(
    prompt: str = PROMPT,
    max_input_len: int = 64,
    device_map="cuda:0",
) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to compare ReCogDrive outputs against TensorRT engines")

    cuda_device = _device_from_device_map(device_map)
    reference_model = load_reference_model(device_map=device_map)
    tokenizer = load_reference_tokenizer()
    image = torch.zeros(1, 3, 448, 448, device=cuda_device, dtype=torch.bfloat16)
    encoded = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len,
        add_special_tokens=True,
    )
    encoded = {name: value.to(cuda_device) for name, value in encoded.items()}

    with torch.inference_mode():
        reference_vision = reference_model.extract_feature(image)
        reference_hidden = reference_model.language_model(
            **encoded,
            output_hidden_states=True,
            return_dict=True,
        ).hidden_states[-1]

    converted_vision = run_vision_trt_engine(image.to(dtype=torch.float16))
    converted_hidden = get_language_last_hidden_state(prompt, max_input_len)

    return {
        "reference_model_source": resolve_model_source(),
        "reference_model_loader": {
            "torch_dtype": "torch.bfloat16",
            "low_cpu_mem_usage": True,
            "trust_remote_code": True,
            "use_flash_attn": True,
            "device_map": str(device_map),
        },
        "prompt": prompt,
        "vision": _metric_summary(reference_vision, converted_vision),
        "last_hidden_state": _metric_summary(reference_hidden, converted_hidden),
        "vision_engine": str(VISION_PLAN),
        "llm_engine": str(build_language_trtllm_engine(max_input_len)),
    }


def verify_recogdrive_engines(prompt: str = PROMPT, max_input_len: int = 64) -> dict:
    image = torch.zeros(1, 3, 448, 448, device="cuda", dtype=torch.float16)

    pytorch_model = _load_full_model(dtype=torch.float16, use_flash_attn=False).eval().cuda()
    with torch.inference_mode():
        pt_image_embeds = pytorch_model.extract_feature(image)
    trt_image_embeds = run_vision_trt_engine(image)
    image_max_diff = (pt_image_embeds - trt_image_embeds).abs().max().item()

    llm_engine_dir = build_language_trtllm_engine(max_input_len)
    last_hidden_state = get_language_last_hidden_state(prompt, max_input_len)

    return {
        "vision_trt_engine": str(VISION_PLAN),
        "vision_output_shape": tuple(trt_image_embeds.shape),
        "vision_max_abs_diff_vs_pytorch": image_max_diff,
        "llm_trtllm_engine": str(llm_engine_dir),
        "last_hidden_state_shape": tuple(last_hidden_state.shape),
    }


GENERATION_ENGINE_ROOT = Path(
    os.getenv(
        "TRT_GENERATION_ENGINE_ROOT",
        "/workspaces/safe-copilot/models/trt_llm_generation_engines",
    )
)


def generation_engine_dir_for(
    model_id: str,
    max_input_len: int,
    max_output_len: int,
    max_prompt_embedding_table_size: int,
) -> Path:
    safe_model_id = model_id.replace("/", "__")
    return (
        GENERATION_ENGINE_ROOT
        / safe_model_id
        / (
            f"max_input_len_{max_input_len}"
            f"__max_output_len_{max_output_len}"
            f"__max_prompt_embedding_table_size_{max_prompt_embedding_table_size}"
        )
    )


def build_language_generation_trtllm_engine(
    max_input_len: int = 1024,
    max_output_len: int = 128,
    max_prompt_embedding_table_size: int = 3072,
) -> Path:
    from tensorrt_llm import AutoModelForCausalLM, BuildConfig, build
    from tensorrt_llm.llmapi.kv_cache_type import KVCacheType

    llm_dir = extract_language_submodel()
    engine_dir = generation_engine_dir_for(
        str(llm_dir),
        max_input_len,
        max_output_len,
        max_prompt_embedding_table_size,
    )
    if (engine_dir / "config.json").exists() and (engine_dir / "rank0.engine").exists():
        return engine_dir

    engine_dir.mkdir(parents=True, exist_ok=True)
    model = AutoModelForCausalLM.from_hugging_face(str(llm_dir), dtype="float16")
    build_config = BuildConfig(
        max_batch_size=1,
        opt_batch_size=1,
        max_input_len=max_input_len,
        max_seq_len=max_input_len + max_output_len,
        max_num_tokens=max_input_len,
        max_beam_width=1,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        kv_cache_type=KVCacheType.PAGED,
    )
    engine = build(model, build_config)
    engine.save(str(engine_dir))
    return engine_dir


def load_language_generation_runner(
    max_input_len: int = 1024,
    max_output_len: int = 128,
    max_prompt_embedding_table_size: int = 3072,
):
    from tensorrt_llm.runtime import ModelRunner

    engine_dir = build_language_generation_trtllm_engine(
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_prompt_embedding_table_size=max_prompt_embedding_table_size,
    )
    return ModelRunner.from_dir(str(engine_dir), max_output_len=max_output_len)
