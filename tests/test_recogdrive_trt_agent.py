import gc
import json
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXAMPLES_MODELS_ROOT = Path(__file__).resolve().parents[1] / "examples" / "models"
if str(EXAMPLES_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_MODELS_ROOT))

from recogdrive import recogdrive_vlm_engines as rv

if "decord" not in sys.modules:
    decord = types.ModuleType("decord")

    class VideoReader:
        pass

    def cpu(*args, **kwargs):
        return None

    decord.VideoReader = VideoReader
    decord.cpu = cpu
    sys.modules["decord"] = decord


def _install_navsim_nuplan_stubs() -> None:
    if "navsim" in sys.modules and "nuplan" in sys.modules:
        return

    def register_module(name: str) -> types.ModuleType:
        module = sys.modules.get(name)
        if module is None:
            module = types.ModuleType(name)
            sys.modules[name] = module
        return module

    navsim = register_module("navsim")
    navsim_agents = register_module("navsim.agents")
    navsim_agents_abstract = register_module("navsim.agents.abstract_agent")
    navsim_common = register_module("navsim.common")
    navsim_common_dataclasses = register_module("navsim.common.dataclasses")
    navsim_common_dataloader = register_module("navsim.common.dataloader")
    navsim_evaluate = register_module("navsim.evaluate")
    navsim_evaluate_pdm = register_module("navsim.evaluate.pdm_score")
    navsim_planning = register_module("navsim.planning")
    navsim_planning_training = register_module("navsim.planning.training")
    navsim_planning_simulation = register_module("navsim.planning.simulation")
    navsim_planning_simulation_planner = register_module("navsim.planning.simulation.planner")
    navsim_planning_training_abstract = register_module("navsim.planning.training.abstract_feature_target_builder")
    navsim_pdm = register_module("navsim.planning.simulation.planner.pdm_planner")
    navsim_pdm_scoring_pkg = register_module("navsim.planning.simulation.planner.pdm_planner.scoring")
    navsim_pdm_simulation_pkg = register_module("navsim.planning.simulation.planner.pdm_planner.simulation")
    navsim_pdm_scoring = register_module("navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer")
    navsim_pdm_simulator = register_module("navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator")

    nuplan = register_module("nuplan")
    nuplan_planning = register_module("nuplan.planning")
    nuplan_planning_simulation = register_module("nuplan.planning.simulation")
    nuplan_planning_trajectory = register_module("nuplan.planning.simulation.trajectory")
    nuplan_trajectory_sampling = register_module("nuplan.planning.simulation.trajectory.trajectory_sampling")

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


_install_navsim_nuplan_stubs()

from src.agents.recogdrive.recogdrive_agent import ReCogDriveAgent
from src.agents.recogdrive.recogdrive_backbone import RecogDriveBackbone
from src.agents.recogdrive.recogdrive_trt_agent import ReCogDriveTRTAgent
from src.agents.recogdrive.recogdrive_trt_backbone import (
    RecogDriveTRTBackbone,
    hidden_state_engine_dir_for,
)
from src.agents.recogdrive.utils.utils import format_number
from src.agents.safe_copilot.recogdrive.recogdrive_agent import (
    ReCogDriveAgent as SafeCopilotReCogDriveAgent,
)
from src.agents.safe_copilot.recogdrive.recogdrive_backbone import (
    IMG_CONTEXT_TOKEN as SAFE_IMG_CONTEXT_TOKEN,
)
from src.agents.safe_copilot.recogdrive.recogdrive_backbone import (
    IMG_END_TOKEN as SAFE_IMG_END_TOKEN,
)
from src.agents.safe_copilot.recogdrive.recogdrive_backbone import (
    IMG_START_TOKEN as SAFE_IMG_START_TOKEN,
)
from src.agents.safe_copilot.recogdrive.recogdrive_backbone import (
    system_message as SAFE_SYSTEM_MESSAGE,
)
from src.agents.safe_copilot.recogdrive.utils.conversation import get_conv_template as get_safe_conv_template
from src.agents.safe_copilot.safe_agent import ReCogDriveAgentTRT as SafeCopilotReCogDriveAgentTRT
from src.agents.safe_copilot.utils.internvl_preprocess import load_image as safe_load_image

HIDDEN_STATE_MAX_INPUT_LEN = 2800
MAX_PROMPT_EMBEDDING_TABLE_SIZE = 3328
HIDDEN_STATE_REMOVE_INPUT_PADDING = False
HIDDEN_STATE_GPT_ATTENTION_PLUGIN = None
HIDDEN_STATE_COSINE_MEAN_MIN = 0.99
HIDDEN_STATE_COSINE_MIN_MIN = 0.97
DIFFUSION_TRAJECTORY_ATOL = 5e-2
DIFFUSION_TRAJECTORY_RTOL = 5e-2
TRAJECTORY_ATOL = 5e-2
TRAJECTORY_RTOL = 5e-2

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for ReCogDrive TRT agent tests"),
    pytest.mark.skipif(not rv.SNAPSHOT.exists(), reason=f"Local ReCogDrive snapshot not found: {rv.SNAPSHOT}"),
]


def _release_cuda_memory(*objects) -> None:
    for obj in objects:
        del obj
    gc.collect()
    torch.cuda.empty_cache()


def _create_test_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "recogdrive_agent_input.png"
    image = Image.new("RGB", (448, 448), color=(96, 140, 192))
    image.save(image_path)
    return image_path


def _make_feature_batch(image_path: Path) -> dict[str, torch.Tensor]:
    history_trajectory = torch.tensor(
        [
            [
                [0.00, 0.00, 0.00],
                [0.25, 0.05, 0.01],
                [0.55, 0.12, 0.03],
                [0.90, 0.20, 0.05],
            ]
        ],
        dtype=torch.float32,
    )
    high_command_one_hot = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)
    status_feature = torch.tensor(
        [[0.0, 1.0, 0.0, 4.5, 0.2, 0.1, -0.05, 0.0]],
        dtype=torch.float32,
    )
    image_path_tensor = torch.tensor([[ord(char) for char in str(image_path)]], dtype=torch.long)
    return {
        "history_trajectory": history_trajectory,
        "high_command_one_hot": high_command_one_hot,
        "status_feature": status_feature,
        "image_path_tensor": image_path_tensor,
    }


def _clone_features(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.clone() for name, value in features.items()}


def _make_init_actions() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [-0.15, -0.05, -0.02],
                [-0.05, 0.00, -0.01],
                [0.00, 0.04, 0.00],
                [0.08, 0.09, 0.02],
                [0.15, 0.12, 0.03],
                [0.20, 0.14, 0.04],
                [0.24, 0.16, 0.05],
                [0.28, 0.18, 0.06],
            ]
        ],
        dtype=torch.float32,
    )


def _make_cached_hidden_state() -> torch.Tensor:
    return torch.linspace(
        -0.25,
        0.25,
        steps=256 * 1536,
        dtype=torch.float32,
    ).reshape(1, 256, 1536)


def _build_questions(features: dict[str, torch.Tensor]) -> list[str]:
    navigation_commands = ["turn left", "go straight", "turn right"]
    command_indices = torch.argmax(features["high_command_one_hot"], dim=-1)
    questions = []

    for batch_index, command_index in enumerate(command_indices):
        history_trajectory = features["history_trajectory"][batch_index]
        history_str = " ".join(
            [
                f"   - t-{3 - j}: ({format_number(history_trajectory[j, 0].item())}, "
                f"{format_number(history_trajectory[j, 1].item())}, "
                f"{format_number(history_trajectory[j, 2].item())})"
                for j in range(history_trajectory.shape[0])
            ]
        )
        prompt = (
            "<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n"
            "1. Visual perception from front camera view\n"
            f"2. Historical motion context (last 4 timesteps):{history_str}\n"
            f"3. Active navigation command: [{navigation_commands[command_index.item()].upper()}]"
        )
        output_requirements = (
            "\nOutput requirements:\n- Predict 8 future trajectory points\n"
            "- Each point format: (x:float, y:float, heading:float)\n"
            "- Use [PT, ...] to encapsulate the trajectory\n"
            "- Maintain numerical precision to 2 decimal places"
        )
        questions.append(f"{prompt}{output_requirements}")
    return questions


def _build_safe_reference_last_hidden_state(
    image_path: Path,
    features: dict[str, torch.Tensor],
    *,
    max_input_len: int = 2800,
    padding: bool | str = "max_length",
) -> torch.Tensor:
    model = AutoModel.from_pretrained(
        rv.resolve_model_source(),
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        use_flash_attn=True,
        device_map="cuda",
    ).eval()
    tokenizer = AutoTokenizer.from_pretrained(
        rv.resolve_model_source(),
        trust_remote_code=True,
        use_fast=False,
    )
    tokenizer.padding_side = "left"
    model.system_message = SAFE_SYSTEM_MESSAGE
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(SAFE_IMG_CONTEXT_TOKEN)

    pixel_values = safe_load_image(str(image_path)).cuda()
    num_patches = pixel_values.shape[0]
    question = _build_questions(features)[0]

    template = get_safe_conv_template("internvl2_5")
    template.system_message = SAFE_SYSTEM_MESSAGE
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = SAFE_IMG_START_TOKEN + SAFE_IMG_CONTEXT_TOKEN * 256 * num_patches + SAFE_IMG_END_TOKEN
    query = query.replace("<image>", image_tokens, 1)

    model_inputs = tokenizer(
        [query],
        return_tensors="pt",
        padding=padding,
        truncation=True,
        max_length=max_input_len,
    )
    input_ids = model_inputs["input_ids"].to("cuda")
    attention_mask = model_inputs["attention_mask"].to("cuda")
    position_ids = attention_mask.long().cumsum(-1) - 1
    position_ids.masked_fill_(attention_mask == 0, 1)
    image_flags = torch.tensor([1] * num_patches, dtype=torch.long, device="cuda")

    try:
        with torch.inference_mode():
            outputs = model(
                pixel_values=pixel_values.to(torch.float16),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags.squeeze(-1),
                output_hidden_states=True,
                return_dict=True,
            )
        return outputs.hidden_states[-1].detach()
    finally:
        del model
        torch.cuda.empty_cache()


def _require_prebuilt_engines() -> tuple[Path, Path, Path]:
    vision_engine_path = rv.VISION_PLAN
    hidden_state_engine_dir = hidden_state_engine_dir_for(
        str(rv.LLM_DIR),
        HIDDEN_STATE_MAX_INPUT_LEN,
        MAX_PROMPT_EMBEDDING_TABLE_SIZE,
        remove_input_padding=HIDDEN_STATE_REMOVE_INPUT_PADDING,
        gpt_attention_plugin=HIDDEN_STATE_GPT_ATTENTION_PLUGIN,
    )
    diffusion_engine_path = rv.DIFFUSION_PLAN
    if not vision_engine_path.exists():
        pytest.skip(
            "Vision TRT engine is missing. Build it with `examples/models/recogdrive/convert_checkpoint.py` first."
        )
    if not (hidden_state_engine_dir / "rank0.engine").exists():
        pytest.skip(
            "Padding-preserving hidden-state TRT-LLM engine is missing. Build it with "
            "`examples/models/recogdrive/convert_checkpoint.py` using the default hidden-state settings first."
        )
    if not diffusion_engine_path.exists():
        pytest.skip(
            "Diffusion TRT engine is missing. Build it with `examples/models/recogdrive/convert_checkpoint.py` first."
        )
    return vision_engine_path, hidden_state_engine_dir, diffusion_engine_path


def _require_diffusion_engine() -> Path:
    diffusion_engine_path = rv.DIFFUSION_PLAN
    if not diffusion_engine_path.exists():
        pytest.skip(
            "Diffusion TRT engine is missing. Build it with `examples/models/recogdrive/convert_checkpoint.py` first."
        )
    return diffusion_engine_path


def test_recogdrive_trt_backbone_matches_recogdrive_backbone(tmp_path: Path) -> None:
    vision_engine_path, hidden_state_engine_dir, _ = _require_prebuilt_engines()
    image_path = _create_test_image(tmp_path)
    features = _make_feature_batch(image_path)

    reference_backbone = RecogDriveBackbone(
        model_type="internvl",
        checkpoint_path=rv.resolve_model_source(),
        device="cuda:0",
    )
    trt_backbone = RecogDriveTRTBackbone(
        device="cuda:0",
        vision_engine_path=vision_engine_path,
        hidden_state_engine_dir=hidden_state_engine_dir,
        hidden_state_max_input_len=HIDDEN_STATE_MAX_INPUT_LEN,
        hidden_state_max_prompt_embedding_table_size=MAX_PROMPT_EMBEDDING_TABLE_SIZE,
        hidden_state_remove_input_padding=HIDDEN_STATE_REMOVE_INPUT_PADDING,
        hidden_state_gpt_attention_plugin=HIDDEN_STATE_GPT_ATTENTION_PLUGIN,
    )

    image_paths = ReCogDriveAgent._decode_paths_from_tensor(features["image_path_tensor"])

    from src.agents.recogdrive.utils.internvl_preprocess import load_image

    pixel_values = load_image(str(image_paths[0])).cuda()
    num_patches_list = [pixel_values.shape[0]]
    questions = _build_questions(features)

    with torch.inference_mode():
        reference_hidden = reference_backbone(pixel_values, questions, num_patches_list).hidden_states[-1]
        trt_hidden = trt_backbone(pixel_values, questions, num_patches_list).hidden_states[-1]

    metrics = rv._metric_summary(reference_hidden, trt_hidden)
    assert metrics["reference_shape"] == metrics["candidate_shape"]
    assert metrics["cosine_mean"] >= HIDDEN_STATE_COSINE_MEAN_MIN
    assert metrics["cosine_min"] >= HIDDEN_STATE_COSINE_MIN_MIN

    _release_cuda_memory(reference_backbone, trt_backbone)


def test_recogdrive_diffusion_trt_matches_recogdrive_action_head(tmp_path: Path) -> None:
    diffusion_engine_path = _require_diffusion_engine()
    image_path = _create_test_image(tmp_path)
    features = _make_feature_batch(image_path)
    features["last_hidden_state"] = _make_cached_hidden_state()
    init_actions = _make_init_actions()
    trajectory_sampling = sys.modules["nuplan.planning.simulation.trajectory.trajectory_sampling"].TrajectorySampling(
        num_poses=8
    )

    reference_agent = ReCogDriveAgent(
        trajectory_sampling=trajectory_sampling,
        checkpoint_path=str(rv.DIFFUSION_CHECKPOINT),
        cache_mode=False,
        cache_hidden_state=True,
        train_backbone=False,
        dit_type="small",
    ).eval()
    trt_agent = ReCogDriveTRTAgent(
        trajectory_sampling=trajectory_sampling,
        checkpoint_path=str(rv.DIFFUSION_CHECKPOINT),
        cache_mode=False,
        cache_hidden_state=True,
        train_backbone=False,
        dit_type="small",
        diffusion_engine_path=diffusion_engine_path,
    ).eval()
    reference_agent.initialize()
    trt_agent.initialize()

    with torch.inference_mode():
        reference_predictions = reference_agent.forward(
            _clone_features(features),
            deterministic=True,
            init_actions=init_actions.clone(),
        )
        trt_predictions = trt_agent.forward(
            _clone_features(features),
            init_actions=init_actions.clone(),
        )

    assert torch.allclose(
        reference_predictions["pred_traj"],
        trt_predictions["pred_traj"],
        atol=DIFFUSION_TRAJECTORY_ATOL,
        rtol=DIFFUSION_TRAJECTORY_RTOL,
    )

    _release_cuda_memory(reference_agent, trt_agent)


def test_safe_copilot_safe_agent_uses_created_diffusion_engine(tmp_path: Path) -> None:
    diffusion_engine_path = REPO_ROOT / "models" / "diffusion_denoising_step_fp16.plan"
    diffusion_metadata_path = diffusion_engine_path.with_suffix(".metadata.json")
    if not diffusion_engine_path.exists():
        pytest.skip(
            "SAFE diffusion denoising-step TRT engine is missing. Build it with "
            "`examples/models/diffusion_planner/diffusion_onnx_trt.py` first."
        )
    if not diffusion_metadata_path.exists():
        pytest.skip(f"SAFE diffusion denoising-step metadata is missing: {diffusion_metadata_path}")

    image_path = _create_test_image(tmp_path)
    features = _make_feature_batch(image_path)
    features["last_hidden_state"] = _make_cached_hidden_state()
    init_actions = _make_init_actions()
    trajectory_sampling = sys.modules["nuplan.planning.simulation.trajectory.trajectory_sampling"].TrajectorySampling(
        num_poses=8
    )

    reference_agent = (
        SafeCopilotReCogDriveAgent(
            trajectory_sampling=trajectory_sampling,
            checkpoint_path=str(rv.DIFFUSION_CHECKPOINT),
            cache_mode=False,
            cache_hidden_state=True,
            dit_type="small",
        )
        .eval()
        .cuda()
    )
    trt_agent = (
        SafeCopilotReCogDriveAgentTRT(
            trajectory_sampling=trajectory_sampling,
            checkpoint_path=str(rv.DIFFUSION_CHECKPOINT),
            cache_mode=False,
            cache_hidden_state=True,
            dit_type="small",
            diffusion_engine_path=diffusion_engine_path,
            diffusion_metadata_path=diffusion_metadata_path,
        )
        .eval()
        .cuda()
    )
    reference_agent.initialize()
    trt_agent.initialize()

    with torch.inference_mode():
        reference_predictions = reference_agent.forward(
            _clone_features(features),
            deterministic=True,
            init_actions=init_actions.clone(),
        )
        trt_predictions = trt_agent.forward(
            _clone_features(features),
            deterministic=True,
            init_actions=init_actions.clone(),
        )

    assert torch.allclose(
        reference_predictions["pred_traj"],
        trt_predictions["pred_traj"],
        atol=DIFFUSION_TRAJECTORY_ATOL,
        rtol=DIFFUSION_TRAJECTORY_RTOL,
    )

    _release_cuda_memory(reference_agent, trt_agent)


def test_recogdrive_trt_agent_matches_recogdrive_agent(tmp_path: Path) -> None:
    vision_engine_path, hidden_state_engine_dir, diffusion_engine_path = _require_prebuilt_engines()
    image_path = _create_test_image(tmp_path)
    features = _make_feature_batch(image_path)
    init_actions = _make_init_actions()
    trajectory_sampling = sys.modules["nuplan.planning.simulation.trajectory.trajectory_sampling"].TrajectorySampling(
        num_poses=8
    )

    reference_agent = ReCogDriveAgent(
        trajectory_sampling=trajectory_sampling,
        vlm_path=rv.resolve_model_source(),
        checkpoint_path=str(rv.DIFFUSION_CHECKPOINT),
        cache_mode=False,
        cache_hidden_state=False,
        train_backbone=False,
        dit_type="small",
    ).eval()
    trt_agent = ReCogDriveTRTAgent(
        trajectory_sampling=trajectory_sampling,
        cache_mode=False,
        cache_hidden_state=False,
        train_backbone=False,
        dit_type="small",
        vision_engine_path=vision_engine_path,
        hidden_state_engine_dir=hidden_state_engine_dir,
        diffusion_engine_path=diffusion_engine_path,
        hidden_state_max_input_len=HIDDEN_STATE_MAX_INPUT_LEN,
        hidden_state_max_prompt_embedding_table_size=MAX_PROMPT_EMBEDDING_TABLE_SIZE,
        hidden_state_remove_input_padding=HIDDEN_STATE_REMOVE_INPUT_PADDING,
        hidden_state_gpt_attention_plugin=HIDDEN_STATE_GPT_ATTENTION_PLUGIN,
    ).eval()
    reference_agent.initialize()
    trt_agent.initialize()

    with torch.inference_mode():
        reference_predictions = reference_agent.forward(
            _clone_features(features),
            deterministic=True,
            init_actions=init_actions.clone(),
        )
        trt_predictions = trt_agent.forward(
            _clone_features(features),
            deterministic=True,
            init_actions=init_actions.clone(),
        )

    assert torch.allclose(
        reference_predictions["pred_traj"],
        trt_predictions["pred_traj"],
        atol=TRAJECTORY_ATOL,
        rtol=TRAJECTORY_RTOL,
    )

    _release_cuda_memory(reference_agent, trt_agent)


def test_safe_copilot_safe_agent_matches_safe_copilot_recogdrive_agent(tmp_path: Path) -> None:
    vision_engine_path = REPO_ROOT / "models" / "vision_projector_fp16.plan"
    qwen_engine_dir = REPO_ROOT / "models" / "qwen_multimodal_256_rip"
    qwen_config_path = qwen_engine_dir / "config.json"
    qwen_engine_path = qwen_engine_dir / "rank0.engine"
    if not vision_engine_path.exists():
        pytest.skip(
            "SAFE vision TRT engine is missing. Build or place "
            "`/workspaces/safe-copilot/models/vision_projector_fp16.plan` first."
        )
    if not qwen_config_path.exists() or not qwen_engine_path.exists():
        pytest.skip(
            "SAFE Qwen hidden-state TRT engine is missing. Build or place "
            "`/workspaces/safe-copilot/models/qwen/{config.json,rank0.engine}` first."
        )
    qwen_config = json.loads(qwen_config_path.read_text())
    max_prompt_embedding_table_size = int(qwen_config["build_config"].get("max_prompt_embedding_table_size", 0))
    remove_input_padding = bool(qwen_config["build_config"]["plugin_config"].get("remove_input_padding", False))
    if max_prompt_embedding_table_size <= 0:
        pytest.skip(
            "The SAFE Qwen hidden-state engine was built without prompt-embedding capacity. Rebuild "
            "`/workspaces/safe-copilot/models/qwen` with max_prompt_embedding_table_size > 0."
        )

    image_path = _create_test_image(tmp_path)
    features = _make_feature_batch(image_path)
    init_actions = _make_init_actions()
    trajectory_sampling = sys.modules["nuplan.planning.simulation.trajectory.trajectory_sampling"].TrajectorySampling(
        num_poses=8
    )

    reference_last_hidden_state = _build_safe_reference_last_hidden_state(
        image_path,
        features,
        padding=False if remove_input_padding else "max_length",
    )

    reference_agent = (
        SafeCopilotReCogDriveAgent(
            trajectory_sampling=trajectory_sampling,
            checkpoint_path=str(rv.DIFFUSION_CHECKPOINT),
            cache_mode=False,
            cache_hidden_state=True,
            dit_type="small",
        )
        .eval()
        .cuda()
    )
    trt_agent = SafeCopilotReCogDriveAgentTRT(
        checkpoint_path=str(rv.DIFFUSION_CHECKPOINT),
        cache_mode=False,
        cache_hidden_state=False,
        dit_type="small",
        vision_engine_path=vision_engine_path,
        hidden_state_engine_dir=qwen_engine_dir,
        llm_dir=qwen_engine_dir,
        hidden_state_max_input_len=2800,
        hidden_state_max_prompt_embedding_table_size=max_prompt_embedding_table_size,
        hidden_state_remove_input_padding=remove_input_padding,
        hidden_state_gpt_attention_plugin=None,
    ).eval()
    reference_agent.initialize()
    trt_agent.initialize()

    with torch.inference_mode():
        reference_predictions = reference_agent.forward(
            _clone_features({**features, "last_hidden_state": reference_last_hidden_state.clone().float().cpu()}),
            deterministic=True,
            init_actions=init_actions.clone(),
        )
        trt_pred_traj = trt_agent.predict_pred_traj(
            _clone_features(features),
            deterministic=True,
            init_actions=init_actions.clone(),
        )

    assert reference_predictions["pred_traj"].shape == trt_pred_traj.shape
    assert torch.allclose(
        reference_predictions["pred_traj"],
        trt_pred_traj,
        atol=TRAJECTORY_ATOL,
        rtol=TRAJECTORY_RTOL,
    )

    _release_cuda_memory(reference_agent, trt_agent)
