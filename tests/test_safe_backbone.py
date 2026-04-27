import gc
import sys
from pathlib import Path

import pytest
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

EXAMPLES_MODELS_ROOT = REPO_ROOT / "examples" / "models"
if str(EXAMPLES_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_MODELS_ROOT))


from recogdrive import recogdrive_vlm_engines as rv

from src.agents.safe_copilot.recogdrive.recogdrive_backbone import (
    IMG_CONTEXT_TOKEN,
    IMG_END_TOKEN,
    IMG_START_TOKEN,
    system_message,
)
from src.agents.safe_copilot.recogdrive.recogdrive_backbone import (
    RecogDriveBackbone as SafeReferenceBackbone,
)
from src.agents.safe_copilot.recogdrive.utils.conversation import get_conv_template
from src.agents.safe_copilot.recogdrive.utils.utils import format_number
from src.agents.safe_copilot.safe_backbone import SAFeCopilotBackbone
from src.agents.safe_copilot.utils.internvl_preprocess import load_image as safe_load_image

VISION_ENGINE_CANDIDATES = [
    REPO_ROOT / "models" / "vision_projector_bf16.plan",
    REPO_ROOT / "models" / "vision_projector_fp16.plan",
]
QWEN_ENGINE_DIR = REPO_ROOT / "models" / "qwen_bf16"
VALID_COSINE_MEAN_MIN = 0.985
TEXT_COSINE_MEAN_MIN = 0.993
IMAGE_COSINE_MEAN_MIN = 0.975

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required for SAFE backbone parity tests"),
    pytest.mark.skipif(not rv.SNAPSHOT.exists(), reason=f"Local ReCogDrive snapshot not found: {rv.SNAPSHOT}"),
]


def _release_cuda_memory(*objects) -> None:
    for obj in objects:
        if hasattr(obj, "close"):
            try:
                obj.close()
            except Exception:
                pass
        del obj
    gc.collect()
    torch.cuda.empty_cache()


def _create_test_image(tmp_path: Path) -> Path:
    image_path = tmp_path / "safe_backbone_input.png"
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


def _require_safe_backbone_artifacts() -> Path:
    vision_engine_path = next((path for path in VISION_ENGINE_CANDIDATES if path.exists()), None)
    if vision_engine_path is None:
        pytest.skip(
            "SAFE vision TRT engine is missing. Build either "
            "`/workspaces/safe-copilot/models/vision_projector_bf16.plan` or "
            "`/workspaces/safe-copilot/models/vision_projector_fp16.plan` first."
        )
    if not QWEN_ENGINE_DIR.exists():
        pytest.skip(
            "SAFE Qwen TRT engine is missing. Build "
            "`/workspaces/safe-copilot/models/qwen_bf16` first."
        )
    return vision_engine_path


def _build_query(question: str, num_patches: int) -> str:
    template = get_conv_template("internvl2_5")
    template.system_message = system_message
    template.append_message(template.roles[0], question)
    template.append_message(template.roles[1], None)
    query = template.get_prompt()
    image_tokens = IMG_START_TOKEN + IMG_CONTEXT_TOKEN * 256 * num_patches + IMG_END_TOKEN
    return query.replace("<image>", image_tokens, 1)


def test_safe_backbone_matches_safe_recogdrive_backbone(tmp_path: Path) -> None:
    vision_engine_path = _require_safe_backbone_artifacts()
    image_path = _create_test_image(tmp_path)
    features = _make_feature_batch(image_path)
    pixel_values = safe_load_image(str(image_path)).cuda()
    num_patches_list = [pixel_values.shape[0]]
    questions = _build_questions(features)

    reference_backbone = SafeReferenceBackbone(
        model_type="internvl",
        checkpoint_path=rv.resolve_model_source(),
        device="cuda:0",
    )
    safe_backbone = SAFeCopilotBackbone(
        vit_engine_path=vision_engine_path,
        qwen_engine_dir=QWEN_ENGINE_DIR,
        checkpoint_path=rv.resolve_model_source(),
        device="cuda:0",
    )

    try:
        with torch.inference_mode():
            reference_hidden = reference_backbone(pixel_values, questions, num_patches_list).hidden_states[-1].detach()
            safe_hidden = safe_backbone(pixel_values, questions, num_patches_list).hidden_states[-1].detach()

        metrics = rv._metric_summary(reference_hidden, safe_hidden)
        query = _build_query(questions[0], num_patches_list[0])
        reference_backbone.tokenizer.padding_side = "left"
        model_inputs = reference_backbone.tokenizer(
            [query],
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=2800,
        )
        input_ids = model_inputs["input_ids"][0]
        attention_mask = model_inputs["attention_mask"][0].bool()
        img_context_token_id = reference_backbone.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        image_mask = input_ids == img_context_token_id
        text_mask = attention_mask & ~image_mask
        valid_metrics = rv._metric_summary(reference_hidden[:, attention_mask, :], safe_hidden[:, attention_mask, :])
        text_metrics = rv._metric_summary(reference_hidden[:, text_mask, :], safe_hidden[:, text_mask, :])
        image_metrics = rv._metric_summary(reference_hidden[:, image_mask, :], safe_hidden[:, image_mask, :])

        assert metrics["reference_shape"] == metrics["candidate_shape"]
        assert valid_metrics["cosine_mean"] >= VALID_COSINE_MEAN_MIN, valid_metrics
        assert text_metrics["cosine_mean"] >= TEXT_COSINE_MEAN_MIN, text_metrics
        assert image_metrics["cosine_mean"] >= IMAGE_COSINE_MEAN_MIN, image_metrics
    finally:
        _release_cuda_memory(reference_backbone, safe_backbone)
