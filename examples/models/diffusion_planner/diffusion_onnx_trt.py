import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

import tensorrt as trt
import torch
from tensorrt_llm._utils import release_gc

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_MODELS_ROOT = REPO_ROOT / "examples" / "models"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXAMPLES_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_MODELS_ROOT))

from recogdrive.recogdrive_vlm_engines import (  # noqa: E402
    DIFFUSION_ENGINE_INTERFACE_VERSION,
    DiffusionDenoisingStepWrapper,
    _disable_compiled_modulate_methods_for_export,
    _load_diffusion_planner_from_checkpoint,
)

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "ReCogDrive_Diffusion_Planner_2B_RL.ckpt"


def default_metadata_path(plan_file_path: str) -> Path:
    return Path(plan_file_path).with_suffix(".metadata.json")


class ONNX_TRT:
    def __init__(self, max_vl_seq_len: int = 2800) -> None:
        self.max_vl_seq_len = max_vl_seq_len

    def export_onnx(
        self,
        onnx_file_path: str,
        checkpoint_path: str,
        image_url: list[str] | None = None,
        sampling_method: str = "ddim",
        metadata_file_path: str | None = None,
    ) -> None:
        print("Start converting ONNX denoising-step model!")
        if image_url:
            warnings.warn("--image_url is ignored for diffusion planner export.")

        planner, spec = _load_diffusion_planner_from_checkpoint(
            checkpoint_path,
            sampling_method=sampling_method,
        )
        _disable_compiled_modulate_methods_for_export(planner)
        wrapper = DiffusionDenoisingStepWrapper(planner).eval().cuda()
        example_seq_len = min(self.max_vl_seq_len, max(spec.action_horizon, 256))
        example_current_actions = torch.zeros(
            1,
            spec.action_horizon,
            spec.action_dim,
            device="cuda",
            dtype=torch.float16,
        )
        example_vl_embeds = torch.zeros(
            1,
            example_seq_len,
            spec.input_embedding_dim,
            device="cuda",
            dtype=torch.float16,
        )
        example_history_embeds = torch.zeros(
            1,
            spec.action_horizon,
            spec.input_embedding_dim,
            device="cuda",
            dtype=torch.float16,
        )
        example_ego_embeds = torch.zeros(
            1,
            spec.input_embedding_dim,
            device="cuda",
            dtype=torch.float16,
        )
        example_timesteps = torch.zeros(1, device="cuda", dtype=torch.int32)

        with torch.inference_mode():
            torch.onnx.export(
                wrapper,
                (
                    example_current_actions,
                    example_vl_embeds,
                    example_history_embeds,
                    example_ego_embeds,
                    example_timesteps,
                ),
                onnx_file_path,
                input_names=["current_actions", "vl_embeds", "history_embeds", "ego_embeds", "timesteps"],
                output_names=["model_prediction"],
                opset_version=18,
                external_data=True,
                do_constant_folding=False,
                dynamic_axes={
                    "current_actions": {0: "batch"},
                    "vl_embeds": {0: "batch", 1: "vl_seq_len"},
                    "history_embeds": {0: "batch"},
                    "ego_embeds": {0: "batch"},
                    "timesteps": {0: "batch"},
                    "model_prediction": {0: "batch"},
                },
                dynamo=False,
            )

        if metadata_file_path:
            Path(metadata_file_path).write_text(
                json.dumps(
                    {
                        "engine_interface_version": DIFFUSION_ENGINE_INTERFACE_VERSION,
                        "checkpoint_path": str(Path(checkpoint_path).expanduser().resolve()),
                        "max_vl_seq_len": self.max_vl_seq_len,
                        "sampling_method": sampling_method,
                        "feature_dim": spec.feature_dim,
                        "input_embedding_dim": spec.input_embedding_dim,
                        "his_traj_dim": spec.his_traj_dim,
                        "status_feature_dim": spec.status_feature_dim,
                        "action_horizon": spec.action_horizon,
                        "action_dim": spec.action_dim,
                        "model_prediction_dim": spec.model_prediction_dim,
                    },
                    indent=2,
                )
            )
        release_gc()
        print(f"Export to ONNX file successfully! The ONNX file stays in {onnx_file_path}")

    def generate_trt_engine(
        self,
        onnxFile: str,
        planFile: str,
        minBS: int = 1,
        optBS: int = 1,
        maxBS: int = 1,
    ) -> None:
        print("Start converting TRT denoising-step engine!")
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        parser = trt.OnnxParser(network, logger)
        onnx_path = Path(onnxFile)

        cwd = os.getcwd()
        try:
            if onnx_path.parent:
                os.chdir(onnx_path.parent)
            with open(onnx_path.name, "rb") as model:
                if not parser.parse(model.read()):
                    print(f"Failed parsing {onnxFile}")
                    for error in range(parser.num_errors):
                        print(parser.get_error(error))
                    raise RuntimeError(f"Failed parsing {onnxFile}")
                print(f"Succeeded parsing {onnxFile}")
        finally:
            os.chdir(cwd)

        for index in range(network.num_inputs):
            inputT = network.get_input(index)
            input_name = inputT.name
            input_shape = list(inputT.shape)

            if input_name == "current_actions":
                action_horizon = input_shape[1]
                action_dim = input_shape[2]
                profile.set_shape(
                    input_name,
                    [minBS, action_horizon, action_dim],
                    [optBS, action_horizon, action_dim],
                    [maxBS, action_horizon, action_dim],
                )
            elif input_name == "vl_embeds":
                hidden_dim = input_shape[2]
                profile.set_shape(
                    input_name,
                    [minBS, 1, hidden_dim],
                    [optBS, min(self.max_vl_seq_len, 256), hidden_dim],
                    [maxBS, self.max_vl_seq_len, hidden_dim],
                )
            elif input_name == "history_embeds":
                action_horizon = input_shape[1]
                hidden_dim = input_shape[2]
                profile.set_shape(
                    input_name,
                    [minBS, action_horizon, hidden_dim],
                    [optBS, action_horizon, hidden_dim],
                    [maxBS, action_horizon, hidden_dim],
                )
            elif input_name == "ego_embeds":
                hidden_dim = input_shape[1]
                profile.set_shape(
                    input_name,
                    [minBS, hidden_dim],
                    [optBS, hidden_dim],
                    [maxBS, hidden_dim],
                )
            elif input_name == "timesteps":
                profile.set_shape(input_name, [minBS], [optBS], [maxBS])
            else:
                raise ValueError(f"Unexpected ONNX input name: {input_name}")

        config.add_optimization_profile(profile)

        t0 = time.time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time.time()
        if engineString is None:
            raise RuntimeError(f"Failed building {planFile}")

        print(f"Succeeded building {planFile} in {t1 - t0} s")
        with open(planFile, "wb") as f:
            f.write(engineString)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnxFile", type=str, default="onnx/diffusion_denoising_step.onnx", help="")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the ReCogDrive diffusion planner checkpoint.",
    )
    parser.add_argument(
        "--planFile",
        type=str,
        default="plan/diffusion_denoising_step_fp16.plan",
        help="",
    )
    parser.add_argument(
        "--metadataFile",
        type=str,
        default="",
        help="Path for runtime metadata. Defaults to <planFile> with .metadata.json suffix.",
    )
    parser.add_argument(
        "--only_trt",
        action="store_true",
        help="Run only convert the onnx to TRT engine.",
    )
    parser.add_argument(
        "--sampling_method",
        choices=["ddim", "flow"],
        default="ddim",
    )
    parser.add_argument("--max_vl_seq_len", type=int, default=2800)
    parser.add_argument("--minBS", type=int, default=1)
    parser.add_argument("--optBS", type=int, default=1)
    parser.add_argument("--maxBS", type=int, default=4)
    parser.add_argument("--image_url", nargs="+", default=["./pics/demo.jpeg"])
    args = parser.parse_args()
    return args


def main() -> None:
    args = parse_arguments()
    onnx_file_dir = os.path.dirname(args.onnxFile)
    if onnx_file_dir != "" and not os.path.exists(onnx_file_dir):
        os.makedirs(onnx_file_dir)
    plan_file_dir = os.path.dirname(args.planFile)
    if plan_file_dir != "" and not os.path.exists(plan_file_dir):
        os.makedirs(plan_file_dir)
    metadata_file = args.metadataFile or str(default_metadata_path(args.planFile))
    metadata_file_dir = os.path.dirname(metadata_file)
    if metadata_file_dir != "" and not os.path.exists(metadata_file_dir):
        os.makedirs(metadata_file_dir)

    onnx_trt_obj = ONNX_TRT(args.max_vl_seq_len)

    if args.only_trt:
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.planFile, args.minBS, args.optBS, args.maxBS)
    else:
        onnx_trt_obj.export_onnx(
            args.onnxFile,
            args.pretrained_model_path,
            args.image_url,
            args.sampling_method,
            metadata_file,
        )
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.planFile, args.minBS, args.optBS, args.maxBS)


if __name__ == "__main__":
    main()
