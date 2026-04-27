import argparse
import os
import sys
import time
import warnings
from pathlib import Path

import tensorrt as trt
import torch
from tensorrt_llm._utils import release_gc
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_MODELS_ROOT = REPO_ROOT / "examples" / "models"

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXAMPLES_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_MODELS_ROOT))

from recogdrive.recogdrive_vlm_engines import (  # noqa: E402
    DiffusionPlannerWrapper as _ExportDiffusionPlannerWrapper,
)
from recogdrive.recogdrive_vlm_engines import (
    _load_diffusion_planner_from_checkpoint,
)

DEFAULT_CHECKPOINT = Path(__file__).resolve().parent / "models" / "ReCogDrive_Diffusion_Planner_2B_RL.ckpt"


class DiffusionPlannerWrapper(nn.Module):
    def __init__(self, checkpoint_path: str, sampling_method: str = "ddim") -> None:
        super().__init__()
        planner, spec = _load_diffusion_planner_from_checkpoint(
            checkpoint_path,
            sampling_method=sampling_method,
        )
        self.model = _ExportDiffusionPlannerWrapper(planner).eval()
        self.spec = spec

    def forward(
        self,
        vl_features: torch.Tensor,
        his_traj: torch.Tensor,
        status_feature: torch.Tensor,
        init_actions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(vl_features, his_traj, status_feature, init_actions)


class ONNX_TRT:
    def __init__(self, max_vl_seq_len: int) -> None:
        self.max_vl_seq_len = max_vl_seq_len

    def export_onnx(
        self,
        onnx_file_path: str,
        checkpoint_path: str,
        image_url: list[str] | None = None,
        sampling_method: str = "ddim",
    ) -> None:
        print("Start converting ONNX model!")
        if image_url:
            warnings.warn("--image_url is ignored for diffusion planner export.")

        wrapper = DiffusionPlannerWrapper(
            checkpoint_path=checkpoint_path,
            sampling_method=sampling_method,
        ).eval()
        spec = wrapper.spec
        example_seq_len = min(self.max_vl_seq_len, max(spec.action_horizon, 256))
        example_vl_features = torch.zeros(
            1,
            example_seq_len,
            spec.feature_dim,
            device="cuda",
            dtype=torch.float16,
        )
        example_his_traj = torch.zeros(
            1,
            spec.his_traj_dim,
            device="cuda",
            dtype=torch.float16,
        )
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

        torch.onnx.export(
            wrapper,
            (
                example_vl_features,
                example_his_traj,
                example_status_feature,
                example_init_actions,
            ),
            onnx_file_path,
            input_names=["vl_features", "his_traj", "status_feature", "init_actions"],
            output_names=["pred_traj"],
            opset_version=18,
            external_data=True,
            do_constant_folding=False,
            dynamic_axes={
                "vl_features": {0: "batch", 1: "vl_seq_len"},
                "his_traj": {0: "batch"},
                "status_feature": {0: "batch"},
                "init_actions": {0: "batch"},
                "pred_traj": {0: "batch"},
            },
            dynamo=False,
        )
        release_gc()  # Further release memory
        print(f"Export to ONNX file successfully! The ONNX file stays in {onnx_file_path}")

    def generate_trt_engine(self, onnxFile: str, planFile: str, minBS: int = 1, optBS: int = 2, maxBS: int = 4) -> None:
        print("Start converting TRT engine!")
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.FP16)
        parser = trt.OnnxParser(network, logger)

        with open(onnxFile, "rb") as model:
            if not parser.parse(model.read(), "/".join(onnxFile.split("/"))):
                print(f"Failed parsing {onnxFile}")
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError(f"Failed parsing {onnxFile}")
            print(f"Succeeded parsing {onnxFile}")

        for index in range(network.num_inputs):
            inputT = network.get_input(index)
            input_name = inputT.name
            input_shape = list(inputT.shape)

            if input_name == "vl_features":
                feature_dim = input_shape[2]
                profile.set_shape(
                    input_name,
                    [minBS, 1, feature_dim],
                    [optBS, min(self.max_vl_seq_len, 256), feature_dim],
                    [maxBS, self.max_vl_seq_len, feature_dim],
                )
            elif input_name == "his_traj":
                his_traj_dim = input_shape[1]
                profile.set_shape(
                    input_name,
                    [minBS, his_traj_dim],
                    [optBS, his_traj_dim],
                    [maxBS, his_traj_dim],
                )
            elif input_name == "status_feature":
                status_feature_dim = input_shape[1]
                profile.set_shape(
                    input_name,
                    [minBS, status_feature_dim],
                    [optBS, status_feature_dim],
                    [maxBS, status_feature_dim],
                )
            elif input_name == "init_actions":
                action_horizon = input_shape[1]
                action_dim = input_shape[2]
                profile.set_shape(
                    input_name,
                    [minBS, action_horizon, action_dim],
                    [optBS, action_horizon, action_dim],
                    [maxBS, action_horizon, action_dim],
                )
            else:
                raise ValueError(f"Unexpected ONNX input name: {input_name}")

        config.add_optimization_profile(profile)

        t0 = time.time()
        engineString = builder.build_serialized_network(network, config)
        t1 = time.time()
        if engineString is None:
            print(f"Failed building {planFile}")
        else:
            print(f"Succeeded building {planFile} in {t1 - t0} s")
            with open(planFile, "wb") as f:
                f.write(engineString)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnxFile", type=str, default="onnx/diffusion_planner.onnx", help="")
    parser.add_argument(
        "--pretrained_model_path",
        type=str,
        default=str(DEFAULT_CHECKPOINT),
        help="Path to the ReCogDrive diffusion planner checkpoint.",
    )
    parser.add_argument(
        "--planFile",
        type=str,
        default="plan/diffusion_planner_fp16.plan",
        help="",
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
    if not onnx_file_dir == "" and not os.path.exists(onnx_file_dir):
        os.makedirs(onnx_file_dir)
    plan_file_dir = os.path.dirname(args.planFile)
    if not os.path.exists(plan_file_dir):
        os.makedirs(plan_file_dir)

    onnx_trt_obj = ONNX_TRT(args.max_vl_seq_len)

    if args.only_trt:
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.planFile, args.minBS, args.optBS, args.maxBS)
    else:
        onnx_trt_obj.export_onnx(
            args.onnxFile,
            args.pretrained_model_path,
            args.image_url,
            args.sampling_method,
        )
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.planFile, args.minBS, args.optBS, args.maxBS)


if __name__ == "__main__":
    main()
