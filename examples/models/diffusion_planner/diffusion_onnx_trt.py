import argparse
import os
import time

import tensorrt as trt
import torch
from tensorrt_llm._utils import release_gc, str_dtype_to_torch
from torch import nn
from transformers import BatchFeature

from navsim.agents.recogdrive.recogdrive_agent import ReCogDriveAgent


class Preprocess:
    def __init__(self):
        pass

    def encode(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        last_hidden_state = torch.randn(1, 2800, 1536)
        input_state = torch.randn(1, 20)
        history_trajectory_reshaped = torch.randn(1, 12)
        status_feature = torch.randn(1, 8)
        return last_hidden_state, input_state, history_trajectory_reshaped, status_feature


class DiffusionPlannerWrapper(nn.Module):
    def __init__(self, checkpoint_path: str) -> None:
        super().__init__()
        agent = ReCogDriveAgent(
            None,
            checkpoint_path=checkpoint_path,
            vlm_path="owl10/ReCogDrive-VLM-2B",
            cam_type="single",
            grpo=False,
            cache_hidden_state=False,
            vlm_type="internvl",
            dit_type="small",
            vlm_size="small",
            sampling_method="ddim",
        )
        agent.initialize()
        self.model = agent.action_head

    def forward(
        self,
        last_hidden_state: torch.Tensor,
        input_state: torch.Tensor,
        history_trajectory_reshaped: torch.Tensor,
        status_feature: torch.Tensor,
    ) -> torch.Tensor:
        action_inputs = BatchFeature(
            {
                "state": input_state,
                "his_traj": history_trajectory_reshaped,
                "status_feature": status_feature,
            }
        )
        return self.model.get_action(last_hidden_state, action_inputs)["pred_traj"]


class ONNX_TRT:
    def __init__(self) -> None:
        pass

    def export_onnx(self, onnx_file_path: str, checkpoint_path: str) -> None:
        print("Start converting ONNX model!")
        processer = Preprocess()
        torch_dtype = str_dtype_to_torch("float16")
        wrapper = DiffusionPlannerWrapper(checkpoint_path).eval().cuda().to(torch_dtype)
        inputs = tuple(input.cuda().to(torch_dtype) for input in processer.encode())

        torch.onnx.export(
            wrapper,
            inputs,
            onnx_file_path,
            input_names=["last_hidden_state", "input_state", "history_trajectory_reshaped", "status_feature"],
            output_names=["pred_traj"],
            opset_version=18,
            dynamo=False,
        )
        release_gc()  # Further release memory
        print(f"Export to ONNX file successfully! The ONNX file stays in {onnx_file_path}")

    def generate_trt_engine(self, onnxFile: str, planFile: str, minBS: int = 1, optBS: int = 1, maxBS: int = 1) -> None:
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
            print(f"Succeeded parsing {onnxFile}")

        nBS = -1
        nMinBS = minBS
        nOptBS = optBS
        nMaxBS = maxBS
        for index in range(network.num_inputs):
            inputT = network.get_input(index)
            input_name = inputT.name
            input_shape = list(inputT.shape)

            if input_name == "last_hidden_state":
                max_vl_seq_len = input_shape[1]
                feature_dim = input_shape[2]
                profile.set_shape(
                    input_name,
                    [nMinBS, max_vl_seq_len, feature_dim],
                    [nOptBS, max_vl_seq_len, feature_dim],
                    [nMaxBS, max_vl_seq_len, feature_dim],
                )
            elif input_name == "input_state":
                input_state_dim = input_shape[1]
                profile.set_shape(
                    input_name,
                    [nMinBS, input_state_dim],
                    [nOptBS, input_state_dim],
                    [nMaxBS, input_state_dim],
                )
            elif input_name == "history_trajectory_reshaped":
                history_trajectory_dim = input_shape[1]
                profile.set_shape(
                    input_name,
                    [nMinBS, history_trajectory_dim],
                    [nOptBS, history_trajectory_dim],
                    [nMaxBS, history_trajectory_dim],
                )
            elif input_name == "status_feature":
                status_feature_dim = input_shape[1]
                profile.set_shape(
                    input_name,
                    [nMinBS, status_feature_dim],
                    [nOptBS, status_feature_dim],
                    [nMaxBS, status_feature_dim],
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
    # onnx/visual_encoder
    parser.add_argument("--onnxFile", type=str, default="visual_encoder/visual_encoder.onnx", help="")
    parser.add_argument("--pretrained_model_path", type=str, default="Qwen-VL-Chat", help="")
    parser.add_argument(
        "--planFile",
        type=str,
        default="plan/visual_encoder/visual_encoder_fp16.plan",
        help="",
    )
    parser.add_argument(
        "--only_trt",
        action="store_true",
        help="Run only convert the onnx to TRT engine.",
    )
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

    onnx_trt_obj = ONNX_TRT()  # or ONNX_TRT(config.visual['image_size'])

    if args.only_trt:
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.planFile)
    else:
        onnx_trt_obj.export_onnx(args.onnxFile, args.pretrained_model_path)
        onnx_trt_obj.generate_trt_engine(args.onnxFile, args.planFile)


if __name__ == "__main__":
    main()
