import tensorrt as trt
import torch
from tensorrt_llm import logger, profiler
from tensorrt_llm.runtime import Session, TensorInfo

from .utils import trt_dtype_to_torch


class DiffusionPlanner:
    def __init__(self, diffusion_engine_path: str, stream: int, device: torch.device = torch.device("cuda")):
        logger.info(f"Loading engine from {diffusion_engine_path}")
        with open(diffusion_engine_path, "rb") as f:
            engine_buffer = f.read()
        logger.info(f"Creating session from engine {diffusion_engine_path}")
        self.session = Session.from_serialized_engine(engine_buffer)

        self.device = device

        last_hidden_state_shape = (1, 2800, 1536)
        history_trajectory_reshaped_shape = (1, 12)
        status_feature_shape = (1, 8)
        traj_output_info = self.session.infer_shapes(
            [
                TensorInfo(
                    "last_hidden_state",
                    trt.DataType.HALF,
                    last_hidden_state_shape,
                ),
                TensorInfo(
                    "history_trajectory_reshaped",
                    trt.DataType.HALF,
                    history_trajectory_reshaped_shape,
                ),
                TensorInfo(
                    "status_feature",
                    trt.DataType.HALF,
                    status_feature_shape,
                ),
            ]
        )
        self.traj_outputs = {
            t.name: torch.empty(tuple(t.shape), dtype=trt_dtype_to_torch(t.dtype), device=self.device)
            for t in traj_output_info
        }
        self.stream = stream

    def infer(
        self,
        last_hidden_state: torch.Tensor,
        input_state: torch.Tensor,
        history_trajectory_reshaped: torch.Tensor,
        status_feature: torch.Tensor,
    ):
        traj_inputs = {
            "last_hidden_state": last_hidden_state.half().to(self.device),
            "history_trajectory_reshaped": history_trajectory_reshaped.half().to(self.device),
            "status_feature": status_feature.half().to(self.device),
        }

        profiler.start("diffusion")
        ok = self.session.run(traj_inputs, self.traj_outputs, self.stream)
        profiler.stop("diffusion")

        diffusion_time = profiler.elapsed_time_in_sec("diffusion")
        logger.info(f"TensorRT LLM diffusion latency: {diffusion_time:3f} sec ")

        assert ok, "Runtime execution failed for diffusion session"

        pred_traj = self.traj_outputs["pred_traj"]
        return pred_traj


if __name__ == "__main__":
    diffusion_engine_path = "./models/diffusion_planner_fp16.plan"
    stream = torch.cuda.Stream().cuda_stream
    diffusion_planner = DiffusionPlanner(diffusion_engine_path, stream)

    last_hidden_state = torch.randn(1, 2800, 1536)
    input_state = torch.randn(1, 20)
    history_trajectory_reshaped = torch.randn(1, 12)
    status_feature = torch.randn(1, 8)

    pred_traj = diffusion_planner.infer(last_hidden_state, input_state, history_trajectory_reshaped, status_feature)
    print(pred_traj.shape)
