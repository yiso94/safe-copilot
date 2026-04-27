import argparse
import os
import time

import tensorrt as trt
import torch
from PIL import Image
from tensorrt_llm._utils import release_gc, str_dtype_to_torch
from torch import nn
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoModel


class Preprocess:
    def __init__(self, image_size: int):
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
        self.image_transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    def encode(self, image_paths: list[str]) -> torch.Tensor:
        images = []
        for image_path in image_paths:
            image = Image.open(image_path)
            image = image.convert("RGB")
            images.append(self.image_transform(image))
        images = torch.stack(images, dim=0)
        return images


class VisionProjectorWrapper(nn.Module):
    def __init__(self, checkpoint_path: str, torch_dtype: torch.dtype) -> None:
        super().__init__()
        self.model = AutoModel.from_pretrained(
            checkpoint_path,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            use_flash_attn=False,
            device_map="cuda",
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.model.extract_feature(pixel_values)


class ONNX_TRT:
    def __init__(self, image_size: int) -> None:
        self.image_size = image_size

    def export_onnx(
        self,
        onnx_file_path: str,
        pretrained_model_path: str,
        image_url: list[str],
        dtype: str = "float16",
    ) -> None:
        print("Start converting ONNX model!")
        image_pre_obj = Preprocess(self.image_size)
        torch_dtype = str_dtype_to_torch(dtype)
        wrapper = VisionProjectorWrapper(pretrained_model_path, torch_dtype).eval()
        image = image_pre_obj.encode(image_url).cuda().to(torch_dtype)

        torch.onnx.export(
            wrapper,
            image,
            onnx_file_path,
            input_names=["pixel_values"],
            output_names=["image_embeds"],
            opset_version=18,
            dynamic_axes={"pixel_values": {0: "batch"}},
            dynamo=False,
        )
        release_gc()  # Further release memory
        print(f"Export to ONNX file successfully! The ONNX file stays in {onnx_file_path}")

    def generate_trt_engine(
        self,
        onnxFile: str,
        planFile: str,
        minBS: int = 1,
        optBS: int = 2,
        maxBS: int = 4,
        dtype: str = "float16",
    ) -> None:
        print("Start converting TRT engine!")
        logger = trt.Logger(trt.Logger.VERBOSE)
        builder = trt.Builder(logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        profile = builder.create_optimization_profile()
        config = builder.create_builder_config()
        if dtype == "float16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif dtype == "bfloat16":
            if not hasattr(trt.BuilderFlag, "BF16"):
                raise RuntimeError("This TensorRT installation does not expose BuilderFlag.BF16.")
            config.set_flag(trt.BuilderFlag.BF16)
        else:
            raise ValueError(f"Unsupported dtype for TensorRT engine build: {dtype}")
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
        inputT = network.get_input(0)
        inputT.shape = [nBS, 3, self.image_size, self.image_size]
        profile.set_shape(
            inputT.name,
            [nMinBS, 3, self.image_size, self.image_size],
            [nOptBS, 3, self.image_size, self.image_size],
            [nMaxBS, 3, self.image_size, self.image_size],
        )

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
    parser.add_argument(
        "--dtype",
        type=str,
        default="float16",
        choices=["float16", "bfloat16"],
        help="Precision to use for ONNX export and TensorRT engine build.",
    )
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

    onnx_trt_obj = ONNX_TRT(448)  # or ONNX_TRT(config.visual['image_size'])

    if args.only_trt:
        onnx_trt_obj.generate_trt_engine(
            args.onnxFile,
            args.planFile,
            args.minBS,
            args.optBS,
            args.maxBS,
            dtype=args.dtype,
        )
    else:
        onnx_trt_obj.export_onnx(
            args.onnxFile,
            args.pretrained_model_path,
            args.image_url,
            dtype=args.dtype,
        )
        onnx_trt_obj.generate_trt_engine(
            args.onnxFile,
            args.planFile,
            args.minBS,
            args.optBS,
            args.maxBS,
            dtype=args.dtype,
        )


if __name__ == "__main__":
    main()
