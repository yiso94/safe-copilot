from collections.abc import Callable
from pathlib import Path

import tensorrt as trt
import torch
import torchvision.transforms as T
from PIL import Image
from tensorrt_llm import logger, profiler
from tensorrt_llm.runtime import Session, TensorInfo
from torchvision.transforms import InterpolationMode

try:
    from .utils import trt_dtype_to_torch
except ImportError:
    from models.utils import trt_dtype_to_torch

DEFAULT_IMAGE_SIZE = 448
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Preprocess:
    def __init__(self, input_size: int = DEFAULT_IMAGE_SIZE):
        mean, std = IMAGENET_MEAN, IMAGENET_STD
        self.transform: Callable[[Image.Image], torch.Tensor] = T.Compose(
            [
                T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
                T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
                T.ToTensor(),
                T.Normalize(mean=mean, std=std),
            ]
        )

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio,
            target_ratios,
            orig_width,
            orig_height,
            image_size,
        )

        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for index in range(blocks):
            box = (
                (index % (target_width // image_size)) * image_size,
                (index // (target_width // image_size)) * image_size,
                ((index % (target_width // image_size)) + 1) * image_size,
                ((index // (target_width // image_size)) + 1) * image_size,
            )
            processed_images.append(resized_img.crop(box))

        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        image = Image.open(image_file).convert("RGB")
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [self.transform(item) for item in images]
        return torch.stack(pixel_values)

    def encode(self, image_paths: list[str]) -> torch.Tensor:
        pixel_values_list = [self.load_image(path) for path in image_paths]
        return torch.cat(pixel_values_list, dim=0)


class VisionInfer:
    def __init__(
        self,
        vit_engine_path: str | Path,
        stream: int,
        device: str = "cuda",
    ):
        self.vit_engine_path = Path(vit_engine_path)
        if not self.vit_engine_path.exists():
            raise FileNotFoundError(f"Vision TensorRT engine not found: {self.vit_engine_path}")

        logger.info(f"Loading engine from {vit_engine_path}")
        with open(self.vit_engine_path, "rb") as f:
            engine_buffer = f.read()
        logger.info(f"Creating session from engine {vit_engine_path}")
        self.session = Session.from_serialized_engine(engine_buffer)
        self.device = device
        self.stream = stream

    def infer(self, pixel_values: torch.Tensor) -> torch.Tensor:
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        pixel_values = pixel_values.to(self.device).to(torch.bfloat16).contiguous()

        output_info = self.session.infer_shapes(
            [TensorInfo("pixel_values", trt.DataType.BF16, tuple(pixel_values.shape))]
        )
        outputs = {
            tensor.name: torch.empty(tuple(tensor.shape), dtype=trt_dtype_to_torch(tensor.dtype), device=self.device)
            for tensor in output_info
        }

        profiler.start("ViT")
        ok = self.session.run({"pixel_values": pixel_values}, outputs, self.stream)
        profiler.stop("ViT")
        assert ok, "Runtime execution failed for vit session"

        vit_time = profiler.elapsed_time_in_sec("ViT")
        logger.info(f"TensorRT LLM ViT latency: {vit_time:3f} sec ")
        return outputs["image_embeds"]
