import tensorrt as trt
import torch


def trt_dtype_to_torch(dtype):
    if dtype == trt.float16:
        return torch.float16
    elif dtype == trt.bfloat16:
        return torch.bfloat16
    elif dtype == trt.float32:
        return torch.float32
    elif dtype == trt.int32:
        return torch.int32
    else:
        raise TypeError(f"{dtype} is not a supported")


def str_dtype_to_torch(dtype: str):
    if dtype == "float16":
        return torch.float16
    elif dtype == "bfloat16":
        return torch.bfloat16
    elif dtype == "float32":
        return torch.float32
    else:
        raise TypeError(f"{dtype} is not a supported dtype string")
