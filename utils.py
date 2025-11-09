from json import JSONEncoder
from typing import Any, Iterator

import numpy as np
import torch
import torchvision
from torch.utils.data import Dataset
from torchvision.io.image import ImageReadMode


class EncodeTensor(JSONEncoder, Dataset):
    def default(self, o: Any):
        if isinstance(o, torch.Tensor):
            return o.cpu().detach().numpy().tolist()
        return super(EncodeTensor, self).default(o)


def preprocess_image(input: str) -> torch.Tensor:
    image = torchvision.io.read_image(input, ImageReadMode.GRAY).flatten().unsqueeze(0)

    image = image.float()
    arr_min, arr_max = image.min(), image.max()

    if arr_max != arr_min:
        if arr_max > 1 or arr_min < 0:
            image = (image - arr_min) / (arr_max - arr_min)
    else:
        image = torch.zeros_like(image)

    return image


def get_shape(x: Any) -> list[int]:
    shape = []
    while isinstance(x, list):
        shape.append(len(x))
        if len(x) == 0:
            break
        x = x[0]
    return shape


def flatten(x: list[Any] | float) -> Iterator[np.float16]:
    if isinstance(x, list):
        for el in x:
            yield from flatten(el)
    else:
        yield np.float16(x)


def write_array(
    f: Any,
    name: str,
    values: list[np.float16],
    comment_shape: torch.Size | list[int] | None = None,
    per_line: int = 8,
) -> None:
    if comment_shape is not None:
        f.write(f"// {name} shape: [")
        f.write(", ".join(str(s) for s in comment_shape))
        f.write("]\n")

    n = len(values)

    f.write(f"extern const fix32 {name}[{n}];\n\n")
    f.write(f"const fix32 {name}[{n}] = {{\n")

    for i, v in enumerate(values):
        if i % per_line == 0:
            f.write("    ")
        f.write(f"FIX32({v:.3f})")
        if i != n - 1:
            f.write(", ")
        if (i + 1) % per_line == 0:
            f.write("\n")

    if n % per_line != 0:
        f.write("\n")

    f.write("};\n\n")
