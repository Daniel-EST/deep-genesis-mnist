import argparse
import glob
import os

import torch

from utils import preprocess_image, write_array


def generate(input: str, output: str) -> None:
    images = []
    if os.path.isdir(input):
        for file in glob.glob(os.path.join(input, "*.jpg")):
            image = preprocess_image(file).squeeze(0)
            images.append(image)
    if os.path.isfile(input):
        image = preprocess_image(input).squeeze(0)
        images.append(image)

    size = len(images)
    images = torch.cat(images, dim=0)
    with open(output, "w") as f:
        f.write(f"/* Auto-generated from {input} */\n")
        f.write("#ifndef _MNIST_H_\n")
        f.write("#define _MNIST_H_\n\n")
        f.write("#include <genesis.h>\n\n")
        f.write("#define IMG_H 28\n")
        f.write("#define IMG_W 28\n\n")

        f.write(f"#define IMG_COUNT {size}\n\n")

        write_array(f, "images", images.tolist(), [size, 28, 28])

        f.write("#endif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Image2Header",
        description="Converts 28x28 MNIST images to C header file",
    )
    parser.add_argument(
        "input",
        help="Path to JSON weights file",
    )
    parser.add_argument(
        "-o", "--output", default="./inc/mnist.h", help="Output C header file path"
    )

    args = parser.parse_args()
    generate(args.input, args.output)
