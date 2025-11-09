import argparse
from typing import Literal

import torch
import torch.nn.functional as F

from model import MNISTModel
from utils import preprocess_image


def predict(
    input: str,
    weights: str,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
) -> None:
    print(f"Using device: {device}")
    model = MNISTModel().to(device)

    # Load the model weights
    checkpoint = torch.load(weights, weights_only=True)
    model.load_state_dict(checkpoint)

    # Preprocess the input image
    image = preprocess_image(input)

    with torch.no_grad():
        image = image.to(device)
        output = model(image)

        label = F.softmax(output, dim=-1).argmax().item()
        probs = "".join(
            [
                f"\n  {i}: {p:.4f}"
                for i, p in enumerate(F.softmax(output, dim=-1).tolist()[0])
            ]
        )
        logits = "".join(
            [f"\n  {i}: {p:.4f}" for i, p in enumerate(output.tolist()[0])]
        )

        print(f"Logits:{logits}\nProbs:{probs}\nLabel: {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Predict",
        description="Predict the label of an image",
    )
    parser.add_argument("input", type=str, help="Path to the input image")
    parser.add_argument(
        "-w",
        "--weights",
        type=str,
        default="./data/models/best.pt",
        help="Path to the model weights",
    )
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help='Device to use for prediction ("cpu", "cuda", "mps")',
    )
    args = parser.parse_args()

    predict(input=args.input, weights=args.weights, device=args.device)
