import argparse
import json
import os
import sys
from datetime import datetime
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import ImageDraw
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.transforms.transforms import InterpolationMode
from tqdm import tqdm

from model import MNISTModel
from utils import EncodeTensor


def train(
    epochs: int = 20,
    learning_rate: float = 1e-3,
    batch_size: int = 16,
    device: Literal["cpu", "cuda", "mps"] = "cpu",
    save_images: bool = False,
) -> None:
    print(f"Using device: {device}")

    print("Downloading MNIST...")
    training_data = datasets.MNIST(
        root="data", train=True, download=True, transform=ToTensor()
    )

    validation_data = datasets.MNIST(
        root="data", train=False, download=True, transform=ToTensor()
    )

    train_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
    validation_loader = DataLoader(
        validation_data, batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(validation_data, batch_size=1, shuffle=False)

    print("MNIST Downloaded")

    model = MNISTModel().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    loss_fn = nn.CrossEntropyLoss()

    best_loss = sys.float_info.max
    for epoch in tqdm(range(epochs)):
        train_loss = 0

        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Forward pass
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = loss_fn(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Calculate metrics
            train_loss += loss.item()

        # Calculate epoch metrics
        epoch_loss = train_loss / len(train_loader)

        # Validation
        val_loss = 0

        with torch.no_grad():
            for data, target in validation_loader:
                data = data.to(device)
                target = target.to(device)

                output = model(data)
                loss = loss_fn(output, target)

                val_loss += loss.item()

        # Calculate and log epoch validation metrics
        val_loss /= len(validation_loader)

        # Log epoch metrics
        print(
            f"Epoch {epoch + 1}/{epochs}, Train loss: {epoch_loss:.3f} Val loss: {val_loss:.3f};",
        )

        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if val_loss < best_loss:
            best_loss = val_loss
            base_path = f"./data/models/{timestamp}"
            os.makedirs(base_path, exist_ok=True)

            model_path = "model_{}_{}".format(timestamp, epoch + 1)
            torch.save(model.state_dict(), f"{base_path}/{model_path}.pt")

            base_path = "./data/models"
            torch.save(model.state_dict(), f"{base_path}/best.pt")
            with open(f"{base_path}/torch_weights.json", "w") as json_file:
                json.dump(model.state_dict(), json_file, cls=EncodeTensor)

    # Final evaluation
    test_loss = 0

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    accuracy = 0
    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, labels)
            test_loss += loss.item()

            if save_images:
                image = TF.to_pil_image(
                    TF.resize(
                        inputs.squeeze(0),
                        size=[320, 320],
                        interpolation=InterpolationMode.NEAREST,
                    )
                )
                draw = ImageDraw.Draw(image)
                draw.text(
                    (10, 10), f"Ground Truth: {labels.item()}", fill="white", size=20
                )
                draw.text(
                    (10, 30),
                    f"Predicted: {outputs.argmax().item()}",
                    fill="white",
                    size=20,
                )

                base_path = f"./data/results/{timestamp}"
                os.makedirs(base_path, exist_ok=True)
                image.save(f"{base_path}/{i:08d}.png")

            outputs = F.softmax(outputs, dim=-1)
            accuracy += outputs.argmax().item() == labels.item()

    # Calculate and log final test metrics
    test_loss /= len(test_loader)
    accuracy /= len(test_loader)

    print(f"Final Test Loss: {test_loss:.3f};")
    print(f"Final Test Accuracy: {accuracy:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Train",
        description="Train the MNIST model",
    )

    parser.add_argument("-e", "--epochs", type=int, default=20)
    parser.add_argument("-l", "--lr", type=float, default=0.001)
    parser.add_argument("-b", "--batch", type=int, default=64)
    parser.add_argument(
        "-d",
        "--device",
        type=str,
        default="cpu",
        help='Select device to run the training ("cpu", "cuda", "mps")',
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Enables saving validation images after training",
    )
    args = parser.parse_args()

    train(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch,
        device=args.device,
        save_images=args.save_images,
    )
