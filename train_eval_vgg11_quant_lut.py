import argparse
import copy
import os
import random
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def quantize_8bit_unit_range(x: torch.Tensor) -> torch.Tensor:
    """Clamp to [-1, 1] and quantize to signed 8-bit levels (256 levels)."""
    x = torch.clamp(x, -1.0, 1.0)
    scale = 255.0 / 2.0  # map [-1, 1] -> [0, 255]
    x_q = torch.round((x + 1.0) * scale) / scale - 1.0
    return x_q


class STEQuantizeFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        return quantize_8bit_unit_range(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


class QuantizedReLU(nn.Module):
    """ReLU with 8-bit quantized input/output in [-1, 1]."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q_in = STEQuantizeFn.apply(x)
        y = F.relu(x_q_in)
        y_q_out = STEQuantizeFn.apply(y)
        return y_q_out


class LUTReLU(nn.Module):
    """LUT-based activation using precomputed 8-bit mappings."""

    def __init__(self, lut_outputs_256: np.ndarray):
        super().__init__()
        if lut_outputs_256.shape[0] != 256:
            raise ValueError("LUT must have exactly 256 output values.")
        self.register_buffer("lut", torch.tensor(lut_outputs_256, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = quantize_8bit_unit_range(x)
        idx = torch.round((x_q + 1.0) * (255.0 / 2.0)).long().clamp(0, 255)
        y = self.lut[idx]
        y_q = quantize_8bit_unit_range(y)
        return y_q


class VGG11Quant(nn.Module):
    def __init__(self, in_channels: int, num_classes: int, activation_ctor):
        super().__init__()
        cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        layers = []
        c = in_channels
        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(c, v, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(v))
                layers.append(activation_ctor())
                c = v
        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            activation_ctor(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            activation_ctor(),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


@dataclass
class DatasetConfig:
    name: str
    in_channels: int
    num_classes: int
    train_set: torch.utils.data.Dataset
    test_set: torch.utils.data.Dataset


def get_dataset_config(name: str, data_root: str) -> DatasetConfig:
    if name == "mnist":
        train_tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        test_tf = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_set = datasets.MNIST(root=data_root, train=True, download=True, transform=train_tf)
        test_set = datasets.MNIST(root=data_root, train=False, download=True, transform=test_tf)
        return DatasetConfig(name, 1, 10, train_set, test_set)

    if name == "cifar10":
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])
        train_set = datasets.CIFAR10(root=data_root, train=True, download=True, transform=train_tf)
        test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=test_tf)
        return DatasetConfig(name, 3, 10, train_set, test_set)

    raise ValueError(f"Unsupported dataset: {name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total = 0
    correct = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        pred = logits.argmax(1)
        correct += (pred == labels).sum().item()
        total += labels.size(0)
    return correct / total


def read_and_interpolate_lut(lut_xlsx_path: str, save_interp_csv: str | None = None):
    if not os.path.exists(lut_xlsx_path):
        raise FileNotFoundError(
            f"Cannot find LUT file: {lut_xlsx_path}. Please put LUT_ReLU.xlsx in the repository root or pass --lut-xlsx."
        )

    df = pd.read_excel(lut_xlsx_path, header=None)
    if df.shape[1] < 2:
        raise ValueError("LUT excel must have at least two columns: input and output.")

    x_raw = df.iloc[:, 0].astype(float).to_numpy()
    y_raw = df.iloc[:, 1].astype(float).to_numpy()

    order = np.argsort(x_raw)
    x_raw = x_raw[order]
    y_raw = y_raw[order]

    x_new = np.linspace(-1.0, 1.0, 256)
    y_new = np.interp(x_new, x_raw, y_raw)
    y_new = np.clip(y_new, -1.0, 1.0)

    if save_interp_csv is not None:
        pd.DataFrame({"input_8bit": x_new, "output_8bit": y_new}).to_csv(save_interp_csv, index=False)

    return x_new, y_new


def replace_activations_with_lut(module: nn.Module, lut_outputs_256: np.ndarray):
    for name, child in module.named_children():
        if isinstance(child, QuantizedReLU):
            setattr(module, name, LUTReLU(lut_outputs_256))
        else:
            replace_activations_with_lut(child, lut_outputs_256)


def train_and_compare_for_dataset(args, dataset_name, lut_outputs_256, device):
    cfg = get_dataset_config(dataset_name, args.data_root)
    train_loader = DataLoader(cfg.train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    test_loader = DataLoader(cfg.test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = VGG11Quant(cfg.in_channels, cfg.num_classes, activation_ctor=QuantizedReLU).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        test_acc = evaluate(model, test_loader, device)
        scheduler.step()
        print(
            f"[{dataset_name}] Epoch {epoch:02d}/{args.epochs:02d} "
            f"loss={train_loss:.4f} train_acc={train_acc*100:.2f}% val_acc={test_acc*100:.2f}%"
        )

    standard_acc = evaluate(model, test_loader, device)

    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, f"vgg11_{dataset_name}_quant8_standard_relu.pth")
    torch.save(model.state_dict(), model_path)

    lut_model = copy.deepcopy(model)
    replace_activations_with_lut(lut_model, lut_outputs_256)
    lut_acc = evaluate(lut_model, test_loader, device)

    return model_path, standard_acc, lut_acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train/evaluate 8-bit quantized VGG11 on MNIST and CIFAR-10 with standard ReLU vs LUT-ReLU.")
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--output-dir", type=str, default="./outputs_vgg11_lut")
    parser.add_argument("--lut-xlsx", type=str, default="./LUT_ReLU.xlsx")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    interp_csv_path = os.path.join(args.output_dir, "lut_relu_interpolated_8bit.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    _, lut_outputs_256 = read_and_interpolate_lut(args.lut_xlsx, save_interp_csv=interp_csv_path)
    print(f"Interpolated 8-bit LUT saved to: {interp_csv_path}")

    all_results = {}
    for dataset_name in ["mnist", "cifar10"]:
        model_path, std_acc, lut_acc = train_and_compare_for_dataset(args, dataset_name, lut_outputs_256, device)
        all_results[dataset_name] = {
            "model_path": model_path,
            "standard_relu_acc": std_acc,
            "lut_relu_acc": lut_acc,
        }

    print("\n========== Final Accuracy Comparison ==========")
    print(f"MNIST  - standard ReLU: {all_results['mnist']['standard_relu_acc']*100:.2f}%")
    print(f"MNIST  - LUT ReLU     : {all_results['mnist']['lut_relu_acc']*100:.2f}%")
    print(f"CIFAR10- standard ReLU: {all_results['cifar10']['standard_relu_acc']*100:.2f}%")
    print(f"CIFAR10- LUT ReLU     : {all_results['cifar10']['lut_relu_acc']*100:.2f}%")


if __name__ == "__main__":
    main()
