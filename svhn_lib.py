import copy
import math
import os
import random
import re
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

SEED = 42
N_CLASSES = 10
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_parquet_split(path):
    import io

    import pandas as pd
    from PIL import Image

    df = pd.read_parquet(path)
    X = np.stack([np.asarray(Image.open(io.BytesIO(im["bytes"])).convert("RGB"), dtype=np.uint8)
                  for im in df["image"]]).transpose(0, 3, 1, 2)
    return torch.from_numpy(np.ascontiguousarray(X)), torch.tensor(df["label"].to_numpy(), dtype=torch.int64)


def load_svhn(root: str = "data", hf_root: str = "data_hf"):
    import torchvision

    out = []
    for split in ("train", "test"):
        pq = Path(hf_root) / f"{split}.parquet"
        if not (Path(root) / f"{split}_32x32.mat").exists() and pq.exists():
            out.append(load_parquet_split(pq))
            continue
        ds = torchvision.datasets.SVHN(root=root, split=split, download=True)
        X = torch.from_numpy(ds.data)
        y = torch.from_numpy(ds.labels.astype(np.int64))
        out.append((X, y))
    return out[0], out[1]


def split_train_val(n: int, val_frac: float = 0.1, seed: int = SEED):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(round(n * val_frac))
    return perm[n_val:], perm[:n_val]


def compute_norm_stats(X: torch.Tensor):
    means, stds = [], []
    for c in range(X.shape[1]):
        xc = X[:, c].to(torch.int64)
        n, s, ss = xc.numel(), int(xc.sum()), int((xc * xc).sum())
        means.append(s / n / 255.0)
        stds.append(((ss * n - s * s) / (n * (n - 1))) ** 0.5 / 255.0)
    return torch.tensor(means, dtype=torch.float32), torch.tensor(stds, dtype=torch.float32)


def preprocess(X: torch.Tensor, mean=None, std=None) -> torch.Tensor:
    Xf = X.float() / 255.0
    if mean is not None:
        Xf = (Xf - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    return Xf


def per_image_norm(Xf: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    m = Xf.mean(dim=(1, 2, 3), keepdim=True)
    s = Xf.std(dim=(1, 2, 3), keepdim=True)
    return (Xf - m) / (s + eps)


def random_affine(x: torch.Tensor, max_shift: int, max_rot: float, max_scale: float,
                  gen: torch.Generator) -> torch.Tensor:
    B = len(x)
    ang = (torch.rand(B, generator=gen) * 2 - 1) * math.radians(max_rot)
    sc = 1 + (torch.rand(B, generator=gen) * 2 - 1) * max_scale
    t = (torch.rand(B, 2, generator=gen) * 2 - 1) * (2 * max_shift / x.shape[-1])
    theta = torch.zeros(B, 2, 3)
    theta[:, 0, 0] = torch.cos(ang) / sc
    theta[:, 0, 1] = -torch.sin(ang) / sc
    theta[:, 1, 0] = torch.sin(ang) / sc
    theta[:, 1, 1] = torch.cos(ang) / sc
    theta[:, :, 2] = t
    grid = F.affine_grid(theta.to(x.device), list(x.shape), align_corners=False)
    return F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)


def shift_fixed(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return x
    p = max(abs(dx), abs(dy))
    H, W = x.shape[-2:]
    xp = F.pad(x, (p, p, p, p), mode="replicate")
    return xp[:, :, p + dy:p + dy + H, p + dx:p + dx + W]


def random_shift(x: torch.Tensor, max_shift: int, gen: torch.Generator) -> torch.Tensor:
    if max_shift <= 0:
        return x
    B, C, H, W = x.shape
    p = max_shift
    xp = F.pad(x, (p, p, p, p), mode="replicate")
    dx = torch.randint(0, 2 * p + 1, (B,), generator=gen).to(x.device)
    dy = torch.randint(0, 2 * p + 1, (B,), generator=gen).to(x.device)
    rows = dy[:, None] + torch.arange(H, device=x.device)
    cols = dx[:, None] + torch.arange(W, device=x.device)
    b = torch.arange(B, device=x.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[b, rows[:, :, None], cols[:, None, :]]
    return out.permute(0, 3, 1, 2).contiguous()


class LinearClassifier(nn.Module):
    def __init__(self, in_features: int = 3 * 32 * 32, n_classes: int = N_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


class MLP(nn.Module):
    def __init__(self, hidden_sizes, dropout: float = 0.0, batchnorm: bool = False,
                 in_features: int = 3 * 32 * 32, n_classes: int = N_CLASSES):
        super().__init__()
        self.config = dict(hidden_sizes=list(hidden_sizes), dropout=dropout, batchnorm=batchnorm)
        layers, prev = [], in_features
        for h in hidden_sizes:
            layers.append(nn.Linear(prev, h))
            if batchnorm:
                layers.append(nn.BatchNorm1d(h))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1))


def _conv_bn(cin, cout):
    return [nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU()]


class CNN(nn.Module):
    def __init__(self, dropout: float = 0.3, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            *_conv_bn(3, 32), *_conv_bn(32, 32), nn.MaxPool2d(2),
            *_conv_bn(32, 64), *_conv_bn(64, 64), nn.MaxPool2d(2),
            *_conv_bn(64, 128), *_conv_bn(128, 128), nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(dropout),
            nn.Linear(128 * 4 * 4, 256), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(model, X, y, batch_size: int = 2048, device=None):
    device = device or next(model.parameters()).device
    model.eval()
    total_loss, correct = 0.0, 0
    for i in range(0, len(X), batch_size):
        xb, yb = X[i:i + batch_size].to(device), y[i:i + batch_size].to(device)
        out = model(xb)
        total_loss += F.cross_entropy(out, yb, reduction="sum").item()
        correct += (out.argmax(1) == yb).sum().item()
    return total_loss / len(X), correct / len(X)


def train_model(model, data, epochs: int, batch_size: int, lr: float,
                optimizer: str = "adam", weight_decay: float = 0.0,
                scheduler: str | None = None, augment_shift: int = 0,
                augment_rot: float = 0.0, augment_scale: float = 0.0,
                seed: int = SEED, device=None, verbose: bool = True):
    device = device or get_device()
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model.to(device)

    if optimizer == "adam":
        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "adamw":
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer == "sgd":
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"otimizador desconhecido: {optimizer}")
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs) if scheduler == "cosine" else None

    X_tr, y_tr = data["X_train"], data["y_train"]
    has_val = "X_val" in data
    n = len(X_tr)
    onecycle = None
    if scheduler == "onecycle":
        onecycle = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=lr, total_steps=epochs * math.ceil(n / batch_size), pct_start=0.1)
    elif scheduler not in (None, "cosine"):
        raise ValueError(f"scheduler desconhecido: {scheduler}")
    hist = {k: [] for k in ("train_loss", "train_acc", "val_loss", "val_acc", "epoch_time")}
    best_acc, best_state, best_epoch = -1.0, None, -1

    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        perm = torch.randperm(n, generator=gen)
        run_loss, run_correct = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = X_tr[idx]
            if augment_rot or augment_scale:
                xb = random_affine(xb, augment_shift, augment_rot, augment_scale, gen)
            elif augment_shift:
                xb = random_shift(xb, augment_shift, gen)
            xb, yb = xb.to(device), y_tr[idx].to(device)

            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            if onecycle:
                onecycle.step()

            run_loss += loss.item() * len(idx)
            run_correct += (out.argmax(1) == yb).sum().item()
        if sched:
            sched.step()

        hist["train_loss"].append(run_loss / n)
        hist["train_acc"].append(run_correct / n)
        msg = (f"época {epoch + 1:2d}/{epochs} | treino loss {hist['train_loss'][-1]:.4f} "
               f"acc {hist['train_acc'][-1]:.4f}")
        if has_val:
            val_loss, val_acc = evaluate(model, data["X_val"], data["y_val"], device=device)
            hist["val_loss"].append(val_loss)
            hist["val_acc"].append(val_acc)
            msg += f" | val loss {val_loss:.4f} acc {val_acc:.4f}"
            if val_acc > best_acc:
                best_acc, best_epoch = val_acc, epoch
                best_state = copy.deepcopy(model.state_dict())
        hist["epoch_time"].append(time.time() - t0)
        if verbose:
            print(f"{msg} | {hist['epoch_time'][-1]:.1f}s")

    if has_val:
        model.load_state_dict(best_state)
        hist["best_epoch"], hist["best_val_acc"] = best_epoch, best_acc
    else:
        hist["best_epoch"], hist["best_val_acc"] = epochs - 1, None
    return hist


def confusion_matrix(y_true, y_pred, n_classes: int = N_CLASSES) -> torch.Tensor:
    idx = y_true.long() * n_classes + y_pred.long()
    return torch.bincount(idx, minlength=n_classes * n_classes).view(n_classes, n_classes)


@torch.no_grad()
def predict(model, X, batch_size: int = 2048, device=None) -> torch.Tensor:
    device = device or next(model.parameters()).device
    model.eval()
    preds = [model(X[i:i + batch_size].to(device)).argmax(1).cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(preds).long()


@torch.no_grad()
def predict_proba(model, X, tta_radius: int = 0, batch_size: int = 2048, device=None) -> torch.Tensor:
    device = device or next(model.parameters()).device
    model.eval()
    r = tta_radius
    out = []
    for i in range(0, len(X), batch_size):
        xb = X[i:i + batch_size].to(device)
        p = torch.zeros(len(xb), N_CLASSES, device=device)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                p += F.softmax(model(shift_fixed(xb, dx, dy)), dim=1)
        out.append((p / (2 * r + 1) ** 2).cpu())
    return torch.cat(out)


def predict_tta(model, X, tta_radius: int = 0, batch_size: int = 2048, device=None) -> torch.Tensor:
    return predict_proba(model, X, tta_radius, batch_size, device).argmax(1).long()


def plot_history(hist, title: str = "", path=None):
    import matplotlib.pyplot as plt

    ep = range(1, len(hist["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(ep, hist["train_loss"], label="treino")
    a1.set(xlabel="época", ylabel="loss (cross-entropy)", title="Loss")
    a2.plot(ep, hist["train_acc"], label="treino")
    if hist["val_loss"]:
        a1.plot(ep, hist["val_loss"], label="validação")
        a2.plot(ep, hist["val_acc"], label="validação")
        a2.axvline(hist["best_epoch"] + 1, ls=":", c="gray", label="melhor época (val)")
    a2.set(xlabel="época", ylabel="acurácia", title="Acurácia")
    from matplotlib.ticker import MaxNLocator
    for a in (a1, a2):
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


def natural_sort_key(name: str):
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(name))]


def load_image_folder(folder):
    from PIL import Image

    folder = Path(folder)
    names = sorted((p.name for p in folder.iterdir()
                    if p.is_file() and p.suffix.lower() in IMG_EXTS), key=natural_sort_key)
    arrs = []
    for nm in names:
        with Image.open(folder / nm) as im:
            im = im.convert("RGB")
            if im.size != (32, 32):
                im = im.resize((32, 32), Image.BILINEAR)
            arrs.append(np.asarray(im, dtype=np.uint8).transpose(2, 0, 1))
    X = torch.from_numpy(np.stack(arrs)) if arrs else torch.empty(0, 3, 32, 32, dtype=torch.uint8)
    return names, X


def write_results(preds, path) -> None:
    vals = [int(v) for v in (preds.tolist() if hasattr(preds, "tolist") else preds)]
    if any(v < 0 or v > 9 for v in vals):
        raise ValueError("predições devem ser inteiros entre 0 e 9")
    Path(path).write_text("".join(f"{v}\n" for v in vals))
