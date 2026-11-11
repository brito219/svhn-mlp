"""Funções de apoio do trabalho de classificação de dígitos no SVHN.

Tudo que o notebook usa fica aqui para poder ser testado (tests/) e reaproveitado
pelo script de predição do conjunto adicional.
"""
import copy
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


# ---------------------------------------------------------------------------
# Reprodutibilidade
# ---------------------------------------------------------------------------

def set_seed(seed: int = SEED) -> None:
    """Fixa todas as sementes (Python, NumPy, PyTorch CPU/GPU)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

def load_svhn(root: str = "data"):
    """Carrega os splits 'train' e 'test' do SVHN como tensores uint8 (N, 3, 32, 32).

    O torchvision já converte o rótulo 10 (dígito zero no .mat original) para 0.
    """
    import torchvision

    out = []
    for split in ("train", "test"):
        ds = torchvision.datasets.SVHN(root=root, split=split, download=True)
        X = torch.from_numpy(ds.data)  # já vem em (N, 3, 32, 32), uint8
        y = torch.from_numpy(ds.labels.astype(np.int64))
        out.append((X, y))
    return out[0], out[1]


def split_train_val(n: int, val_frac: float = 0.1, seed: int = SEED):
    """Índices de treino/validação: embaralhamento fixo pela semente."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_val = int(round(n * val_frac))
    return perm[n_val:], perm[:n_val]


def compute_norm_stats(X: torch.Tensor):
    """Média e desvio padrão por canal (em escala [0, 1]), calculados só no treino."""
    Xf = X.float() / 255.0
    return Xf.mean(dim=(0, 2, 3)), Xf.std(dim=(0, 2, 3))


def preprocess(X: torch.Tensor, mean=None, std=None) -> torch.Tensor:
    """uint8 -> float em [0, 1]; se houver mean/std, normaliza por canal."""
    Xf = X.float() / 255.0
    if mean is not None:
        Xf = (Xf - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    return Xf


def random_shift(x: torch.Tensor, max_shift: int, gen: torch.Generator) -> torch.Tensor:
    """Data augmentation: translada cada imagem até `max_shift` pixels (borda replicada).

    Não usamos flip horizontal porque ele muda o significado de alguns dígitos.
    """
    if max_shift <= 0:
        return x
    B, C, H, W = x.shape
    p = max_shift
    xp = F.pad(x, (p, p, p, p), mode="replicate")
    dx = torch.randint(0, 2 * p + 1, (B,), generator=gen).to(x.device)
    dy = torch.randint(0, 2 * p + 1, (B,), generator=gen).to(x.device)
    rows = dy[:, None] + torch.arange(H, device=x.device)  # (B, H)
    cols = dx[:, None] + torch.arange(W, device=x.device)  # (B, W)
    b = torch.arange(B, device=x.device)[:, None, None]
    out = xp.permute(0, 2, 3, 1)[b, rows[:, :, None], cols[:, None, :]]  # (B, H, W, C)
    return out.permute(0, 3, 1, 2).contiguous()


# ---------------------------------------------------------------------------
# Modelos
# ---------------------------------------------------------------------------

class LinearClassifier(nn.Module):
    """logits = x W + b, com x = imagem achatada (3*32*32 = 3072 entradas)."""

    def __init__(self, in_features: int = 3 * 32 * 32, n_classes: int = N_CLASSES):
        super().__init__()
        self.fc = nn.Linear(in_features, n_classes)

    def forward(self, x):
        return self.fc(x.flatten(1))


class MLP(nn.Module):
    """Rede totalmente conectada com camadas ocultas configuráveis.

    Cada camada oculta: Linear -> [BatchNorm] -> ReLU -> [Dropout].
    """

    def __init__(self, hidden_sizes, dropout: float = 0.0, batchnorm: bool = False,
                 in_features: int = 3 * 32 * 32, n_classes: int = N_CLASSES):
        super().__init__()
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
    """CNN pequena: 3 blocos convolucionais (32 -> 64 -> 128 filtros) + classificador MLP."""

    def __init__(self, dropout: float = 0.3, n_classes: int = N_CLASSES):
        super().__init__()
        self.features = nn.Sequential(
            *_conv_bn(3, 32), *_conv_bn(32, 32), nn.MaxPool2d(2),     # 32 -> 16
            *_conv_bn(32, 64), *_conv_bn(64, 64), nn.MaxPool2d(2),    # 16 -> 8
            *_conv_bn(64, 128), *_conv_bn(128, 128), nn.MaxPool2d(2),  # 8 -> 4
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


# ---------------------------------------------------------------------------
# Treino e avaliação
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(model, X, y, batch_size: int = 2048, device=None):
    """Retorna (loss média, acurácia) com o modelo em modo de avaliação."""
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
                seed: int = SEED, device=None, verbose: bool = True):
    """Treina `model` e registra loss/acurácia de treino e validação a cada época.

    `data` precisa das chaves X_train, y_train, X_val, y_val (tensores já pré-processados).
    Ao final o modelo fica com os pesos da época de MAIOR acurácia de validação
    (checkpoint do melhor modelo). Retorna o histórico (dict).
    """
    device = device or get_device()
    torch.manual_seed(seed)  # dropout e inicializações tardias
    gen = torch.Generator().manual_seed(seed)  # ordem dos mini-batches e augmentation
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
    n = len(X_tr)
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
            if augment_shift:
                xb = random_shift(xb, augment_shift, gen)
            xb, yb = xb.to(device), y_tr[idx].to(device)

            out = model(xb)
            loss = F.cross_entropy(out, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()

            run_loss += loss.item() * len(idx)
            run_correct += (out.argmax(1) == yb).sum().item()
        if sched:
            sched.step()

        val_loss, val_acc = evaluate(model, data["X_val"], data["y_val"], device=device)
        hist["train_loss"].append(run_loss / n)
        hist["train_acc"].append(run_correct / n)
        hist["val_loss"].append(val_loss)
        hist["val_acc"].append(val_acc)
        hist["epoch_time"].append(time.time() - t0)

        if val_acc > best_acc:
            best_acc, best_epoch = val_acc, epoch
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            print(f"época {epoch + 1:2d}/{epochs} | treino loss {hist['train_loss'][-1]:.4f} "
                  f"acc {hist['train_acc'][-1]:.4f} | val loss {val_loss:.4f} acc {val_acc:.4f} "
                  f"| {hist['epoch_time'][-1]:.1f}s")

    model.load_state_dict(best_state)
    hist["best_epoch"] = best_epoch
    hist["best_val_acc"] = best_acc
    return hist


@torch.no_grad()
def predict(model, X, batch_size: int = 2048, device=None) -> torch.Tensor:
    device = device or next(model.parameters()).device
    model.eval()
    preds = [model(X[i:i + batch_size].to(device)).argmax(1).cpu() for i in range(0, len(X), batch_size)]
    return torch.cat(preds).long()


# ---------------------------------------------------------------------------
# Gráficos
# ---------------------------------------------------------------------------

def plot_history(hist, title: str = "", path=None):
    """Curvas de Loss e Acurácia (treino x validação) ao longo das épocas."""
    import matplotlib.pyplot as plt

    ep = range(1, len(hist["train_loss"]) + 1)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.8))
    a1.plot(ep, hist["train_loss"], label="treino")
    a1.plot(ep, hist["val_loss"], label="validação")
    a1.set(xlabel="época", ylabel="loss (cross-entropy)", title="Loss")
    a2.plot(ep, hist["train_acc"], label="treino")
    a2.plot(ep, hist["val_acc"], label="validação")
    a2.axvline(hist["best_epoch"] + 1, ls=":", c="gray", label="melhor época (val)")
    a2.set(xlabel="época", ylabel="acurácia", title="Acurácia")
    for a in (a1, a2):
        a.grid(alpha=0.3)
        a.legend()
    fig.suptitle(title)
    fig.tight_layout()
    if path:
        fig.savefig(path, dpi=130)
    return fig


# ---------------------------------------------------------------------------
# Conjunto adicional e arquivo de resultados
# ---------------------------------------------------------------------------

def natural_sort_key(name: str):
    """'img_2' < 'img_10' (ordenação numérica, não alfabética)."""
    return [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", str(name))]


def load_image_folder(folder):
    """Lê todas as imagens da pasta em ordem natural do nome -> (nomes, uint8 (N, 3, 32, 32))."""
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
    """Grava uma predição (inteiro 0-9) por linha, na ordem recebida."""
    vals = [int(v) for v in (preds.tolist() if hasattr(preds, "tolist") else preds)]
    if any(v < 0 or v > 9 for v in vals):
        raise ValueError("predições devem ser inteiros entre 0 e 9")
    Path(path).write_text("".join(f"{v}\n" for v in vals))
