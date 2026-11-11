import os

import numpy as np
import pytest
import torch
from PIL import Image

import svhn_lib as L


# ---------- divisão treino / validação ----------

def test_split_e_deterministico_disjunto_e_completo():
    a_tr, a_va = L.split_train_val(1000, val_frac=0.1, seed=42)
    b_tr, b_va = L.split_train_val(1000, val_frac=0.1, seed=42)
    assert torch.equal(a_tr, b_tr) and torch.equal(a_va, b_va)
    assert len(a_va) == 100 and len(a_tr) == 900
    assert len(set(a_tr.tolist()) & set(a_va.tolist())) == 0
    assert sorted(a_tr.tolist() + a_va.tolist()) == list(range(1000))


def test_split_muda_com_outra_semente():
    _, va1 = L.split_train_val(1000, 0.1, seed=1)
    _, va2 = L.split_train_val(1000, 0.1, seed=2)
    assert not torch.equal(va1, va2)


# ---------- pré-processamento ----------

def test_normalizacao_por_canal():
    g = torch.Generator().manual_seed(0)
    X = torch.randint(0, 256, (50, 3, 32, 32), dtype=torch.uint8, generator=g)
    mean, std = L.compute_norm_stats(X)
    assert mean.shape == (3,) and std.shape == (3,)
    Z = L.preprocess(X, mean, std)
    assert Z.dtype == torch.float32
    assert torch.allclose(Z.mean(dim=(0, 2, 3)), torch.zeros(3), atol=1e-4)
    assert torch.allclose(Z.std(dim=(0, 2, 3)), torch.ones(3), atol=1e-2)


def test_preprocess_sem_stats_vai_para_0_1():
    X = torch.full((2, 3, 32, 32), 255, dtype=torch.uint8)
    assert torch.allclose(L.preprocess(X), torch.ones(2, 3, 32, 32))


def test_random_shift_mantem_formato_e_e_deterministico():
    x = torch.randn(8, 3, 32, 32)
    a = L.random_shift(x, 2, torch.Generator().manual_seed(0))
    b = L.random_shift(x, 2, torch.Generator().manual_seed(0))
    assert a.shape == x.shape and torch.equal(a, b)
    assert torch.equal(L.random_shift(x, 0, torch.Generator()), x)


# ---------- modelos ----------

@pytest.mark.parametrize("model", [
    L.LinearClassifier(),
    L.MLP([64, 32]),
    L.MLP([64], dropout=0.3, batchnorm=True),
    L.CNN(),
])
def test_modelos_produzem_10_logits(model):
    model.eval()
    out = model(torch.randn(4, 3, 32, 32))
    assert out.shape == (4, 10)


# ---------- treino ----------

def _dados_sinteticos(n=64):
    g = torch.Generator().manual_seed(0)
    X = torch.randn(n, 3, 32, 32, generator=g)
    y = torch.randint(0, 10, (n,), generator=g)
    return {"X_train": X, "y_train": y, "X_val": X[:16], "y_val": y[:16]}


def test_treino_registra_historico_por_epoca():
    data = _dados_sinteticos()
    L.set_seed(0)
    hist = L.train_model(L.MLP([32]), data, epochs=3, batch_size=16, lr=1e-3, verbose=False)
    for k in ("train_loss", "train_acc", "val_loss", "val_acc"):
        assert len(hist[k]) == 3
    assert 0 <= hist["best_epoch"] < 3
    assert hist["best_val_acc"] == max(hist["val_acc"])


def test_treino_e_reprodutivel_com_mesma_semente():
    data = _dados_sinteticos()
    L.set_seed(0)
    h1 = L.train_model(L.MLP([32], dropout=0.2), data, epochs=2, batch_size=16, lr=1e-3, seed=7, verbose=False)
    L.set_seed(0)
    h2 = L.train_model(L.MLP([32], dropout=0.2), data, epochs=2, batch_size=16, lr=1e-3, seed=7, verbose=False)
    assert h1["train_loss"] == h2["train_loss"]
    assert h1["val_acc"] == h2["val_acc"]


def test_treino_aprende_problema_facil():
    # classe c (0..2) soma +1 no canal c -> separável linearmente
    g = torch.Generator().manual_seed(0)
    y = torch.randint(0, 3, (600,), generator=g)
    X = torch.randn(600, 3, 32, 32, generator=g)
    X[torch.arange(600), y] += 1.0
    data = {"X_train": X[:500], "y_train": y[:500], "X_val": X[500:], "y_val": y[500:]}
    L.set_seed(0)
    hist = L.train_model(L.LinearClassifier(), data, epochs=15, batch_size=50, lr=1e-2, verbose=False)
    assert hist["best_val_acc"] > 0.8


# ---------- conjunto adicional e arquivo de resultados ----------

def test_ordenacao_natural_dos_nomes():
    nomes = ["img_10.png", "img_2.png", "img_1.png", "img_100.png"]
    assert sorted(nomes, key=L.natural_sort_key) == ["img_1.png", "img_2.png", "img_10.png", "img_100.png"]


def test_carrega_imagens_em_ordem_natural(tmp_path):
    for i in [10, 2, 1]:
        Image.fromarray(np.full((32, 32, 3), i, dtype=np.uint8)).save(tmp_path / f"{i}.png")
    (tmp_path / "leia-me.txt").write_text("x")
    nomes, X = L.load_image_folder(tmp_path)
    assert nomes == ["1.png", "2.png", "10.png"]
    assert X.shape == (3, 3, 32, 32) and X.dtype == torch.uint8
    assert X[:, 0, 0, 0].tolist() == [1, 2, 10]


def test_carrega_imagem_de_outro_tamanho_redimensiona(tmp_path):
    Image.fromarray(np.zeros((40, 50), dtype=np.uint8)).save(tmp_path / "a.png")
    _, X = L.load_image_folder(tmp_path)
    assert X.shape == (1, 3, 32, 32)


def test_write_results_um_inteiro_por_linha(tmp_path):
    p = tmp_path / "A_B_RESULTS.txt"
    L.write_results(torch.tensor([7, 2, 1, 0, 9]), p)
    assert p.read_text() == "7\n2\n1\n0\n9\n"


def test_write_results_rejeita_fora_de_0_9(tmp_path):
    with pytest.raises(ValueError):
        L.write_results(torch.tensor([3, 10]), tmp_path / "x.txt")


def test_predict_retorna_inteiros(tmp_path):
    model = L.MLP([16])
    preds = L.predict(model, torch.randn(5, 3, 32, 32), batch_size=2)
    assert preds.shape == (5,) and preds.dtype == torch.int64
    assert int(preds.min()) >= 0 and int(preds.max()) <= 9
