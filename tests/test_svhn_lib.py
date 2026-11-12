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


# ---------- leitura do espelho parquet (Hugging Face) ----------

def test_le_split_do_parquet(tmp_path):
    import io
    import pandas as pd
    rows = []
    for lbl in [3, 0, 9]:
        buf = io.BytesIO()
        Image.fromarray(np.full((32, 32, 3), lbl * 10, dtype=np.uint8)).save(buf, format="PNG")
        rows.append({"image": {"bytes": buf.getvalue(), "path": None}, "label": lbl})
    pd.DataFrame(rows).to_parquet(tmp_path / "train.parquet")
    X, y = L.load_parquet_split(tmp_path / "train.parquet")
    assert X.shape == (3, 3, 32, 32) and X.dtype == torch.uint8
    assert y.tolist() == [3, 0, 9] and y.dtype == torch.int64
    assert X[:, 0, 0, 0].tolist() == [30, 0, 90]


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


# ---------- retreino final (treino + validação, sem conjunto de validação) ----------

def test_treino_sem_validacao_fica_com_pesos_da_ultima_epoca():
    data = _dados_sinteticos()
    data = {"X_train": data["X_train"], "y_train": data["y_train"]}
    L.set_seed(0)
    model = L.MLP([32])
    hist = L.train_model(model, data, epochs=3, batch_size=16, lr=1e-3, verbose=False)
    assert len(hist["train_loss"]) == 3 and len(hist["train_acc"]) == 3
    assert hist["val_loss"] == [] and hist["val_acc"] == []
    assert hist["best_epoch"] == 2 and hist["best_val_acc"] is None


def test_plot_history_funciona_sem_validacao(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    hist = {"train_loss": [1.0, 0.5], "train_acc": [0.5, 0.8], "val_loss": [], "val_acc": [],
            "best_epoch": 1, "best_val_acc": None}
    L.plot_history(hist, "x", tmp_path / "f.png")
    assert (tmp_path / "f.png").exists()


def test_matriz_de_confusao():
    y = torch.tensor([0, 0, 1, 2, 2, 2])
    p = torch.tensor([0, 1, 1, 2, 2, 0])
    cm = L.confusion_matrix(y, p, n_classes=3)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [1, 0, 2]]


# ---------- estatísticas de normalização exatas (reprodutibilidade) ----------

def test_norm_stats_sao_exatas_e_independem_do_layout_de_memoria():
    g = torch.Generator().manual_seed(1)
    X = torch.randint(0, 256, (4000, 3, 32, 32), dtype=torch.uint8, generator=g)
    mean, std = L.compute_norm_stats(X)
    # referência em float64 (NumPy)
    Xd = X.numpy().astype(np.float64) / 255.0
    ref_mean = torch.tensor(Xd.mean(axis=(0, 2, 3)), dtype=torch.float32)
    ref_std = torch.tensor(Xd.std(axis=(0, 2, 3), ddof=1), dtype=torch.float32)
    assert torch.equal(mean, ref_mean) and torch.equal(std, ref_std)
    # mesmo conteúdo, tensor não contíguo (como o que vem do torchvision): resultado idêntico bit a bit
    X_nc = X.permute(0, 2, 3, 1).contiguous().permute(0, 3, 1, 2)
    assert not X_nc.is_contiguous()
    mean2, std2 = L.compute_norm_stats(X_nc)
    assert torch.equal(mean, mean2) and torch.equal(std, std2)
    assert mean.dtype == torch.float32 and std.dtype == torch.float32


def test_per_image_norm_zera_media_e_ignora_brilho_e_contraste():
    x = torch.rand(4, 3, 32, 32)
    a = L.per_image_norm(x)
    assert torch.allclose(a.mean(dim=(1, 2, 3)), torch.zeros(4), atol=1e-5)
    assert torch.allclose(L.per_image_norm(x * 0.5 + 0.2), a, atol=2e-2)


def test_random_affine_formato_determinismo_e_identidade():
    x = torch.rand(5, 3, 32, 32)
    a = L.random_affine(x, 2, 10.0, 0.1, torch.Generator().manual_seed(0))
    b = L.random_affine(x, 2, 10.0, 0.1, torch.Generator().manual_seed(0))
    assert a.shape == x.shape and torch.equal(a, b) and not torch.equal(a, x)
    assert torch.allclose(L.random_affine(x, 0, 0.0, 0.0, torch.Generator()), x, atol=1e-5)


def test_shift_fixed_desloca_o_conteudo():
    x = torch.arange(16.0).view(1, 1, 4, 4)
    assert torch.equal(L.shift_fixed(x, 0, 0), x)
    assert torch.equal(L.shift_fixed(x, 1, 0)[0, 0, :, :3], x[0, 0, :, 1:])
    assert torch.equal(L.shift_fixed(x, 0, -1)[0, 0, 1:], x[0, 0, :3])


def test_predict_tta_raio_zero_igual_a_predict_e_probabilidades_somam_1():
    torch.manual_seed(0)
    m = L.MLP([16])
    x = torch.randn(7, 3, 32, 32)
    assert torch.equal(L.predict_tta(m, x, 0), L.predict(m, x))
    p = L.predict_proba(m, x, 2)
    assert p.shape == (7, 10) and torch.allclose(p.sum(1), torch.ones(7), atol=1e-5)


def test_treino_com_onecycle_e_affine_e_reprodutivel():
    def go():
        L.set_seed(0)
        X = torch.randn(64, 3, 32, 32)
        d = dict(X_train=X, y_train=torch.randint(0, 10, (64,)), X_val=X[:16], y_val=torch.randint(0, 10, (16,)))
        m = L.MLP([32], dropout=0.1, batchnorm=True)
        return L.train_model(m, d, epochs=2, batch_size=16, lr=1e-3, optimizer="adamw", scheduler="onecycle",
                             augment_shift=2, augment_rot=10.0, augment_scale=0.1, seed=0, verbose=False)
    a, b = go(), go()
    assert a["train_loss"] == b["train_loss"] and a["val_acc"] == b["val_acc"]


def test_scheduler_desconhecido_da_erro():
    X = torch.randn(8, 3, 32, 32)
    with pytest.raises(ValueError):
        L.train_model(L.MLP([8]), dict(X_train=X, y_train=torch.zeros(8, dtype=torch.long)),
                      epochs=1, batch_size=8, lr=1e-3, scheduler="xyz", verbose=False)
