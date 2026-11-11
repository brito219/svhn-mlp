"""Verificação independente dos entregáveis (rode: .venv/bin/python tools/verificar.py).

1. Re-treina E5 e E11 do zero e compara o histórico com resultados/*.json (reprodutibilidade bit a bit).
2. Recarrega melhor_modelo.pt, confere a acurácia de teste e refaz as predições do conjunto adicional.
3. Gera o arquivo reserva com a melhor MLP (reserva_mlp/), caso o modelo final precise ser uma MLP.
"""
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import svhn_lib as L  # noqa: E402

SEED = 42
NOMES = "JOAOPEDRO_SAMYR_VALBER"
ok = True


def check(nome, cond, detalhe=""):
    global ok
    ok &= bool(cond)
    print(f"[{'OK' if cond else 'FALHOU'}] {nome} {detalhe}", flush=True)


L.set_seed(SEED)
(X_all, y_all), (X_te, y_te) = L.load_svhn(ROOT / "data")
itr, iva = L.split_train_val(len(X_all), 0.1, SEED)
mean, std = L.compute_norm_stats(X_all[itr])
D = dict(X_train=L.preprocess(X_all[itr], mean, std), y_train=y_all[itr],
         X_val=L.preprocess(X_all[iva], mean, std), y_val=y_all[iva])
Xt = L.preprocess(X_te, mean, std)

# 1. reprodutibilidade
MODELOS = {5: lambda: L.MLP([500, 500]), 11: lambda: L.MLP([1024, 512, 256], dropout=0.3, batchnorm=True)}
for e, mk in MODELOS.items():
    ref = json.load(open(ROOT / f"resultados/E{e}.json"))
    L.set_seed(SEED)
    m = mk()
    h = L.train_model(m, D, seed=SEED, verbose=False, **ref["row"]["config"])
    _, ta = L.evaluate(m, Xt, y_te)
    igual = all(h[k] == ref["hist"][k] for k in ("train_loss", "train_acc", "val_loss", "val_acc"))
    check(f"E{e} re-treinado: histórico idêntico bit a bit em {len(h['val_acc'])} épocas", igual,
          f"| val {h['best_val_acc']:.4f} (salvo {ref['row']['val_acc']:.4f}) | teste {ta:.4f} (salvo {ref['row']['test_acc']:.4f})")
    check(f"E{e}: acurácia de teste idêntica", ta == ref["row"]["test_acc"])

# 2. modelo final
ck = torch.load(ROOT / "melhor_modelo.pt")
fin = json.load(open(ROOT / "resultados/final.json"))
final = L.CNN(dropout=0.3)
final.load_state_dict(ck["state_dict"])
_, ta = L.evaluate(final, L.preprocess(X_te, ck["mean"], ck["std"]), y_te)
check("melhor_modelo.pt recarregado: acurácia de teste igual à do relatório", ta == fin["test_acc"], f"| {ta:.4f}")
mf, sf = L.compute_norm_stats(X_all)
check("normalização salva = estatísticas de todo o treino", torch.equal(mf, ck["mean"]) and torch.equal(sf, ck["std"]))

nomes, X_extra = L.load_image_folder(ROOT / "extra")
preds = L.predict(final, L.preprocess(X_extra, ck["mean"], ck["std"]))
arq = [int(l) for l in open(ROOT / f"{NOMES}_RESULTS.txt")]
check("arquivo de predições: nº de linhas = nº de imagens", len(arq) == len(nomes), f"| {len(arq)}")
check("arquivo de predições idêntico ao refeito com melhor_modelo.pt", preds.tolist() == arq)
check("imagens em ordem crescente do nome", nomes == sorted(nomes) and nomes[0] == "000000.png")

# 3. reserva: melhor MLP (E11)
ck11 = torch.load(ROOT / "pesos/E11.pt")
mlp = MODELOS[11]()
mlp.load_state_dict(ck11["state_dict"])
_, ta = L.evaluate(mlp, Xt, y_te)
check("pesos da melhor MLP (E11) conferem com o teste salvo", ta == ck11["row"]["test_acc"], f"| {ta:.4f}")
p_mlp = L.predict(mlp, L.preprocess(X_extra, mean, std))
(ROOT / "reserva_mlp").mkdir(exist_ok=True)
L.write_results(p_mlp, ROOT / "reserva_mlp" / f"{NOMES}_RESULTS.txt")
torch.save({"exp": 11, "state_dict": ck11["state_dict"], "mean": mean, "std": std, "row": ck11["row"]},
           ROOT / "reserva_mlp" / "melhor_mlp.pt")
print(f"reserva_mlp/ gerada | concordância MLP x modelo final no conjunto adicional: "
      f"{(p_mlp == preds).float().mean().item():.4f}")
print("RESULTADO:", "TUDO OK" if ok else "HÁ FALHAS")
