"""Gera trabalho_svhn.ipynb (rode: .venv/bin/python tools/build_notebook.py)."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Classificação de Dígitos no SVHN — Classificador Linear, MLP e CNN

Notebook com a implementação e todos os experimentos do trabalho.

**Como rodar**
- **Local:** `jupyter notebook trabalho_svhn.ipynb` com o ambiente `.venv` (roda em CPU; ~1h no total).
- **Colab:** envie `svhn_lib.py` para a pasta do Colab (ícone de pasta à esquerda) antes de rodar, ou rode a célula de setup que pede o upload. Ative GPU em *Ambiente de execução → Alterar tipo*.

Toda a lógica reaproveitável (split, modelos, laço de treino, gráficos, predição) está em `svhn_lib.py`, coberta por testes em `tests/`.

**Protocolo**
- Dados via `torchvision.datasets.SVHN`; se o servidor de Stanford falhar, `svhn_lib` lê o espelho oficial do Hugging Face (`ufldl-stanford/svhn`) em `data_hf/`, verificado byte a byte como idêntico e na mesma ordem.\n- `split="train"` (73.257 imagens) → **90% treino / 10% validação**, divisão fixa com semente 42.
- Validação: acompanhar os experimentos e escolher o modelo (inclusive a melhor época = checkpoint).
- `split="test"` (26.032 imagens): só para relatar o desempenho final de cada experimento — **não** é usado para nenhuma escolha.
- Conjunto adicional do professor: usado **apenas** no final, depois de escolhido o modelo.
""")

code(r"""
import os, sys, json, time
try:
    import svhn_lib
except ModuleNotFoundError:  # Colab sem o arquivo
    from google.colab import files
    files.upload()  # selecione svhn_lib.py
    import svhn_lib

import importlib; importlib.reload(svhn_lib)
import svhn_lib as L
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SEED = 42
L.set_seed(SEED)
device = L.get_device()
os.makedirs('resultados', exist_ok=True)
os.makedirs('pesos', exist_ok=True)
print('device:', device, '| torch', torch.__version__, '| threads', torch.get_num_threads())
""")

md("## 1. Dados")

code(r"""
(X_all, y_all), (X_test_u8, y_test) = L.load_svhn('data')
idx_tr, idx_va = L.split_train_val(len(X_all), val_frac=0.1, seed=SEED)
X_tr_u8, y_train = X_all[idx_tr], y_all[idx_tr]
X_va_u8, y_val = X_all[idx_va], y_all[idx_va]
print('treino   :', tuple(X_tr_u8.shape))
print('validação:', tuple(X_va_u8.shape))
print('teste    :', tuple(X_test_u8.shape))
""")

code(r"""
fig, ax = plt.subplots(2, 10, figsize=(14, 3.2))
for i, a in enumerate(ax.flat):
    a.imshow(X_tr_u8[i].permute(1, 2, 0)); a.set_title(int(y_train[i])); a.axis('off')
plt.suptitle('Exemplos do conjunto de treino'); plt.tight_layout(); plt.savefig('resultados/exemplos.png', dpi=120); plt.show()

dist = pd.DataFrame({s: np.bincount(y.numpy(), minlength=10) for s, y in
                     [('treino', y_train), ('validação', y_val), ('teste', y_test)]})
dist.index.name = 'dígito'; dist
""")

md(r"""
**Pré-processamento.** Duas versões dos dados:
- `bruto`: pixels em [0, 1] (como no notebook da aula, só `ToTensor`).
- `norm`: normalização por canal com média/desvio **calculados só no treino** (sem vazamento da validação/teste).
""")

code(r"""
mean, std = L.compute_norm_stats(X_tr_u8)
print('média por canal:', mean.numpy().round(4), '| desvio:', std.numpy().round(4))
torch.save({'mean': mean, 'std': std}, 'pesos/normalizacao.pt')

DATA = {
    'bruto': dict(X_train=L.preprocess(X_tr_u8), y_train=y_train,
                  X_val=L.preprocess(X_va_u8), y_val=y_val,
                  X_test=L.preprocess(X_test_u8), y_test=y_test),
    'norm':  dict(X_train=L.preprocess(X_tr_u8, mean, std), y_train=y_train,
                  X_val=L.preprocess(X_va_u8, mean, std), y_val=y_val,
                  X_test=L.preprocess(X_test_u8, mean, std), y_test=y_test),
}
""")

md(r"""
## 2. Função que roda um experimento

Para cada experimento: fixa a semente, cria o modelo, treina registrando **loss e acurácia de treino e validação por época**,
guarda os pesos da melhor época na validação, mede a acurácia no teste e salva histórico (`resultados/*.json`), curva (`resultados/*.png`) e pesos (`pesos/*.pt`).

> A acurácia de treino é a média durante a época (com dropout/augmentation ativos); validação e teste são medidos com o modelo em modo `eval`.
""")

code(r"""
RESULTS = []

def run(exp_id, nome, modificacao, make_model, dados='norm', **train_kw):
    print(f'=== E{exp_id}: {nome} ===')
    L.set_seed(SEED)
    model = make_model()
    data = DATA[dados]
    t0 = time.time()
    hist = L.train_model(model, data, seed=SEED, device=device, **train_kw)
    test_loss, test_acc = L.evaluate(model, data['X_test'], data['y_test'], device=device)
    b = hist['best_epoch']
    row = dict(exp=exp_id, nome=nome, modificacao=modificacao, dados=dados,
               params=L.count_params(model), epocas=train_kw['epochs'], melhor_epoca=b + 1,
               train_loss=hist['train_loss'][b], val_loss=hist['val_loss'][b],
               train_acc=hist['train_acc'][b], val_acc=hist['val_acc'][b],
               test_acc=test_acc, tempo_min=(time.time() - t0) / 60,
               config={k: v for k, v in train_kw.items()})
    RESULTS.append(row)
    torch.save({'state_dict': model.state_dict(), 'row': row}, f'pesos/E{exp_id}.pt')
    with open(f'resultados/E{exp_id}.json', 'w') as f:
        json.dump({'row': row, 'hist': hist}, f, indent=1)
    L.plot_history(hist, f'E{exp_id} — {nome}', f'resultados/E{exp_id}_curvas.png'); plt.show()
    print(f"melhor época {b+1}: val_acc={row['val_acc']:.4f} | test_acc={test_acc:.4f} | {row['tempo_min']:.1f} min")
    return model, hist
""")

md(r"""
## 3. Experimentos

Cada experimento muda **uma coisa** em relação ao anterior (exceto E7 e E8, explicados abaixo), para dar para atribuir a melhora/piora.
""")

md(r"""
### E1 — Classificador linear
`logits = x·W + b` sobre os 3072 pixels normalizados, cross-entropy + regularização L2 (weight decay), SGD com momentum — mesma ideia do classificador linear da aula, agora com `nn.Linear`.
""")
code(r"""
run(1, 'Classificador linear', 'Linear 3072→10, SGD+L2',
    lambda: L.LinearClassifier(), dados='norm',
    epochs=20, batch_size=128, lr=1e-3, optimizer='sgd', weight_decay=1e-4);
""")

md(r"""
### E2 — MLP baseline (código da aula)
Exatamente a MLP do notebook da aula: 2 camadas ocultas de 500 neurônios, ReLU, Adam lr=1e-3, batch 100, entrada só em [0, 1].
""")
code(r"""
run(2, 'MLP baseline', 'MLP 2×500, entrada [0,1]',
    lambda: L.MLP([500, 500]), dados='bruto',
    epochs=25, batch_size=100, lr=1e-3);
""")

md("### E3 — + normalização por canal")
code(r"""
run(3, 'MLP + normalização', 'E2 + normalização média/desvio',
    lambda: L.MLP([500, 500]), dados='norm',
    epochs=25, batch_size=100, lr=1e-3);
""")

md("### E4 — + mais camadas e neurônios (1024 → 512 → 256)")
code(r"""
run(4, 'MLP maior', 'E3 com 3 camadas 1024-512-256',
    lambda: L.MLP([1024, 512, 256]), dados='norm',
    epochs=25, batch_size=100, lr=1e-3);
""")

md("### E5 — + Dropout 0.3")
code(r"""
run(5, 'MLP + dropout', 'E4 + dropout 0.3',
    lambda: L.MLP([1024, 512, 256], dropout=0.3), dados='norm',
    epochs=25, batch_size=100, lr=1e-3);
""")

md("### E6 — + Batch Normalization")
code(r"""
run(6, 'MLP + dropout + BN', 'E5 + BatchNorm',
    lambda: L.MLP([1024, 512, 256], dropout=0.3, batchnorm=True), dados='norm',
    epochs=25, batch_size=100, lr=1e-3);
""")

md(r"""
### E7 — Otimização: batch 256 + taxa de aprendizado com decaimento cosseno
Batch maior dá passos menos ruidosos mas em menor número; o decaimento cosseno da lr compensa, permitindo passos grandes no começo e ajuste fino no fim.
""")
code(r"""
run(7, 'MLP batch 256 + cosine', 'E6 com batch 256 e lr cosseno',
    lambda: L.MLP([1024, 512, 256], dropout=0.3, batchnorm=True), dados='norm',
    epochs=25, batch_size=256, lr=1e-3, scheduler='cosine');
""")

md(r"""
### E8 — CNN (rede convolucional) + data augmentation
O enunciado deixa a arquitetura livre e cita "número de filtros". A MLP trata cada pixel como uma entrada independente;
a convolução reaproveita os mesmos filtros em toda a imagem, o que é muito mais adequado a imagens.
3 blocos conv (32→64→128 filtros, BN, ReLU, max-pool) + classificador com dropout 0.3.
Augmentation: translação aleatória de até 2 px (sem flip, que altera dígitos). AdamW (weight decay 5e-4) + lr cosseno.
""")
code(r"""
run(8, 'CNN + augmentation', 'CNN 3 blocos + translação ±2px',
    lambda: L.CNN(dropout=0.3), dados='norm',
    epochs=20, batch_size=128, lr=1e-3, optimizer='adamw', weight_decay=5e-4,
    scheduler='cosine', augment_shift=2);
""")

md("## 4. Comparação dos experimentos")
code(r"""
df = pd.DataFrame(RESULTS).drop(columns=['config'])
df['gap_treino_val'] = df['train_acc'] - df['val_acc']
df.to_csv('resultados/tabela_experimentos.csv', index=False)
fmt = {c: '{:.4f}' for c in ['train_loss', 'val_loss', 'train_acc', 'val_acc', 'test_acc', 'gap_treino_val']}
df.style.format(fmt | {'tempo_min': '{:.1f}', 'params': '{:,}'})
""")

code(r"""
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
for r in RESULTS:
    h = json.load(open(f"resultados/E{r['exp']}.json"))['hist']
    ax[0].plot(range(1, len(h['val_acc']) + 1), h['val_acc'], label=f"E{r['exp']} {r['nome']}")
    ax[1].plot(range(1, len(h['val_loss']) + 1), h['val_loss'], label=f"E{r['exp']}")
ax[0].set(title='Acurácia de validação por época', xlabel='época', ylabel='acurácia')
ax[1].set(title='Loss de validação por época', xlabel='época', ylabel='loss', ylim=(0, 2.5))
for a in ax: a.grid(alpha=.3)
ax[0].legend(fontsize=8); plt.tight_layout(); plt.savefig('resultados/comparacao_validacao.png', dpi=130); plt.show()
""")

md(r"""
## 5. Escolha do modelo final
Critério: **maior acurácia de validação** (o teste não participa da escolha).
""")
code(r"""
best = max(RESULTS, key=lambda r: r['val_acc'])
print(f"Modelo escolhido: E{best['exp']} — {best['nome']}")
print(f"val_acc = {best['val_acc']:.4f} | test_acc = {best['test_acc']:.4f} | época {best['melhor_epoca']}")

ckpt = torch.load(f"pesos/E{best['exp']}.pt")
torch.save({'exp': best['exp'], 'nome': best['nome'], 'state_dict': ckpt['state_dict'],
            'mean': mean, 'std': std, 'row': best}, 'melhor_modelo.pt')
with open('resultados/melhor_modelo.json', 'w') as f:
    json.dump(best, f, indent=1)
print('pesos salvos em melhor_modelo.pt')
""")

md(r"""
## 6. Predição do conjunto adicional
Feita **somente agora**, com o modelo já escolhido e congelado. Pré-processamento idêntico ao do treino (mesma média/desvio).
As imagens são lidas em **ordem natural do nome do arquivo** (`2.png` antes de `10.png`), e o arquivo sai com um inteiro (0–9) por linha.

Coloque as imagens do professor em `extra/` (descompacte o zip ali) e ajuste `NOMES` abaixo.
""")
code(r"""
NOMES = 'NOME1_NOME2'          # <- troque pelos nomes do grupo
PASTA_EXTRA = 'extra'

MODELOS = {1: lambda: L.LinearClassifier(), 2: lambda: L.MLP([500, 500]), 3: lambda: L.MLP([500, 500]),
           4: lambda: L.MLP([1024, 512, 256]), 5: lambda: L.MLP([1024, 512, 256], dropout=0.3),
           6: lambda: L.MLP([1024, 512, 256], dropout=0.3, batchnorm=True),
           7: lambda: L.MLP([1024, 512, 256], dropout=0.3, batchnorm=True), 8: lambda: L.CNN(dropout=0.3)}

final = torch.load('melhor_modelo.pt')
model = MODELOS[final['exp']]().to(device)
model.load_state_dict(final['state_dict'])
usa_norm = final['row']['dados'] == 'norm'

if os.path.isdir(PASTA_EXTRA):
    nomes_arq, X_extra_u8 = L.load_image_folder(PASTA_EXTRA)
    X_extra = L.preprocess(X_extra_u8, final['mean'], final['std']) if usa_norm else L.preprocess(X_extra_u8)
    preds = L.predict(model, X_extra, device=device)
    saida = f'{NOMES}_RESULTS.txt'
    L.write_results(preds, saida)
    print(f'{len(preds)} imagens -> {saida}')
    print('primeiros arquivos:', nomes_arq[:5], '... último:', nomes_arq[-1])
    print('distribuição das predições:', np.bincount(preds.numpy(), minlength=10))
else:
    print(f'Pasta {PASTA_EXTRA}/ não encontrada — baixe o conjunto adicional e rode esta célula de novo.')
""")

nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, "trabalho_svhn.ipynb")
print("ok:", len(C), "células")
