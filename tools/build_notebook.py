"""Gera trabalho_svhn.ipynb (rode: .venv/bin/python tools/build_notebook.py).

O código de svhn_lib.py é embutido na primeira célula (%%writefile), para o notebook ser
autossuficiente (basta ele para rodar no Colab) e mostrar toda a implementação.
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parent.parent
nb = nbf.v4.new_notebook()
C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s.strip()))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))

md(r"""
# Classificação de Dígitos no SVHN — Classificador Linear, MLP e CNN

**Grupo:** João Pedro, Samyr e Valber

Notebook com a implementação e todos os experimentos do trabalho. Ele é autossuficiente: a primeira célula grava
`svhn_lib.py` (modelos, laço de treino, avaliação, gráficos e predição) e as demais executam os experimentos.

**Como rodar**
- **Local:** `jupyter notebook trabalho_svhn.ipynb` (roda em CPU; cerca de 9 a 10 h no total).
- **Colab:** basta abrir este notebook e executar tudo (ative a GPU em *Ambiente de execução → Alterar tipo*).

**Protocolo**
- Dados: `torchvision.datasets.SVHN`, `split="train"` (73.257 imagens) e `split="test"` (26.032).
- O `train` é dividido em **90% treino / 10% validação**, com divisão fixa pela semente 42.
- A validação acompanha os experimentos e escolhe o modelo (e a melhor época de cada treino, por checkpoint).
- O `test` só é usado para relatar o desempenho de cada experimento; **não** participa de nenhuma escolha.
- O **modelo final é uma rede totalmente conectada (MLP)**, como definido pelo professor. A CNN (E12 e E13) entra
  apenas como comparação e não concorre à escolha do modelo final.
- Depois de escolhida a configuração, o modelo final é retreinado com todo o `train` (treino + validação), como o enunciado permite.
- O conjunto adicional do professor é usado **apenas** na última seção, com o modelo final já definido.
- Semente fixa (42) em tudo: rodar de novo reproduz os mesmos números na mesma máquina.
""")

md(r"""
## 0. Implementação (`svhn_lib.py`)

Todo o código reaproveitável fica nesta célula, que grava o arquivo `svhn_lib.py`. Partimos dos notebooks da aula
(classificador linear e MLP) e adaptamos ao SVHN: entrada de 3×32×32 = 3072 valores, divisão treino/validação e
registro de loss e acurácia por época. O arquivo também é coberto por testes automatizados (`tests/`).
""")
code("%%writefile svhn_lib.py\n" + (ROOT / "svhn_lib.py").read_text())

code(r"""
import os, sys, json, time, importlib
import svhn_lib
importlib.reload(svhn_lib)
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
**Pré-processamento.** Três versões dos dados:
- `bruto`: pixels em [0, 1] (como no notebook da aula, só `ToTensor`).
- `norm`: subtrai a média e divide pelo desvio padrão **de cada canal de cor**, calculados só no treino
  (slide de pré-processamento: "subtraia a média por canal"). Validação e teste usam os mesmos valores, sem vazamento.
- `norm_img`: a versão `norm` seguida de uma normalização **por imagem** (subtrai a média e divide pelo desvio da
  própria imagem). Usada só a partir do E16; remove diferenças de brilho e contraste entre as fotos.
""")

code(r"""
mean, std = L.compute_norm_stats(X_tr_u8)
print('média por canal:', mean.numpy().round(4), '| desvio:', std.numpy().round(4))

DATA = {
    'bruto': dict(X_train=L.preprocess(X_tr_u8), y_train=y_train,
                  X_val=L.preprocess(X_va_u8), y_val=y_val,
                  X_test=L.preprocess(X_test_u8), y_test=y_test),
    'norm':  dict(X_train=L.preprocess(X_tr_u8, mean, std), y_train=y_train,
                  X_val=L.preprocess(X_va_u8, mean, std), y_val=y_val,
                  X_test=L.preprocess(X_test_u8, mean, std), y_test=y_test),
}
DATA['norm_img'] = {k: (L.per_image_norm(v) if k.startswith('X_') else v) for k, v in DATA['norm'].items()}
""")

md(r"""
**Checagem de sanidade (da aula de loss).** Com pesos iniciais aleatórios, as 10 classes são quase equiprováveis,
então a cross-entropy inicial deve ficar perto de $\ln 10 \approx 2{,}303$ e a acurácia perto de 10%.
""")
code(r"""
L.set_seed(SEED)
for nome, m in [('Linear', L.LinearClassifier()), ('MLP 2×500', L.MLP([500, 500])), ('CNN', L.CNN())]:
    loss0, acc0 = L.evaluate(m, DATA['norm']['X_val'], y_val)
    print(f'{nome:10s} loss inicial = {loss0:.3f} (ln 10 = {np.log(10):.3f}) | acurácia inicial = {acc0:.3f}')
""")

md(r"""
## 2. Função que roda um experimento

Para cada experimento: fixa a semente, cria o modelo, treina registrando **loss e acurácia de treino e de validação
a cada época**, fica com os pesos da melhor época na validação, mede a acurácia no teste e salva o histórico
(`resultados/*.json`), as curvas (`resultados/*.png`) e os pesos (`pesos/*.pt`).

> A acurácia de treino é a média ao longo da época, com dropout e augmentation ativos. Validação e teste são medidos
> com o modelo em modo `eval`.
""")

code(r"""
RESULTS, REGISTRY = [], {}

def run(exp_id, nome, modificacao, make_model, dados='norm', pai=None, **train_kw):
    print(f'=== E{exp_id}: {nome} ===')
    REGISTRY[exp_id] = (make_model, dados, train_kw)
    L.set_seed(SEED)
    model = make_model()
    data = DATA[dados]
    t0 = time.time()
    hist = L.train_model(model, data, seed=SEED, device=device, **train_kw)
    test_loss, test_acc = L.evaluate(model, data['X_test'], data['y_test'], device=device)
    b = hist['best_epoch']
    row = dict(exp=exp_id, nome=nome, modificacao=modificacao, dados=dados, pai=pai,
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
    print(f"melhor época {b+1}: train_loss={row['train_loss']:.4f} val_loss={row['val_loss']:.4f} | "
          f"train_acc={row['train_acc']:.4f} val_acc={row['val_acc']:.4f} | test_acc={test_acc:.4f} | {row['tempo_min']:.1f} min")
    return model, hist
""")

md(r"""
## 3. Experimentos

Cada experimento muda **um fator** em relação a um experimento anterior (o "pai", indicado em cada um), para que a
melhora ou a piora possa ser atribuída a esse fator. Quando duas ou mais alternativas são comparadas (taxa de
aprendizado, tamanho do batch), o código segue automaticamente com a de **maior acurácia de validação**.
""")

md(r"""
### E1 — Classificador linear
`logits = x·W + b` sobre os 3072 pixels normalizados, com cross-entropy (softmax) e regularização L2
(`weight_decay`), treinado por gradiente descendente em mini-batches com momentum. É o classificador linear da aula,
agora com `nn.Linear`.
""")
code(r"""
run(1, 'Classificador linear', 'Linear 3072→10, SGD + L2',
    lambda: L.LinearClassifier(), dados='norm',
    epochs=20, batch_size=128, lr=1e-4, optimizer='sgd', weight_decay=1e-4);
""")

md(r"""
### E2 — MLP baseline (código da aula)
A MLP do notebook da aula, sem alterações: 2 camadas ocultas de 500 neurônios, ReLU, Adam com lr = 1e-3, batch 100
e entrada apenas em [0, 1].
""")
code(r"""
run(2, 'MLP baseline', 'MLP 2×500, entrada em [0,1]',
    lambda: L.MLP([500, 500]), dados='bruto', pai=1,
    epochs=25, batch_size=100, lr=1e-3);
""")

md("### E3 — E2 + normalização por canal")
code(r"""
run(3, 'MLP + normalização', 'E2 + normalização (média/desvio por canal)',
    lambda: L.MLP([500, 500]), dados='norm', pai=2,
    epochs=25, batch_size=100, lr=1e-3);
""")

md(r"""
### E4 e E5 — Taxa de aprendizado (10× maior e 10× menor que a do E3)
Mesmo modelo do E3, mudando só a taxa de aprendizado: 1e-2 (E4) e 1e-4 (E5).
""")
code(r"""
run(4, 'MLP lr alta (1e-2)', 'E3 com lr = 1e-2',
    lambda: L.MLP([500, 500]), dados='norm', pai=3,
    epochs=25, batch_size=100, lr=1e-2);
""")
code(r"""
run(5, 'MLP lr baixa (1e-4)', 'E3 com lr = 1e-4',
    lambda: L.MLP([500, 500]), dados='norm', pai=3,
    epochs=25, batch_size=100, lr=1e-4);
""")
code(r"""
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8))
for e, lr in [(4, '1e-2'), (3, '1e-3'), (5, '1e-4')]:
    h = json.load(open(f'resultados/E{e}.json'))['hist']
    ep = range(1, len(h['val_acc']) + 1)
    ax[0].plot(ep, h['train_loss'], label=f'lr = {lr} (E{e})')
    ax[1].plot(ep, h['val_loss'], label=f'lr = {lr} (E{e})')
    ax[2].plot(ep, h['val_acc'], label=f'lr = {lr} (E{e})')
ax[0].set(title='Loss de treino', xlabel='época'); ax[1].set(title='Loss de validação', xlabel='época')
ax[2].set(title='Acurácia de validação', xlabel='época')
for a in ax: a.grid(alpha=.3); a.legend()
plt.suptitle('Efeito da taxa de aprendizado (MLP 2×500 normalizada)'); plt.tight_layout()
plt.savefig('resultados/comparacao_lr.png', dpi=130); plt.show()
""")

md(r"""
**Qual taxa de aprendizado seguir usando?** Os próximos experimentos partem do melhor dos três (E3, E4, E5)
segundo a acurácia de validação. A escolha é feita pelo código, não à mão.
""")
code(r"""
def melhor(*exps):
    "Experimento com a maior acurácia de validação entre os indicados."
    return max((r for r in RESULTS if r['exp'] in exps), key=lambda r: r['val_acc'])

B_LR = melhor(3, 4, 5)
LR = B_LR['config']['lr']
print(f"Melhor taxa na validação: lr = {LR:g} (E{B_LR['exp']}, val_acc = {B_LR['val_acc']:.4f}).")
print(f"Os próximos experimentos partem do E{B_LR['exp']}.")
""")

md("### E6 — mais camadas e neurônios (1024 → 512 → 256)")
code(r"""
run(6, 'MLP maior', f"E{B_LR['exp']} com 3 camadas ocultas: 1024-512-256",
    lambda: L.MLP([1024, 512, 256]), dados='norm', pai=B_LR['exp'],
    epochs=25, batch_size=100, lr=LR);
""")

md(r"""
### E7 — E6 + Dropout 0,3
Regularização vista em aula: a cada passo de treino, 30% dos neurônios de cada camada oculta são zerados.
""")
code(r"""
run(7, 'MLP + dropout', 'E6 + dropout 0,3',
    lambda: L.MLP([1024, 512, 256], dropout=0.3), dados='norm', pai=6,
    epochs=25, batch_size=100, lr=LR);
""")

md(r"""
### E8 — E7 + Batch Normalization
Normaliza as ativações de cada camada oculta dentro do mini-batch (a mesma ideia da normalização da entrada,
aplicada dentro da rede).
""")
code(r"""
run(8, 'MLP + dropout + BN', 'E7 + BatchNorm',
    lambda: L.MLP([1024, 512, 256], dropout=0.3, batchnorm=True), dados='norm', pai=7,
    epochs=25, batch_size=100, lr=LR);
""")

md(r"""
### E9 — E8 com a outra taxa de aprendizado
A BatchNorm muda a escala das ativações e costuma permitir taxas maiores, então a taxa escolhida antes (sem BN)
pode não ser mais a melhor. Repetimos o E8 trocando só a taxa (1e-3 ↔ 1e-4) e seguimos com a melhor das duas.
""")
code(r"""
LR_ALT = 1e-3 if LR != 1e-3 else 1e-4
run(9, f'MLP + BN, lr {LR_ALT:g}', f'E8 com lr = {LR_ALT:g}',
    lambda: L.MLP([1024, 512, 256], dropout=0.3, batchnorm=True), dados='norm', pai=8,
    epochs=25, batch_size=100, lr=LR_ALT);
B_BN = melhor(8, 9)
print(f"\nCom BatchNorm, a melhor taxa na validação é lr = {B_BN['config']['lr']:g} (E{B_BN['exp']}). Seguimos a partir dele.")
""")

md(r"""
### E10 — batch 256 (em vez de 100)
Batch maior: gradiente menos ruidoso, mas menos atualizações por época.
""")
code(r"""
run(10, 'MLP batch 256', f"E{B_BN['exp']} com batch 256",
    lambda: L.MLP([1024, 512, 256], dropout=0.3, batchnorm=True), dados='norm', pai=B_BN['exp'],
    **dict(B_BN['config'], batch_size=256));
B_BATCH = melhor(B_BN['exp'], 10)
print(f"\nMelhor tamanho de batch na validação: {B_BATCH['config']['batch_size']} (E{B_BATCH['exp']}). Seguimos a partir dele.")
""")

md(r"""
### E11 — taxa de aprendizado com decaimento cosseno
A taxa começa no valor escolhido e cai suavemente até zero: passos grandes no início e ajuste fino no fim.
""")
code(r"""
run(11, 'MLP + lr cosseno', f"E{B_BATCH['exp']} + decaimento cosseno da lr",
    lambda: L.MLP([1024, 512, 256], dropout=0.3, batchnorm=True), dados='norm', pai=B_BATCH['exp'],
    **dict(B_BATCH['config'], scheduler='cosine'));
B_MLP = melhor(*range(2, 12))
print(f"\nMelhor MLP na validação: E{B_MLP['exp']} — {B_MLP['nome']} (val_acc = {B_MLP['val_acc']:.4f}, test_acc = {B_MLP['test_acc']:.4f})")
""")

md(r"""
### E12 — CNN (rede convolucional)
O enunciado deixa a arquitetura livre e cita o "número de filtros". A MLP trata cada pixel como uma entrada
independente; a convolução aplica os mesmos filtros em toda a imagem e aproveita a vizinhança dos pixels.
Arquitetura: 3 blocos convolucionais (32 → 64 → 128 filtros 3×3, BatchNorm, ReLU, max-pool) e um classificador
totalmente conectado com dropout 0,3. Treino: Adam, lr 1e-3 com decaimento cosseno, batch 128, 20 épocas.
""")
code(r"""
run(12, 'CNN', 'CNN 3 blocos conv (32-64-128 filtros)',
    lambda: L.CNN(dropout=0.3), dados='norm', pai=B_MLP['exp'],
    epochs=20, batch_size=128, lr=1e-3, scheduler='cosine');
""")

md(r"""
### E13 — E12 + data augmentation
Translação aleatória de até 2 pixels em cada imagem de treino. Não usamos espelhamento horizontal, que muda o
significado de alguns dígitos.
""")
code(r"""
run(13, 'CNN + augmentation', 'E12 + translação aleatória de ±2 px',
    lambda: L.CNN(dropout=0.3), dados='norm', pai=12,
    epochs=20, batch_size=128, lr=1e-3, scheduler='cosine', augment_shift=2);
""")

md(r"""
### De volta à MLP: E14 a E17

O modelo final precisa ser uma rede totalmente conectada, então os próximos experimentos voltam à melhor MLP e
tentam aproximá-la da CNN. A CNN ganha da MLP porque já "sabe" que um dígito deslocado é o mesmo dígito; a MLP
precisa aprender isso com os dados.

### E14 — melhor MLP + data augmentation
A mesma translação aleatória de ±2 px do E13, agora na melhor MLP. Para a MLP, cada imagem deslocada é um exemplo
novo.
""")
code(r"""
run(14, 'MLP + augmentation', f"E{B_MLP['exp']} + translação aleatória de ±2 px",
    lambda: L.MLP([1024, 512, 256], dropout=0.3, batchnorm=True), dados='norm', pai=B_MLP['exp'],
    **dict(B_MLP['config'], augment_shift=2));
r14 = RESULTS[-1]
print(f"\nAcurácia de treino ({r14['train_acc']:.4f}) x validação ({r14['val_acc']:.4f}) na melhor época.")
""")

md(r"""
### E15 — rede maior e treino mais longo
Com dropout e augmentation, a acurácia de treino do E14 fica **abaixo** da de validação: a rede não está
decorando o treino, está faltando capacidade e tempo de treino (underfitting). Por isso este experimento muda o
"tamanho" do treino como um todo: 4 camadas ocultas de 2048 neurônios, dropout menor (0,1), 60 épocas e taxa de
aprendizado maior (1e-3) com a política *one-cycle* (sobe nos primeiros 10% dos passos e depois decai até quase
zero), otimizador AdamW e batch 128. São várias mudanças juntas, todas na mesma direção (mais capacidade e menos
regularização); o efeito medido é o do conjunto.
""")
code(r"""
MLP_G = lambda: L.MLP([2048, 2048, 2048, 2048], dropout=0.1, batchnorm=True)
KW_G = dict(epochs=60, batch_size=128, lr=1e-3, optimizer='adamw', scheduler='onecycle', augment_shift=2)
run(15, 'MLP grande', 'E14 com rede 4×2048, dropout 0,1, 60 épocas, lr 1e-3 one-cycle',
    MLP_G, dados='norm', pai=14, **KW_G);
""")

md(r"""
### E16 — E15 + normalização por imagem
Além da média e do desvio por canal (do treino), cada imagem é normalizada pela sua própria média e desvio. Fotos
escuras, claras ou de baixo contraste passam a ter a mesma faixa de valores.
""")
code(r"""
run(16, 'MLP grande + norm. por imagem', 'E15 + normalização de contraste por imagem',
    MLP_G, dados='norm_img', pai=15, **KW_G);
""")

md(r"""
### E17 — E16 + rotação e escala no augmentation
Além da translação, cada imagem de treino é girada em até ±10° e ampliada ou reduzida em até 10%.
""")
code(r"""
run(17, 'MLP grande + rotação/escala', 'E16 + rotação ±10° e escala ±10% no augmentation',
    MLP_G, dados='norm_img', pai=16, **dict(KW_G, augment_rot=10.0, augment_scale=0.1));
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
fig, ax = plt.subplots(1, 2, figsize=(14, 4.4))
for r in RESULTS:
    h = json.load(open(f"resultados/E{r['exp']}.json"))['hist']
    ax[0].plot(range(1, len(h['val_acc']) + 1), h['val_acc'], label=f"E{r['exp']} {r['nome']}")
    ax[1].plot(range(1, len(h['val_loss']) + 1), h['val_loss'], label=f"E{r['exp']}")
ax[0].set(title='Acurácia de validação por época', xlabel='época', ylabel='acurácia')
ax[1].set(title='Loss de validação por época', xlabel='época', ylabel='loss', ylim=(0, 2.5))
for a in ax: a.grid(alpha=.3)
ax[0].legend(fontsize=6.5, ncol=2); plt.tight_layout(); plt.savefig('resultados/comparacao_validacao.png', dpi=130); plt.show()
""")

md(r"""
## 5. Escolha do modelo
Critério: **maior acurácia de validação entre as redes totalmente conectadas** (o professor definiu que o modelo
final deve ser uma MLP; a CNN fica como comparação). O teste não participa da escolha.
""")
code(r"""
CNNS = (12, 13)
best = max((r for r in RESULTS if r['exp'] not in CNNS), key=lambda r: r['val_acc'])
cnn = max((r for r in RESULTS if r['exp'] in CNNS), key=lambda r: r['val_acc'])
print(f"Modelo escolhido: E{best['exp']} — {best['nome']}")
print(f"val_acc = {best['val_acc']:.4f} | test_acc = {best['test_acc']:.4f} | melhor época {best['melhor_epoca']} de {best['epocas']}")
print(f"Comparação: a melhor CNN (E{cnn['exp']}) tem val_acc = {cnn['val_acc']:.4f} e test_acc = {cnn['test_acc']:.4f}.")
""")

md(r"""
### Predição com média sobre translações (test-time augmentation)
A MLP não é invariante a translação. Na hora de predizer, em vez de passar a imagem uma única vez, passamos também
as versões deslocadas de −r a +r pixels em cada direção e fazemos a **média das probabilidades**. Os pesos não
mudam; só a forma de predizer. O raio r (0 = sem média, 1 = 9 translações, 2 = 25 translações) é escolhido pela
**validação**.
""")
code(r"""
make_model, dados_best, kw_best = REGISTRY[best['exp']]
m_best = make_model()
m_best.load_state_dict(torch.load(f"pesos/E{best['exp']}.pt")['state_dict'])
d = DATA[dados_best]
TTA = []
for r in (0, 1, 2):
    va = (L.predict_tta(m_best, d['X_val'], r) == d['y_val']).float().mean().item()
    te = (L.predict_tta(m_best, d['X_test'], r) == d['y_test']).float().mean().item()
    TTA.append(dict(raio=r, n_translacoes=(2 * r + 1) ** 2, val_acc=va, test_acc=te))
    print(f"raio {r} ({(2 * r + 1) ** 2:2d} translações): val_acc = {va:.4f} | test_acc = {te:.4f}")
T_BEST = max(TTA, key=lambda t: t['val_acc'])
TTA_R = T_BEST['raio']
print(f"\nRaio escolhido pela validação: {TTA_R}")
with open('resultados/melhor_modelo.json', 'w') as f:
    json.dump(dict(best, tta=TTA, tta_raio=TTA_R, val_acc_tta=T_BEST['val_acc'], test_acc_tta=T_BEST['test_acc'],
                   arquitetura=m_best.config, cnn=cnn), f, indent=1)
""")

md(r"""
## 6. Modelo final: retreino com todo o conjunto de treino

O enunciado permite que a versão final use todo o `train`. Com a configuração já escolhida pela validação
(arquitetura, hiperparâmetros, número de épocas e raio da média sobre translações), retreinamos o modelo com
**treino + validação (73.257 imagens)**, com a mesma semente. Nada é ajustado nesta etapa: não há mais validação,
e o modelo fica com os pesos da última época (a taxa de aprendizado termina perto de zero, então o fim do treino
é estável). O teste continua servindo só para relatar o resultado.
""")
code(r"""
mean_f, std_f = L.compute_norm_stats(X_all)   # estatísticas de todo o treino

def prep(X):
    "Mesmo pré-processamento do experimento escolhido, com as estatísticas de todo o treino."
    if dados_best == 'bruto':
        return L.preprocess(X)
    Xf = L.preprocess(X, mean_f, std_f)
    return L.per_image_norm(Xf) if dados_best == 'norm_img' else Xf

L.set_seed(SEED)
final_model = make_model()
t0 = time.time()
hist_final = L.train_model(final_model, dict(X_train=prep(X_all), y_train=y_all), seed=SEED, device=device, **kw_best)
X_test_f = prep(X_test_u8)
final_test_loss, acc_sem_tta = L.evaluate(final_model, X_test_f, y_test, device=device)
pred_test = L.predict_tta(final_model, X_test_f, TTA_R, device=device)
final_test_acc = (pred_test == y_test).float().mean().item()
print(f"\nModelo final (E{best['exp']} retreinado com treino + validação) | {(time.time() - t0) / 60:.1f} min")
print(f"test_acc = {final_test_acc:.4f} com média sobre translações (raio {TTA_R}) | {acc_sem_tta:.4f} sem a média")
print(f"(o mesmo modelo treinado só com 90%: {T_BEST['test_acc']:.4f} com a média | {best['test_acc']:.4f} sem)")

torch.save({'exp': best['exp'], 'nome': best['nome'], 'state_dict': final_model.state_dict(),
            'arquitetura': final_model.config, 'mean': mean_f, 'std': std_f, 'dados': dados_best,
            'tta_raio': TTA_R, 'config': kw_best,
            'val_acc_selecao': best['val_acc'], 'test_acc': final_test_acc}, 'melhor_modelo.pt')
with open('resultados/final.json', 'w') as f:
    json.dump({'exp': best['exp'], 'nome': best['nome'], 'n_treino': len(X_all), 'hist': hist_final,
               'test_loss': final_test_loss, 'test_acc': final_test_acc, 'test_acc_sem_tta': acc_sem_tta,
               'tta_raio': TTA_R, 'test_acc_selecao': best['test_acc'], 'val_acc_selecao': best['val_acc']}, f, indent=1)
L.plot_history(hist_final, 'Modelo final — retreino com treino + validação', 'resultados/final_curvas.png'); plt.show()
print('pesos salvos em melhor_modelo.pt')
""")

md("### Matriz de confusão do modelo final no teste")
code(r"""
cm = L.confusion_matrix(y_test, pred_test)
acc_classe = cm.diag().float() / cm.sum(1).float()

fig, ax = plt.subplots(figsize=(6.4, 5.6))
im = ax.imshow(cm, cmap='Blues')
for i in range(10):
    for j in range(10):
        v = int(cm[i, j])
        ax.text(j, i, v, ha='center', va='center', fontsize=7, color='white' if v > cm.max() / 2 else 'black')
ax.set(xticks=range(10), yticks=range(10), xlabel='dígito predito', ylabel='dígito verdadeiro',
       title=f'Matriz de confusão no teste (acurácia {final_test_acc:.2%})')
plt.tight_layout(); plt.savefig('resultados/final_confusao.png', dpi=130); plt.show()

print('acurácia por dígito:', {d: round(float(a), 4) for d, a in enumerate(acc_classe)})
erros = cm.clone(); erros.fill_diagonal_(0)
top = torch.topk(erros.flatten(), 5)
print('confusões mais comuns (verdadeiro → predito):',
      [(int(i) // 10, int(i) % 10, int(v)) for v, i in zip(top.values, top.indices)])
with open('resultados/final_confusao.json', 'w') as f:
    json.dump({'cm': cm.tolist(), 'acc_classe': acc_classe.tolist()}, f)
""")

md(r"""
## 7. Predição do conjunto adicional
Feita **somente agora**, com o modelo final já definido. O pré-processamento é o mesmo do treino do modelo final
(mesma média e desvio, mesma média sobre translações). As imagens são lidas em **ordem do nome do arquivo** (ordenação numérica: `2.png` antes de
`10.png`), e o arquivo de saída tem um inteiro de 0 a 9 por linha, na mesma ordem.

As imagens do professor devem estar na pasta `extra/` (conteúdo de `images.zip`).
""")
code(r"""
NOMES = 'JOAOPEDRO_SAMYR_VALBER'
PASTA_EXTRA = 'extra'

if os.path.isdir(PASTA_EXTRA):
    nomes_arq, X_extra_u8 = L.load_image_folder(PASTA_EXTRA)
    preds = L.predict_tta(final_model, prep(X_extra_u8), TTA_R, device=device)
    saida = f'{NOMES}_RESULTS.txt'
    L.write_results(preds, saida)
    print(f'{len(preds)} imagens -> {saida}')
    print('primeiros arquivos:', nomes_arq[:5], '... último:', nomes_arq[-1])
    print('distribuição das predições:', np.bincount(preds.numpy(), minlength=10))
else:
    print(f'Pasta {PASTA_EXTRA}/ não encontrada: baixe o conjunto adicional e rode esta célula de novo.')
""")

import os, re
SMOKE = os.environ.get("SVHN_SMOKE")
if SMOKE:  # versão reduzida só para testar o fluxo do notebook (não é a entrega)
    for c in C:
        c.source = re.sub(r"epochs=\d+", "epochs=2", c.source)
        c.source = c.source.replace("idx_tr, idx_va =", "X_all, y_all, X_test_u8, y_test = X_all[:3000], y_all[:3000], X_test_u8[:800], y_test[:800]\nidx_tr, idx_va =")
        c.source = c.source.replace("PASTA_EXTRA = 'extra'", "PASTA_EXTRA = 'extra_mini'")
nb.cells = C
nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
nbf.write(nb, Path(SMOKE) / "trabalho_svhn.ipynb" if SMOKE else ROOT / "trabalho_svhn.ipynb")
print("ok:", len(C), "células")
