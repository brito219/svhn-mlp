"""Texto do relatório (o layout fica em gerar_relatorio.py).

Os números citados no texto foram conferidos com resultados/*.json. Os do modelo final e da matriz de
confusão são lidos dos JSONs aqui mesmo, para o texto nunca divergir dos resultados.
"""
import json
from pathlib import Path

RES = Path(__file__).resolve().parent.parent / "resultados"
_final = json.load(open(RES / "final.json"))
_best = json.load(open(RES / "melhor_modelo.json"))
_conf = json.load(open(RES / "final_confusao.json"))
pct = lambda v: f"{100 * v:.2f}%".replace(".", ",")

AUTORES = "João Pedro · Samyr · Valber"
DISCIPLINA = "Inteligência Artificial — Trabalho Final"
ARQ_RESULTADOS = "JOAOPEDRO_SAMYR_VALBER_RESULTS.txt"

SOLUCAO = [
    "Implementamos em PyTorch um classificador linear, uma MLP (rede totalmente conectada) e, como experimento "
    "adicional, uma rede convolucional (CNN). O notebook <b>trabalho_svhn.ipynb</b> contém toda a implementação "
    "e executa os 13 experimentos de ponta a ponta. Partimos do código das aulas (classificador linear e "
    "notebook da MLP) e o adaptamos ao SVHN: entrada de 3×32×32 = 3072 valores, divisão treino/validação e "
    "registro de loss e acurácia de treino e de validação a cada época.",
    "Todos os modelos são treinados com cross-entropy (softmax). Ao fim de cada treino, o modelo fica com os "
    "pesos da época de <b>maior acurácia de validação</b> (checkpoint). Assim, a validação escolhe tanto o "
    "experimento quanto o ponto de parada, sem consultar o teste.",
]

PROTOCOLO = [
    "Dados: <i>torchvision.datasets.SVHN</i>, splits <i>train</i> (73.257 imagens) e <i>test</i> (26.032).",
    "Divisão do <i>train</i>: <b>90% treino (65.931) / 10% validação (7.326)</b>, por permutação com semente fixa 42.",
    "Normalização por canal de cor (subtrai a média e divide pelo desvio padrão), com estatísticas calculadas "
    "<b>apenas no conjunto de treino</b> e aplicadas à validação, ao teste e ao conjunto adicional.",
    "A validação acompanha os experimentos e decide todas as escolhas. O conjunto de teste só é usado para "
    "relatar o desempenho de cada experimento.",
    "Cada experimento muda <b>um fator</b> em relação a um experimento anterior. Quando há alternativas (taxa de "
    "aprendizado, tamanho do batch), o próprio notebook segue com a de maior acurácia de validação.",
    "Modelo final: depois de escolhida a configuração, ela é retreinada com todo o <i>train</i> (treino + "
    "validação), como o enunciado permite. O conjunto adicional do professor só foi usado para gerar as "
    "predições, com o modelo já definido; nunca para treino, ajuste ou escolha.",
    "Reprodutibilidade: semente 42 em tudo (inicialização dos pesos, ordem dos mini-batches, dropout e "
    "augmentation). Na mesma máquina, rodar o notebook de novo reproduz exatamente os mesmos números.",
]

# (nome, experimentos, experimento de referência para a contagem de parâmetros, arquitetura)
ARQUITETURAS = [
    ("Classificador linear", "E1", 1, "3072 → 10 (logits = xW + b)"),
    ("MLP da aula", "E2 a E5", 2, "3072 → 500 → 500 → 10, ReLU"),
    ("MLP maior", "E6", 6, "3072 → 1024 → 512 → 256 → 10, ReLU"),
    ("MLP maior + dropout", "E7", 7, "A mesma do E6, com Dropout 0,3 depois de cada camada oculta"),
    ("MLP maior + dropout + BN", "E8 a E11", 8,
     "Cada camada oculta: Linear → BatchNorm → ReLU → Dropout 0,3"),
    ("CNN", "E12 e E13", 12,
     "3 blocos [Conv 3×3 → BatchNorm → ReLU]×2 → MaxPool, com 32, 64 e 128 filtros; depois Dropout 0,3 → "
     "Linear 2048→256 → ReLU → Dropout 0,3 → Linear 256→10"),
]

HIPER_NOTA = [
    "As MLPs foram treinadas por 25 épocas e as CNNs por 20 (o treino foi feito em CPU). O Adam e o batch 100 "
    "vêm do notebook da aula. O classificador linear usa gradiente descendente com momentum 0,9 e "
    "regularização L2.",
]

EXPERIMENTOS_INTRO = (
    "A tabela mostra os 13 experimentos. A coluna <b>Δ val.</b> é a variação da acurácia de validação em relação "
    "ao experimento do qual cada um partiu (indicado na coluna Modificação). Todos os valores são os da época "
    "de melhor validação."
)

TABELA_LEGENDA = ("Loss e acurácia na época de melhor validação (checkpoint). Δ val. em pontos percentuais (pp). "
                  "Linha verde: configuração escolhida para o modelo final.")

VEREDITO = {
    1: "Referência. Underfitting: o modelo linear não tem capacidade",
    2: "Underfitting: a entrada não centrada trava a otimização",
    3: "Melhora grande. Passa a ter overfitting",
    4: "Piora: o treino colapsa (taxa alta demais)",
    5: "Melhora. Melhor taxa; adotada a partir daqui",
    6: "Empate. Mais overfitting",
    7: "Melhora. O overfitting desaparece",
    8: "Empate na validação",
    9: "Empate com o E8. Com BN, as duas taxas funcionam",
    10: "Empate na validação, pior no teste. Mantido o batch 100",
    11: "Melhora. Melhor MLP",
    12: "Melhora grande. Tem overfitting",
    13: "Melhora. Menos overfitting. Modelo escolhido",
}

ANALISE_EXPERIMENTOS = [
    "<b>Classificador linear (E1).</b> Chegou a 28,8% na validação, com a loss parada em 2,15, pouco abaixo de "
    "ln 10 ≈ 2,30 (o valor de um chute uniforme entre 10 classes). Treino e validação ficam iguais: é "
    "underfitting por falta de capacidade. Um hiperplano por classe sobre os pixels não dá conta da variação de "
    "cor, fundo e posição dos dígitos.",
    "<b>MLP da aula (E2) e normalização (E3).</b> Com a entrada só em [0, 1], a MLP parou em 58,4%, com treino e "
    "validação colados (57,2% e 58,4%) e loss alta (1,30): underfitting. Subtrair a média e dividir pelo desvio "
    "de cada canal (E3) foi a mudança isolada de maior impacto do trabalho: <b>+24,7 pp</b>, chegando a 83,0%. "
    "Com a entrada centrada, a otimização funciona, e aparece o overfitting: a loss de validação atinge o mínimo "
    "na época 9 (0,66) e sobe até 0,88, enquanto a de treino continua caindo.",
    "<b>Taxa de aprendizado (E4 e E5).</b> Com uma taxa 10 vezes maior (1e-2), o treino colapsa logo na primeira "
    "época: a rede passa a responder o dígito 1, o mais frequente, para praticamente todas as imagens, e fica "
    "nos 19,7% que essa classe representa. Com uma taxa 10 vezes menor (1e-4), o resultado melhora "
    "<b>3,2 pp</b> (86,2%) e o overfitting diminui. A taxa do notebook da aula (1e-3) era alta demais para este "
    "problema, e os experimentos seguintes partem do E5.",
    "<b>Capacidade e regularização (E6 a E8).</b> Dobrar o tamanho da rede (E6, 3,8 milhões de parâmetros) não "
    "melhorou a validação (+0,04 pp) e aumentou o overfitting: a acurácia de treino chegou a 95,5%, 9,9 pp acima "
    "da de validação. O <b>dropout</b> (E7) eliminou essa diferença (86,8% no treino e 86,9% na validação) e "
    "melhorou a validação em 0,7 pp e o teste em 1,0 pp. A <b>BatchNorm</b> (E8) manteve o resultado (+0,1 pp).",
    "<b>BatchNorm e taxa de aprendizado (E9).</b> Repetimos o E8 com a taxa 1e-3: deu 87,0%, empate com o E8 "
    "(diferença de uma imagem). Sem BN, a mesma troca custava 3,2 pp (E3 contra E5). A BatchNorm, portanto, "
    "deixou a rede bem menos sensível à taxa de aprendizado. Seguimos com o E8, que ficou à frente.",
    "<b>Batch e agenda da taxa (E10 e E11).</b> O batch 256 empatou na validação (−0,03 pp) e piorou o teste em "
    "0,4 pp; mantivemos o batch 100. O <b>decaimento cosseno</b> da taxa de aprendizado deu o último ganho das "
    "MLPs: +0,8 pp na validação e +1,1 pp no teste. A melhor MLP (E11) tem 87,8% na validação e 85,5% no teste.",
    "<b>CNN (E12) e augmentation (E13).</b> O enunciado deixa a arquitetura livre e cita o número de filtros "
    "entre as estratégias. A CNN chegou a 95,5% na validação, <b>+7,7 pp</b> sobre a melhor MLP, com 4,7 vezes "
    "menos parâmetros (815 mil contra 3,8 milhões). Ela aplica os mesmos filtros em toda a imagem e aproveita a "
    "vizinhança dos pixels, enquanto a MLP trata cada pixel como uma entrada independente. Sem augmentation, a "
    "CNN ainda tem overfitting (98,4% no treino; a loss de validação sobe de 0,20 para 0,24). A translação "
    "aleatória de ±2 px (E13) reduz essa diferença de 3,0 para 1,0 pp e melhora a validação em 0,3 pp e o "
    "teste em 1,1 pp (96,3%).",
    "Nas MLPs, a acurácia de teste ficou de 2 a 4 pp abaixo da de validação. A validação vem da mesma "
    "distribuição do treino, e o <i>test</i> do SVHN é um pouco mais difícil. Nas CNNs essa diferença "
    "desaparece (E12: 95,2% contra 95,5%; E13: 96,3% contra 95,8%), outro sinal de que elas generalizam melhor.",
]

CURVAS_INTRO = (
    "As figuras mostram loss e acurácia de treino e de validação a cada época. A acurácia de treino é a média "
    "ao longo da época, com dropout e augmentation ativos; por isso pode ficar abaixo da de validação, que é "
    "medida com o modelo em modo de avaliação. A linha pontilhada marca a época escolhida pela validação. As "
    "curvas dos 13 experimentos estão no notebook."
)

# (arquivo em resultados/, legenda, comentário, largura relativa)
CURVAS = [
    ("comparacao_validacao.png", "Acurácia e loss de validação por época, todos os experimentos.", "", 1.0),
    ("E1_curvas.png", "E1, classificador linear.",
     "<b>E1 — underfitting.</b> Treino e validação andam juntos e quase não saem do lugar: a loss cai de 2,29 "
     "para 2,14 em 20 épocas. O treino é estável; o limite é o modelo.", 0.7),
    ("E2_curvas.png", "E2, MLP baseline da aula.",
     "<b>E2 — underfitting.</b> As duas curvas estabilizam em torno de 58%, com loss de 1,30. Há pequenas "
     "oscilações na validação, mas nenhum sinal de overfitting: o modelo não consegue ajustar nem o treino.", 0.7),
    ("E3_curvas.png", "E3, MLP com entrada normalizada.",
     "<b>E3 — overfitting.</b> A partir da época 9, a loss de validação sobe (0,66 → 0,88) enquanto a de treino "
     "continua caindo. A acurácia de validação estaciona perto de 82–83% e a de treino passa de 90%. O "
     "checkpoint pela melhor validação (época 13) limita o dano.", 0.7),
    ("comparacao_lr.png", "Efeito da taxa de aprendizado na MLP normalizada (E3, E4 e E5).",
     "<b>Taxa de aprendizado.</b> Com 1e-2 (E4), a loss fica travada em 2,24 desde a segunda época: o treino "
     "colapsou. Com 1e-3 (E3), o treino converge, mas a loss de validação volta a subir cedo. Com 1e-4 (E5), a "
     "loss de validação desce mais (0,51) e a acurácia é a mais alta das três.", 1.0),
    ("E11_curvas.png", "E11, melhor MLP (dropout, BatchNorm e lr com decaimento cosseno).",
     "<b>E11 — melhor MLP.</b> As curvas de treino e de validação ficam próximas até o fim, a loss de validação "
     "cai de forma contínua até 0,40 e o decaimento cosseno suaviza as últimas épocas. Não há overfitting nem "
     "instabilidade; a validação fica acima do treino porque o dropout só atua no treino.", 0.7),
    ("E12_curvas.png", "E12, CNN sem augmentation.",
     "<b>E12 — CNN com overfitting.</b> A validação já começa em 88% e chega a 95,5%. Depois da época 8, a loss "
     "de validação sobe de 0,20 para 0,24, enquanto a de treino cai até 0,05 (98,4% de acurácia de treino).",
     0.7),
    ("E13_curvas.png", "E13, CNN com augmentation (configuração escolhida).",
     "<b>E13 — CNN com augmentation.</b> A loss de validação cai até a última época (0,16) e a diferença entre "
     "treino e validação fica em 1,0 pp. O overfitting do E12 foi contido, e a curva ainda não havia saturado "
     "por completo ao fim das 20 épocas.", 0.7),
]

MELHOR = [
    f"A configuração escolhida é a <b>CNN com augmentation (E13)</b>, por ter a maior acurácia de validação "
    f"entre os 13 experimentos ({pct(_best['val_acc'])}, contra 87,78% da melhor MLP). O teste não participou "
    f"da escolha.",
    f"Como o enunciado permite, o modelo final é essa mesma configuração <b>retreinada com todo o conjunto de "
    f"treino</b> (73.257 imagens: treino + validação), com a mesma semente, as mesmas 20 épocas e nenhum outro "
    f"ajuste. Ele atinge <b>{pct(_final['test_acc'])} no teste</b> (o E13, treinado com 90% dos dados, teve "
    f"{pct(_final['test_acc_selecao'])}). Foi esse modelo que gerou o arquivo de predições do conjunto "
    f"adicional: 203.856 imagens, lidas em ordem do nome do arquivo, com a mesma normalização do treino e uma "
    f"predição de 0 a 9 por linha.",
]

MELHOR_ARQ = dict((a[0], a[3]) for a in ARQUITETURAS)["CNN"]
MELHOR_HP = ("Adam, lr inicial 1e-3 com decaimento cosseno, batch 128, 20 épocas, dropout 0,3, augmentation por "
             "translação aleatória de ±2 px (sem espelhamento, que altera dígitos), entrada normalizada por canal.")


def _erros():
    cm, acc = _conf["cm"], _conf["acc_classe"]
    pares = sorted(((cm[i][j], i, j) for i in range(10) for j in range(10) if i != j), reverse=True)[:3]
    pior, melhor = min(range(10), key=lambda d: acc[d]), max(range(10), key=lambda d: acc[d])
    lista = "; ".join(f"{i} classificado como {j} ({n} vezes)" for n, i, j in pares)
    return [f"A acurácia por dígito vai de {pct(acc[pior])} (dígito {pior}) a {pct(acc[melhor])} (dígito "
            f"{melhor}). As confusões mais comuns são entre dígitos de forma parecida: {lista}."]


MELHOR_ERROS = _erros()

CONCLUSAO = [
    f"A abordagem que funcionou melhor foi a <b>rede convolucional com augmentation</b>: 95,80% na validação e, "
    f"no modelo final retreinado com todo o treino, {pct(_final['test_acc'])} no teste. A melhor MLP ficou em "
    f"87,78% na validação e 85,46% no teste, e o classificador linear, em 28,83% e 24,02%.",
    "As modificações de maior impacto foram, em ordem: (1) a <b>normalização da entrada</b> (+24,7 pp); (2) a "
    "<b>troca da MLP pela CNN</b> (+7,7 pp); (3) a <b>taxa de aprendizado</b> correta (+3,2 pp, e uma taxa alta "
    "demais derruba o modelo para 19,7%); (4) o <b>decaimento cosseno</b> (+0,8 pp), o <b>dropout</b> (+0,7 pp) "
    "e a <b>augmentation</b> (+0,3 pp na validação e +1,1 pp no teste). Aumentar a rede sem regularizar e "
    "aumentar o batch não ajudaram. Dropout e augmentation tiveram ganho modesto na acurácia de validação, mas "
    "foram o que eliminou o overfitting.",
]

DIFICULDADES = [
    "<b>Reprodutibilidade.</b> Mesmo com a semente fixa, duas execuções davam resultados ligeiramente "
    "diferentes nos modelos com entrada normalizada. A causa era a média por canal calculada em float32: somar "
    "67 milhões de valores dá um resultado que muda na 7ª casa decimal conforme a disposição do tensor na "
    "memória, e essa diferença se amplifica no treino. Passamos a calcular a média e o desvio com somas "
    "inteiras exatas, e os resultados ficaram idênticos entre execuções.",
    "<b>Treino em CPU.</b> Sem GPU, cada MLP levou de 3 a 12 minutos e cada CNN, cerca de 55 minutos. Isso "
    "limitou o número de épocas e a quantidade de combinações de hiperparâmetros testadas.",
    "<b>Encadear os experimentos.</b> Na primeira versão, os experimentos seguiam com a taxa de aprendizado do "
    "notebook da aula mesmo depois de a validação mostrar que outra era melhor, e o dropout parecia piorar o "
    "modelo. Refizemos a sequência para que cada passo parta do melhor resultado anterior na validação.",
    "<b>Leitura das curvas com dropout.</b> A acurácia de treino abaixo da de validação não é erro: o dropout e "
    "a augmentation só atuam durante o treino.",
    "<b>Download do SVHN.</b> O servidor de Stanford ficou instável em parte do desenvolvimento, e usamos "
    "temporariamente um espelho. Com os arquivos oficiais, verificamos que as imagens e os rótulos são idênticos.",
]

ENTREGAVEIS = [
    "<b>trabalho_svhn.ipynb</b>: implementação e todos os experimentos, com as saídas (também grava "
    "<i>svhn_lib.py</i>, com 23 testes automatizados em <i>tests/</i>).",
    "<b>melhor_modelo.pt</b>: pesos do modelo final, com a média e o desvio usados na normalização.",
    f"<b>{ARQ_RESULTADOS}</b>: 203.856 linhas, uma predição (0 a 9) por imagem do conjunto adicional.",
    "<b>relatorio.pdf</b>: este relatório. O histórico completo de cada experimento está em <i>resultados/</i>.",
]
