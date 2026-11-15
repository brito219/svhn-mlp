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

AUTORES = "João Pedro Santos de Brito · Samyr Arruda Villanova · Valber Oliveira da Conceição"
DISCIPLINA = "Trabalho Final de Inteligência Artificial"
ARQ_RESULTADOS = "JOAOBRITO_SAMYRARRUDA_VALBERCONCEICAO_RESULTS.txt"

SOLUCAO = [
    "Implementamos em PyTorch um classificador linear, várias MLPs (redes totalmente conectadas) e, como "
    "comparação, uma rede convolucional (CNN). Seguindo a orientação do professor, o <b>modelo final é uma "
    "MLP</b>, e a CNN aparece apenas para mostrar como as duas abordagens se comparam, sem concorrer à escolha "
    "final. Partimos do código das aulas, o classificador linear e o notebook da MLP, e o adaptamos ao SVHN, com "
    "entrada de 3×32×32 = 3072 valores, divisão entre treino e validação e registro de loss e acurácia de treino "
    "e de validação a cada época. O notebook <b>trabalho_svhn.ipynb</b> contém toda a implementação e executa os "
    "17 experimentos de ponta a ponta.",
    "Todos os modelos são treinados com cross-entropy (softmax) e, ao fim de cada treino, ficam com os pesos da "
    "época de <b>maior acurácia de validação</b> (checkpoint), de modo que a validação escolhe tanto o "
    "experimento quanto o ponto de parada, sem nunca consultar o teste.",
]

PROTOCOLO = [
    "Usamos os splits <i>train</i> (73.257 imagens) e <i>test</i> (26.032) do <i>torchvision.datasets.SVHN</i>. "
    "O <i>train</i> foi dividido em <b>90% para treino (65.931 imagens) e 10% para validação (7.326)</b>, por "
    "uma permutação com semente fixa 42. A entrada é normalizada por canal de cor, subtraindo a média e dividindo "
    "pelo desvio padrão, com estatísticas calculadas <b>apenas no conjunto de treino</b> e aplicadas depois à "
    "validação, ao teste e ao conjunto adicional. A partir do E16, cada imagem também passa a ser normalizada "
    "pela própria média e desvio (normalização de contraste por imagem), o que não depende de nenhum outro dado.",
    "A validação acompanha os experimentos e decide todas as escolhas, enquanto o conjunto de teste serve apenas "
    "para relatar o desempenho de cada um. Cada experimento muda <b>um fator</b> em relação a um experimento "
    "anterior, com exceção do E15, explicada na seção 2, e, quando há alternativas, como a taxa de aprendizado "
    "ou o tamanho do batch, o próprio notebook segue com a de maior acurácia de validação.",
    "O modelo final é a MLP de maior acurácia de validação. Depois de escolhida a configuração, ela foi "
    "retreinada com todo o <i>train</i> (treino e validação), como o enunciado permite. O conjunto adicional do "
    "professor só foi usado para gerar as predições, com o modelo já definido, e nunca para treino, ajuste ou "
    "escolha. Usamos a semente 42 em tudo (inicialização dos pesos, ordem dos mini-batches, dropout e "
    "augmentation), e, na mesma máquina, rodar o notebook de novo reproduz exatamente os mesmos números.",
]

# (nome, experimentos, experimento de referência para a contagem de parâmetros, arquitetura)
ARQUITETURAS = [
    ("Classificador linear", "E1", 1, "3072 → 10 (logits = xW + b)"),
    ("MLP da aula", "E2 a E5", 2, "3072 → 500 → 500 → 10, ReLU"),
    ("MLP maior", "E6", 6, "3072 → 1024 → 512 → 256 → 10, ReLU"),
    ("MLP maior + dropout", "E7", 7, "A mesma do E6, com Dropout 0,3 depois de cada camada oculta"),
    ("MLP maior + dropout + BN", "E8 a E11 e E14", 8,
     "Cada camada oculta: Linear → BatchNorm → ReLU → Dropout 0,3"),
    ("MLP grande", "E15 a E17", 15,
     "3072 → 2048 → 2048 → 2048 → 2048 → 10; cada camada oculta: Linear → BatchNorm → ReLU → Dropout 0,1"),
    ("CNN (comparação)", "E12 e E13", 12,
     "3 blocos [Conv 3×3 → BatchNorm → ReLU]×2 → MaxPool, com 32, 64 e 128 filtros; depois Dropout 0,3 → "
     "Linear 2048→256 → ReLU → Dropout 0,3 → Linear 256→10"),
]

HIPER_NOTA = [
    "As MLPs dos experimentos E2 a E14 foram treinadas por 25 épocas, as CNNs por 20 e as MLPs grandes por 60, "
    "já que todo o treino foi feito em CPU. O Adam e o batch 100 vêm do notebook da aula, e o classificador "
    "linear usa gradiente descendente com momentum 0,9 e regularização L2. Na política <i>one-cycle</i>, a taxa "
    "de aprendizado sobe até o valor indicado nos primeiros 10% dos passos e depois decai até quase zero.",
]

EXPERIMENTOS_INTRO = (
    "A tabela mostra os 17 experimentos, com os valores da época de melhor validação. A coluna <b>Δ val.</b> é "
    "a variação da acurácia de validação em relação ao experimento do qual cada um partiu, indicado na coluna "
    "Modificação."
)

TABELA_LEGENDA = ("Loss e acurácia na época de melhor validação (checkpoint), com Δ val. em pontos percentuais (pp). "
                  "A linha verde marca a configuração escolhida para o modelo final, a melhor MLP.")

VEREDITO = {
    1: "Referência. Underfitting: o modelo linear não tem capacidade",
    2: "Underfitting: a entrada não centrada trava a otimização",
    3: "Melhora grande. Passa a ter overfitting",
    4: "Piora: o treino colapsa (taxa alta demais)",
    5: "Melhora. Melhor taxa, adotada a partir daqui",
    6: "Empate. Mais overfitting",
    7: "Melhora. O overfitting desaparece",
    8: "Empate na validação",
    9: "Empate com o E8. Com BN, as duas taxas funcionam",
    10: "Empate na validação, pior no teste. Mantido o batch 100",
    11: "Melhora. Melhor MLP",
    12: "Melhora grande. Tem overfitting. Só comparação",
    13: "Melhora. Menos overfitting. Melhor resultado geral, só comparação",
    14: "Empate na validação, melhora no teste. Underfitting",
    15: "Melhora grande",
    16: "Melhora",
    17: "Melhora. Menos overfitting. Modelo escolhido",
}

ANALISE_EXPERIMENTOS = [
    "<b>Classificador linear (E1).</b> O classificador linear chegou a 28,8% na validação, com a loss parada em "
    "2,15, pouco abaixo de ln 10 ≈ 2,30, que é o valor de um chute uniforme entre 10 classes. Como treino e "
    "validação ficam iguais, trata-se de underfitting por falta de capacidade, já que um hiperplano por classe "
    "sobre os pixels não dá conta da variação de cor, fundo e posição dos dígitos.",
    "<b>MLP da aula (E2) e normalização (E3).</b> Com a entrada só em [0, 1], a MLP parou em 58,4%, com treino e "
    "validação colados (57,2% e 58,4%) e loss alta (1,30), o que também caracteriza underfitting. Subtrair a "
    "média e dividir pelo desvio de cada canal (E3) foi a mudança isolada de maior impacto do trabalho, com "
    "<b>+24,7 pp</b>, chegando a 83,0%. Com a entrada centrada, a otimização passa a funcionar e aparece o "
    "overfitting, pois a loss de validação atinge o mínimo na época 9 (0,66) e sobe até 0,88, enquanto a de "
    "treino continua caindo.",
    "<b>Taxa de aprendizado (E4 e E5).</b> Com uma taxa 10 vezes maior (1e-2), o treino colapsa logo na primeira "
    "época, e a rede passa a responder o dígito 1, o mais frequente, para praticamente todas as imagens, ficando "
    "nos 19,7% que essa classe representa. Já com uma taxa 10 vezes menor (1e-4), o resultado melhora "
    "<b>3,2 pp</b> (86,2%) e o overfitting diminui. A taxa do notebook da aula (1e-3) era, portanto, alta demais "
    "para este problema, e os experimentos seguintes partem do E5.",
    "<b>Capacidade e regularização (E6 a E8).</b> Dobrar o tamanho da rede (E6, com 3,8 milhões de parâmetros) "
    "não melhorou a validação (+0,04 pp) e aumentou o overfitting, com a acurácia de treino chegando a 95,5%, "
    "9,9 pp acima da de validação. O <b>dropout</b> (E7) eliminou essa diferença, com 86,8% no treino e 86,9% na "
    "validação, e melhorou a validação em 0,7 pp e o teste em 1,0 pp, enquanto a <b>BatchNorm</b> (E8) manteve "
    "o resultado (+0,1 pp).",
    "<b>BatchNorm e taxa de aprendizado (E9).</b> Ao repetir o E8 com a taxa 1e-3, obtivemos 87,0%, um empate "
    "com o E8 (diferença de uma imagem). Sem BN, a mesma troca custava 3,2 pp (E3 contra E5), ou seja, a "
    "BatchNorm deixou a rede bem menos sensível à taxa de aprendizado. Seguimos com o E8, que ficou à frente.",
    "<b>Batch e agenda da taxa (E10 e E11).</b> O batch 256 empatou na validação (−0,03 pp) e piorou o teste em "
    "0,4 pp, então mantivemos o batch 100. O <b>decaimento cosseno</b> da taxa de aprendizado deu o último ganho "
    "das MLPs pequenas, com +0,8 pp na validação e +1,1 pp no teste, e a melhor delas (E11) chegou a 87,8% na "
    "validação e 85,5% no teste.",
    "<b>CNN (E12) e augmentation (E13), como comparação.</b> A CNN chegou a 95,5% na validação, <b>+7,7 pp</b> "
    "sobre a melhor MLP até ali, com 4,7 vezes menos parâmetros (815 mil contra 3,8 milhões). Isso acontece "
    "porque ela aplica os mesmos filtros em toda a imagem e aproveita a vizinhança dos pixels, enquanto a MLP "
    "trata cada pixel como uma entrada independente. Sem augmentation, a CNN tem overfitting, com 98,4% no "
    "treino e a loss de validação subindo de 0,20 para 0,24, e a translação aleatória de ±2 px (E13) reduz a "
    "diferença entre treino e validação de 3,0 para 1,0 pp, melhorando a validação em 0,3 pp e o teste em "
    "1,1 pp (96,3%).",
    "<b>Augmentation na MLP (E14).</b> A mesma translação, aplicada à melhor MLP (E11), empatou na validação "
    "(+0,05 pp) e melhorou o teste em 0,8 pp, mas o dado mais útil foi outro, pois a acurácia de treino caiu "
    "para 80,4%, 7,4 pp <b>abaixo</b> da de validação. Com dropout 0,3 e augmentation, a rede de 3,8 milhões de "
    "parâmetros já não conseguia ajustar o treino em 25 épocas com taxa 1e-4, e o gargalo passou a ser o "
    "underfitting.",
    "<b>Rede maior e treino mais longo (E15).</b> Para atacar esse underfitting, o E15 muda o treino como um "
    "todo, com 4 camadas ocultas de 2048 neurônios (18,9 milhões de parâmetros), dropout 0,1, 60 épocas, taxa "
    "1e-3 com política one-cycle, AdamW e batch 128. São várias mudanças juntas, todas na mesma direção (mais "
    "capacidade e menos regularização), e por isso o efeito medido é o do conjunto, de <b>+4,4 pp</b> na "
    "validação (92,2%) e +4,3 pp no teste, o maior ganho entre as MLPs depois da normalização da entrada.",
    "<b>Normalização por imagem (E16).</b> Normalizar cada imagem pela própria média e desvio deu <b>+1,0 pp</b> "
    "na validação (93,2%) e +1,2 pp no teste. As fotos do SVHN variam muito em brilho e contraste, e a MLP não "
    "tem como compensar isso sozinha. Em contrapartida, o overfitting aumentou, com 97,8% no treino e a loss de "
    "validação parando de cair por volta da época 34.",
    "<b>Rotação e escala (E17).</b> Girar as imagens de treino em até ±10° e variar a escala em até 10% deu "
    "+0,3 pp na validação (93,5%) e +0,7 pp no teste (92,4%), além de reduzir o overfitting, com 96,7% no treino "
    "e a loss de validação caindo até a época 56, quando chega a 0,26. Essa é a melhor MLP e a configuração "
    "escolhida, que fica a 2,3 pp da melhor CNN na validação, mesmo com 23 vezes mais parâmetros.",
    "Nas MLPs, a acurácia de teste ficou de 1 a 4 pp abaixo da de validação, já que a validação vem da mesma "
    "distribuição do treino e o <i>test</i> do SVHN é um pouco mais difícil. Nas CNNs essa diferença desaparece "
    "(95,2% contra 95,5% no E12 e 96,3% contra 95,8% no E13), outro sinal de que elas generalizam melhor.",
]

CURVAS_INTRO = (
    "As figuras mostram a loss e a acurácia de treino e de validação a cada época, e a linha pontilhada marca a "
    "época escolhida pela validação. A acurácia de treino é a média ao longo da época, com dropout e augmentation "
    "ativos, por isso pode ficar abaixo da de validação, que é medida com o modelo em modo de avaliação. As "
    "curvas de todos os 17 experimentos estão no notebook."
)

# (arquivo em resultados/, legenda, comentário, largura relativa)
CURVAS = [
    ("comparacao_validacao.png", "Acurácia e loss de validação por época, todos os experimentos.", "", 1.0),
    ("E1_curvas.png", "E1, classificador linear.",
     "<b>E1: underfitting.</b> Treino e validação andam juntos e quase não saem do lugar, com a loss caindo de "
     "2,29 para 2,14 em 20 épocas. O treino é estável, e o limite está no próprio modelo.", 0.7),
    ("E2_curvas.png", "E2, MLP baseline da aula.",
     "<b>E2: underfitting.</b> As duas curvas estabilizam em torno de 58%, com loss de 1,30. Há pequenas "
     "oscilações na validação, mas nenhum sinal de overfitting, já que o modelo não consegue ajustar nem o "
     "treino.", 0.7),
    ("E3_curvas.png", "E3, MLP com entrada normalizada.",
     "<b>E3: overfitting.</b> A partir da época 9, a loss de validação sobe de 0,66 para 0,88 enquanto a de "
     "treino continua caindo, e a acurácia de validação estaciona entre 82% e 83% enquanto a de treino passa de "
     "90%. O checkpoint pela melhor validação (época 13) limita o dano.", 0.7),
    ("comparacao_lr.png", "Efeito da taxa de aprendizado na MLP normalizada (E3, E4 e E5).",
     "<b>Taxa de aprendizado.</b> Com 1e-2 (E4), a loss fica travada em 2,24 desde a segunda época, sinal de que "
     "o treino colapsou. Com 1e-3 (E3), o treino converge, mas a loss de validação volta a subir cedo, e com "
     "1e-4 (E5) ela desce mais (0,51) e a acurácia é a mais alta das três.", 1.0),
    ("E13_curvas.png", "E13, CNN com augmentation (comparação).",
     "<b>E13: CNN com augmentation.</b> A validação já começa perto de 90%, a loss de validação cai até a última "
     "época (0,16) e a diferença entre treino e validação fica em 1,0 pp. Sem augmentation (E12), a loss de "
     "validação subia de 0,20 para 0,24 depois da época 8.", 0.7),
    ("E14_curvas.png", "E14, melhor MLP pequena com augmentation.",
     "<b>E14: underfitting por regularização.</b> A curva de treino fica bem abaixo da de validação o tempo todo "
     "(80% contra 88% no fim), e as duas ainda sobem na última época. Não há overfitting, falta capacidade e "
     "tempo de treino, e foi isso que motivou o E15.", 0.7),
    ("E17_curvas.png", "E17, MLP grande com normalização por imagem e augmentation (configuração escolhida).",
     "<b>E17: configuração escolhida.</b> Nas primeiras épocas, enquanto a taxa de aprendizado sobe (one-cycle), "
     "a validação oscila, e depois as curvas ficam suaves. O treino passa a validação por volta da metade e "
     "termina 3,2 pp acima (96,7% contra 93,5%), o que indica um overfitting moderado, mas a loss de validação "
     "não volta a subir e se estabiliza em 0,26 nas últimas épocas.", 0.7),
    ("final_curvas.png", "Modelo final, o E17 retreinado com treino e validação (sem curva de validação).",
     "", 0.7),
]

_t = {t["raio"]: t for t in _best["tta"]}
_r = _best["tta_raio"]
_n = (2 * _r + 1) ** 2

MELHOR = [
    f"Como o modelo final precisa ser uma rede totalmente conectada, escolhemos, entre as MLPs, a <b>MLP grande "
    f"com normalização por imagem e augmentation (E17)</b>, que teve a maior acurácia de validação "
    f"({pct(_best['val_acc'])}), sem que o teste participasse da escolha. A melhor CNN (E13) teve "
    f"{pct(_best['cnn']['val_acc'])} na validação e fica no trabalho como comparação.",
    f"<b>Predição com média sobre translações.</b> Como a MLP não é invariante a translação, na predição o "
    f"modelo recebe, além da imagem original, as versões deslocadas de −r a +r pixels em cada direção, e a classe "
    f"escolhida é a de maior probabilidade média, sem nenhuma mudança nos pesos. O raio foi escolhido pela "
    f"validação, que deu {pct(_t[0]['val_acc'])} sem a média, {pct(_t[1]['val_acc'])} com r = 1 (9 translações) "
    f"e {pct(_t[2]['val_acc'])} com r = 2 (25 translações). Adotamos r = {_r}, que no teste leva o E17 de "
    f"{pct(_t[0]['test_acc'])} para {pct(_t[_r]['test_acc'])}.",
    f"Como o enunciado permite, o modelo final é essa mesma configuração <b>retreinada com todo o conjunto de "
    f"treino</b> (73.257 imagens, somando treino e validação), com a mesma semente, as mesmas 60 épocas e nenhum "
    f"outro ajuste. Ele atinge <b>{pct(_final['test_acc'])} no teste</b> ({pct(_final['test_acc_sem_tta'])} sem "
    f"a média sobre translações) e foi o modelo usado para gerar o arquivo de predições do conjunto adicional, "
    f"com as 203.856 imagens lidas em ordem do nome do arquivo, o mesmo pré-processamento do treino e uma "
    f"predição de 0 a 9 por linha.",
]

MELHOR_ARQ = dict((a[0], a[3]) for a in ARQUITETURAS)["MLP grande"]
MELHOR_HP = ("AdamW, lr máxima 1e-3 com política one-cycle, batch 128, 60 épocas, dropout 0,1 e BatchNorm, entrada "
             "normalizada por canal e por imagem, augmentation com translação de ±2 px, rotação de ±10° e escala "
             f"de ±10% (sem espelhamento, que altera dígitos) e predição com média sobre {_n} translações.")


def _erros():
    cm, acc = _conf["cm"], _conf["acc_classe"]
    pares = sorted(((cm[i][j], i, j) for i in range(10) for j in range(10) if i != j), reverse=True)[:3]
    pior, melhor = min(range(10), key=lambda d: acc[d]), max(range(10), key=lambda d: acc[d])
    itens = [f"{i} classificado como {j} ({n} vezes)" for n, i, j in pares]
    lista = ", ".join(itens[:-1]) + " e " + itens[-1]
    return [f"A acurácia por dígito vai de {pct(acc[pior])} (dígito {pior}) a {pct(acc[melhor])} (dígito "
            f"{melhor}), e as confusões mais comuns são entre dígitos de forma parecida, como {lista}."]


MELHOR_ERROS = _erros()

CONCLUSAO = [
    f"Entre as redes totalmente conectadas, a abordagem que funcionou melhor foi a <b>MLP grande (4×2048) com "
    f"normalização por imagem, augmentation e média sobre translações na predição</b>, com "
    f"{pct(_best['val_acc_tta'])} na validação e, no modelo final retreinado com todo o treino, "
    f"{pct(_final['test_acc'])} no teste. O caminho foi de 24,0% no teste com o classificador linear para 55,0% "
    f"com a MLP da aula, 85,5% com a melhor MLP pequena e, por fim, {pct(_final['test_acc'])}. A CNN, usada como "
    f"comparação, chegou a 96,3% no teste com 23 vezes menos parâmetros, o que mostra que uma arquitetura "
    f"adequada a imagens ainda vale mais do que todas as melhorias feitas na MLP.",
    "As modificações de maior impacto nas MLPs foram, em ordem, a <b>normalização da entrada</b> por canal "
    "(+24,7 pp), a <b>rede maior com treino mais longo</b>, adotada depois que a augmentation mostrou que o "
    "problema era underfitting (+4,4 pp), a <b>taxa de aprendizado</b> correta (+3,2 pp, sendo que uma taxa alta "
    "demais derruba o modelo para 19,7%), a <b>normalização por imagem</b> (+1,0 pp), a <b>média sobre "
    "translações</b> na predição (+0,8 pp), o <b>decaimento cosseno</b> (+0,8 pp) e o <b>dropout</b> (+0,7 pp). "
    "Aumentar a rede sem regularizar (E6) e aumentar o batch não ajudaram. A augmentation sozinha quase não mudou "
    "a validação da MLP pequena, mas foi ela que permitiu treinar uma rede cinco vezes maior sem que essa rede "
    "decorasse o treino.",
]

DIFICULDADES = [
    "A principal dificuldade foi a <b>reprodutibilidade</b>. Mesmo com a semente fixa, duas execuções davam "
    "resultados ligeiramente diferentes nos modelos com entrada normalizada, e a causa era a média por canal "
    "calculada em float32, já que somar 67 milhões de valores dá um resultado que muda na 7ª casa decimal "
    "conforme a disposição do tensor na memória, e essa diferença se amplifica no treino. Passamos a calcular a "
    "média e o desvio com somas inteiras exatas, e os resultados ficaram idênticos entre execuções.",
    "O <b>treino em CPU</b> também pesou, pois sem GPU cada MLP pequena levou de 4 a 12 minutos, cada CNN cerca "
    "de 55 minutos e cada MLP grande cerca de 95 minutos, o que limitou o número de épocas e de combinações de "
    "hiperparâmetros testadas. Além disso, na primeira versão os experimentos seguiam com a taxa de aprendizado "
    "do notebook da aula mesmo depois de a validação mostrar que outra era melhor, o que fazia o dropout parecer "
    "piorar o modelo, e por isso refizemos a sequência para que cada passo parta do melhor resultado anterior na "
    "validação.",
    "Também foi preciso cuidado na leitura das curvas com dropout, em que a acurácia de treino abaixo da de "
    "validação não é erro, já que o dropout e a augmentation só atuam durante o treino. Por fim, o servidor de "
    "Stanford ficou instável em parte do desenvolvimento, e usamos temporariamente um espelho do SVHN, "
    "verificando depois, com os arquivos oficiais, que as imagens e os rótulos são idênticos.",
]

ENTREGAVEIS = [
    f"Junto com este relatório (<b>relatorio.pdf</b>) seguem o notebook <b>trabalho_svhn.ipynb</b>, com a "
    f"implementação e todos os experimentos, incluindo as saídas e as curvas, o arquivo <b>melhor_modelo.pt</b>, "
    f"com os pesos do modelo final, a arquitetura, a média e o desvio da normalização e o raio da média sobre "
    f"translações, e o arquivo <b>{ARQ_RESULTADOS}</b>, com 203.856 linhas e uma predição (0 a 9) por imagem do "
    f"conjunto adicional. O arquivo <b>svhn_lib.py</b> vai junto por conveniência e é o mesmo código que a "
    f"primeira célula do notebook grava.",
]
