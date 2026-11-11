"""Texto do relatório (separado do layout em gerar_relatorio.py). Números conferidos com resultados/*.json."""

AUTORES = "NOME1 · NOME2 · NOME3"  # <- trocar pelos nomes do grupo
DISCIPLINA = "Inteligência Artificial — Trabalho Final"

SOLUCAO = [
    "Implementamos em PyTorch um classificador linear, uma MLP (rede totalmente conectada) e, como experimento "
    "adicional, uma rede convolucional (CNN). Todo o código reaproveitável — divisão dos dados, modelos, laço de "
    "treino, gráficos e predição — está em <b>svhn_lib.py</b> (coberto por 19 testes automatizados em "
    "<i>tests/</i>), e o notebook <b>trabalho_svhn.ipynb</b> executa todos os experimentos de ponta a ponta. "
    "Partimos do código das aulas (notebook MLP e classificador linear do MNIST) e o adaptamos ao SVHN: "
    "entrada de 3×32×32 = 3072 valores em vez de 784, e registro de loss e acurácia de treino e validação a cada época.",
    "Todos os modelos são treinados com cross-entropy. Ao fim de cada treino, o modelo volta aos pesos da época "
    "de <b>maior acurácia de validação</b> (checkpoint). Assim a validação escolhe tanto o experimento quanto o "
    "ponto de parada, sem nunca consultar o teste.",
]

PROTOCOLO = [
    "Dados: <i>torchvision.datasets.SVHN</i>, splits <i>train</i> (73.257 imagens) e <i>test</i> (26.032). "
    "Como o servidor de Stanford estava instável, os arquivos foram lidos do espelho oficial no Hugging Face "
    "(<i>ufldl-stanford/svhn</i>). Verificamos byte a byte que as imagens são as mesmas e estão na mesma ordem.",
    "Divisão do <i>train</i>: <b>90% treino (65.931) / 10% validação (7.326)</b>, por permutação com semente fixa 42.",
    "Normalização por canal com média e desvio calculados <b>apenas no conjunto de treino</b>; os mesmos valores são "
    "aplicados à validação, ao teste e ao conjunto adicional.",
    "Reprodutibilidade: semente 42 para Python, NumPy e PyTorch, e também para a ordem dos mini-batches e a "
    "augmentation (<i>torch.Generator</i>), além de cuDNN determinístico. Rodar o notebook de novo reproduz os números.",
    "O conjunto de teste serve só para relatar o desempenho. O conjunto adicional do professor foi usado uma "
    "única vez, depois de escolhido o modelo final.",
]

ARQUITETURAS = [
    (1, "Linear: 3072 → 10 (logits = xW + b)"),
    (2, "MLP: 3072 → 500 → 500 → 10, ReLU (igual à da aula)"),
    (3, "MLP: 3072 → 500 → 500 → 10, ReLU"),
    (4, "MLP: 3072 → 1024 → 512 → 256 → 10, ReLU"),
    (5, "MLP: 3072 → 1024 → 512 → 256 → 10, ReLU + Dropout 0,3"),
    (6, "MLP: 3072 → 1024 → 512 → 256 → 10, Linear → BatchNorm → ReLU → Dropout 0,3"),
    (7, "Mesma do E6"),
    (8, "CNN: [Conv3×3(32)-BN-ReLU]×2 → MaxPool → [Conv3×3(64)-BN-ReLU]×2 → MaxPool → "
        "[Conv3×3(128)-BN-ReLU]×2 → MaxPool → Dropout 0,3 → FC 2048→256 → ReLU → Dropout 0,3 → FC 256→10"),
]

EXPERIMENTOS_INTRO = (
    "A partir do baseline, cada experimento muda <b>um fator</b> por vez, para que a melhora ou piora possa ser "
    "atribuída a ele. As exceções são o E7, que mexe em dois fatores ligados (tamanho do batch e agenda da taxa "
    "de aprendizado), e o E8, que troca a arquitetura."
)

VEREDITO = {
    1: "Referência: underfitting (modelo linear)",
    2: "Underfitting; otimização ruim sem normalização",
    3: "Melhora grande (+24,7 pp); passa a ter overfitting",
    4: "Melhora pequena; overfitting continua",
    5: "Piora: acaba o overfitting, mas o treino fica lento",
    6: "Melhora (+5,3 pp); BN acelera e estabiliza",
    7: "Melhora (+0,8 pp); melhor MLP",
    8: "Melhora grande (+8,1 pp); modelo final",
}

ANALISE_EXPERIMENTOS = [
    "<b>Classificador linear (E1).</b> Ficou em 28,4% de validação. A loss parou em torno de 2,16, pouco abaixo "
    "de ln(10) ≈ 2,30 (chute uniforme). Um modelo linear sobre pixels não separa os dígitos do SVHN: a cor, o "
    "brilho, a posição e os dígitos vizinhos variam demais. É um caso claro de underfitting por falta de capacidade.",
    "<b>MLP baseline (E2).</b> Com o código da aula (entrada em [0, 1]), a MLP chegou a 58,4%, com treino e "
    "validação colados (57,2% × 58,4%). Também é underfitting. Os pixels são todos positivos e as entradas têm "
    "escalas parecidas, o que deixa a otimização com o Adam lenta e mal condicionada.",
    "<b>Normalização (E3)</b> foi a mudança isolada de maior impacto entre as MLPs: +24,7 pp, chegando a 83,1%. "
    "Com ela aparece o overfitting: a loss de treino cai até 0,29, enquanto a de validação atinge o mínimo na "
    "época 9 (0,66) e sobe até 0,83. A diferença entre as acurácias de treino e de validação chega a 7,9 pp.",
    "<b>Rede maior (E4):</b> dobrar a capacidade (3,8 M parâmetros) rendeu só +0,4 pp, com o mesmo padrão de "
    "overfitting. Sem regularização, aumentar a rede não compensa. <b>Dropout (E5)</b> eliminou o overfitting "
    "(a acurácia de treino ficou <i>abaixo</i> da de validação e a loss de validação caiu de forma contínua), "
    "mas a acurácia piorou 1,6 pp: com o ruído do dropout, a rede ainda estava aprendendo quando acabaram as 25 "
    "épocas, ou seja, ficou subtreinada.",
    "<b>BatchNorm (E6)</b> resolveu isso: o treino ficou mais rápido e estável, chegando a 87,2% (+5,3 pp) sem "
    "overfitting (loss de validação 0,43). <b>Batch 256 com taxa de aprendizado em decaimento cosseno (E7)</b> "
    "deu o último ganho das MLPs: 88,0% na validação e 85,9% no teste, com as curvas mais suaves.",
    "<b>CNN (E8).</b> O enunciado deixa a arquitetura livre e cita o número de filtros entre os experimentos "
    "possíveis. A CNN chegou a <b>96,1% na validação e 96,2% no teste</b>: +8,1 pp sobre a melhor MLP, com "
    "4,7 vezes menos parâmetros (815 mil × 3,8 milhões). Ela compartilha os mesmos filtros por toda a imagem e "
    "explora a vizinhança dos pixels, enquanto a MLP precisa aprender cada posição separadamente. A translação "
    "aleatória de ±2 px (augmentation) reforça a tolerância a pequenos deslocamentos do dígito.",
    "Em todas as MLPs, a acurácia de teste ficou de 2 a 4 pp abaixo da de validação (por exemplo, 85,9% × 88,0% "
    "no E7). A validação vem da mesma distribuição do treino, enquanto o <i>test</i> do SVHN é um pouco mais "
    "difícil. Na CNN essa diferença desapareceu (96,2% × 96,1%), outro sinal de que ela generaliza melhor.",
]

CURVAS_INTRO = (
    "As curvas mostram loss e acurácia de treino e de validação a cada época. A acurácia de treino é a média "
    "durante a época, com dropout e augmentation ativos; por isso pode ficar abaixo da de validação, que é medida "
    "com o modelo em modo de avaliação. A linha pontilhada marca a época escolhida pela validação."
)

CURVAS = [
    (2, "<b>E2 — underfitting.</b> Treino e validação andam juntos e estabilizam em ~58%, com loss alta (~1,3). "
        "O modelo não consegue nem ajustar bem o próprio treino."),
    (3, "<b>E3 — overfitting.</b> Depois da época ~9, a loss de validação sobe (0,66 → 0,83) enquanto a de treino "
        "continua caindo, e a acurácia de validação estaciona em ~82–83% com o treino passando de 90%. O "
        "checkpoint pela melhor validação limita o dano, mas o padrão justifica regularizar (E5–E6)."),
    (5, "<b>E5 — regularização forte.</b> O dropout remove o overfitting (validação acima do treino), mas as duas "
        "curvas ainda sobem no fim: faltou treino. Há pequenas oscilações da validação entre épocas."),
    (7, "<b>E7 — melhor MLP.</b> As curvas de treino e validação ficam próximas (diferença < 1 pp), a loss de "
        "validação cai até ~0,39 e o decaimento cosseno suaviza o final do treino. Não há sinal de overfitting "
        "nem de instabilidade."),
    (8, "<b>E8 — CNN (modelo final).</b> A validação começa em ~90% já na primeira época e sobe até 96,1%. A "
        "loss de validação estabiliza em ~0,155 nas últimas épocas; a de treino só a ultrapassa a partir da "
        "época 15, com diferença pequena (96,7% × 96,1% de acurácia). Há um início de saturação, mas não "
        "overfitting relevante."),
]

MELHOR = [
    "O modelo final é a <b>CNN do E8</b>, escolhida por ter a maior acurácia de validação entre os oito "
    "experimentos (96,1%, contra 88,0% da melhor MLP). A acurácia de teste (96,2%) confirmou a escolha sem ter "
    "participado dela. Esse mesmo modelo, com os pesos da época 19, gerou o arquivo de predições do conjunto "
    "adicional (203.856 imagens lidas em ordem numérica do nome do arquivo, uma predição inteira de 0 a 9 por "
    "linha). As imagens passaram pela mesma normalização do treino.",
]

MELHOR_HP = ("AdamW, lr inicial 1e-3 com decaimento cosseno, weight decay 5e-4, batch 128, 20 épocas, dropout 0,3, "
             "augmentation por translação aleatória de ±2 px (sem flip horizontal, que altera dígitos), "
             "entrada normalizada por canal.")

CONCLUSAO = [
    "A abordagem que funcionou melhor foi a <b>rede convolucional</b> (96,1% na validação e 96,2% no teste), "
    "seguida da MLP com normalização, BatchNorm, dropout e taxa de aprendizado cosseno (88,0% / 85,9%). O "
    "classificador linear (28,4%) serviu de referência e mostra que o problema não é linearmente separável "
    "sobre os pixels.",
    "As modificações de maior impacto foram, em ordem: (1) a <b>normalização da entrada</b> (+24,7 pp na MLP); "
    "(2) a <b>troca para CNN</b> (+8,1 pp sobre a melhor MLP); (3) a <b>BatchNorm</b> (+5,3 pp). Dropout sozinho "
    "piorou a acurácia no limite de 25 épocas, mas foi essencial para eliminar o overfitting quando combinado "
    "com BatchNorm. Aumentar a MLP sem regularizar quase não ajudou.",
]

DIFICULDADES = [
    "O download pelo torchvision falhava (o servidor de Stanford cortava a conexão). Usamos o espelho oficial "
    "no Hugging Face, depois de verificar que as imagens são idênticas e estão na mesma ordem.",
    "O treino foi feito em CPU, sem GPU. Cada MLP levou de 4 a 12 minutos e a CNN, cerca de 53 minutos. Isso "
    "limitou o número de épocas e impediu uma busca mais ampla de hiperparâmetros.",
    "Com dropout, foi preciso interpretar as curvas com cuidado: a acurácia de treino abaixo da de validação não "
    "indica erro, e sim que o dropout só atua durante o treino.",
    "Garantir que a ordem das predições seguisse a ordem dos nomes dos arquivos: a leitura usa ordenação "
    "numérica, que é testada.",
]

ENTREGAVEIS = [
    "<b>trabalho_svhn.ipynb</b> + <b>svhn_lib.py</b>: implementação e todos os experimentos (com saídas).",
    "<b>melhor_modelo.pt</b>: pesos da CNN final, junto com a média e o desvio da normalização.",
    "<b>NOME1_NOME2_RESULTS.txt</b>: 203.856 linhas, uma predição (0–9) por imagem do conjunto adicional.",
    "<b>relatorio.pdf</b>: este relatório. Histórico completo de cada experimento em <i>resultados/</i>.",
]
