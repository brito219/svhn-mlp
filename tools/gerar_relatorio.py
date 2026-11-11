"""Gera relatorio.pdf a partir de resultados/*.json e do texto em tools/relatorio_texto.py.

Uso: .venv/bin/python tools/gerar_relatorio.py
"""
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import relatorio_texto as T  # noqa: E402

RES = ROOT / "resultados"
rows = [json.load(open(RES / f"E{i}.json"))["row"] for i in range(1, 9)]
best = json.load(open(RES / "melhor_modelo.json"))

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=15, spaceAfter=6, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12.5, spaceBefore=10, spaceAfter=4, keepWithNext=1)
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=10, leading=13.5, alignment=TA_JUSTIFY)
from reportlab.lib.enums import TA_LEFT
SMALL = ParagraphStyle("s", parent=P, fontSize=8.5, leading=11, alignment=TA_LEFT)
CAP = ParagraphStyle("cap", parent=P, fontSize=8.5, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
TITLE = ParagraphStyle("t", parent=ss["Title"], fontSize=18)

W = A4[0] - 4 * cm


def para(txt, st=P):
    return Paragraph(txt, st)


def bullets(items):
    return [para(f"• {it}") for it in items]


def fig(path, caption, width=W):
    from PIL import Image as PI
    w, h = PI.open(path).size
    return KeepTogether([Image(str(path), width=width, height=width * h / w), para(caption, CAP), Spacer(1, 6)])


def table(data, col_widths, header_bg="#dbe7f3", highlight_row=None):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(header_bg)),
             ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
             ("FONTSIZE", (0, 0), (-1, -1), 8.5),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("ALIGN", (2, 1), (-1, -1), "CENTER")]
    if highlight_row is not None:
        style.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor("#e6f4e6")))
    t.setStyle(TableStyle(style))
    return KeepTogether([t, Spacer(1, 8)])


pct = lambda v: f"{100 * v:.2f}%"
story = []

# ---------------- capa / cabeçalho ----------------
story += [para("Classificação de Dígitos no Dataset SVHN", TITLE),
          para(T.AUTORES, CAP), para(T.DISCIPLINA, CAP), Spacer(1, 10)]

# ---------------- 9.1 solução ----------------
story.append(para("1. Solução desenvolvida", H1))
story += [para(t) for t in T.SOLUCAO]
story.append(para("Protocolo de avaliação", H2))
story += bullets(T.PROTOCOLO)

mod = [["Modelo", "Arquitetura", "Parâmetros"]]
for e, arq in T.ARQUITETURAS:
    r = rows[e - 1]
    mod.append([para(f"E{e} — {r['nome']}", SMALL), para(arq, SMALL), f"{r['params']:,}".replace(",", ".")])
story += [para("Modelos utilizados", H2), table(mod, [4.2 * cm, 9.8 * cm, 3 * cm])]

hp = [["Exp.", "Otimizador", "lr", "Batch", "Épocas", "Weight decay", "Outros"]]
for r in rows:
    c = r["config"]
    outros = []
    if c.get("scheduler"):
        outros.append("lr cosseno")
    if c.get("augment_shift"):
        outros.append(f"translação ±{c['augment_shift']}px")
    outros.append("entrada [0,1]" if r["dados"] == "bruto" else "normalizada")
    hp.append([f"E{r['exp']}", c.get("optimizer", "adam").upper().replace("ADAMW", "AdamW").replace("ADAM", "Adam"),
               f"{c['lr']:g}", c["batch_size"], c["epochs"], f"{c.get('weight_decay', 0):g}", ", ".join(outros)])
story += [para("Hiperparâmetros mais importantes", H2), table(hp, [1.3 * cm, 2.3 * cm, 1.5 * cm, 1.4 * cm, 1.6 * cm, 2.4 * cm, 6.5 * cm])]

# ---------------- 9.2 experimentos ----------------
story.append(para("2. Experimentos", H1))
story.append(para(T.EXPERIMENTOS_INTRO))
tab = [["Exp.", "Modificação", "Acc. treino", "Acc. val.", "Acc. teste", "Δ val.", "Resultado"]]
prev = None
for r in rows:
    d = "—" if prev is None else f"{100 * (r['val_acc'] - prev):+.2f} pp"
    tab.append([f"E{r['exp']}", para(r["modificacao"], SMALL), pct(r["train_acc"]), pct(r["val_acc"]),
                pct(r["test_acc"]), d, para(T.VEREDITO[r["exp"]], SMALL)])
    prev = r["val_acc"]
story.append(table(tab, [1.1 * cm, 4.6 * cm, 1.9 * cm, 2.1 * cm, 1.8 * cm, 1.7 * cm, 3.8 * cm],
                   highlight_row=best["exp"]))
story.append(para("Acurácias na época de melhor validação (checkpoint). Δ val. = variação da acurácia de validação "
                  "em relação ao experimento anterior; linha verde = modelo final.", CAP))
story.append(Spacer(1, 6))
story += [para(t) for t in T.ANALISE_EXPERIMENTOS]

# ---------------- 9.3 curvas ----------------
story.append(PageBreak())
story.append(para("3. Curvas de treinamento", H1))
story.append(para(T.CURVAS_INTRO))
story.append(fig(RES / "comparacao_validacao.png", "Figura 1 — Acurácia e loss de validação por época, todos os experimentos."))
for e, txt in T.CURVAS:
    story.append(fig(RES / f"E{e}_curvas.png", f"Figura — E{e}: {rows[e - 1]['nome']} (treino x validação).", width=W * 0.92))
    story.append(para(txt))
    story.append(Spacer(1, 4))

# ---------------- 9.4 melhor solução ----------------
story.append(KeepTogether([para("4. Melhor solução", H1)] + [para(t) for t in T.MELHOR]))
mb = [["Item", "Valor"],
      ["Modelo", f"E{best['exp']} — {best['nome']}"],
      ["Arquitetura", para(dict(T.ARQUITETURAS)[best["exp"]], SMALL)],
      ["Hiperparâmetros", para(T.MELHOR_HP, SMALL)],
      ["Melhor época (validação)", f"{best['melhor_epoca']} de {best['epocas']}"],
      ["Acurácia de validação", pct(best["val_acc"])],
      ["Acurácia de teste", pct(best["test_acc"])],
      ["Pesos", "melhor_modelo.pt"]]
story.append(table(mb, [5 * cm, 12 * cm]))

# ---------------- 9.5 conclusão ----------------
story.append(para("5. Conclusão", H1))
story += [para(t) for t in T.CONCLUSAO]
story.append(para("Principais dificuldades", H2))
story += bullets(T.DIFICULDADES)
story.append(para("Reprodutibilidade e entregáveis", H2))
story += bullets(T.ENTREGAVEIS)


def rodape(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(ROOT / "relatorio.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="Relatório — SVHN").build(
    story, onFirstPage=rodape, onLaterPages=rodape)
print("relatorio.pdf gerado")
