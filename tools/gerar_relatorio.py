"""Gera relatorio.pdf a partir de resultados/*.json e do texto em tools/relatorio_texto.py.

Uso: .venv/bin/python tools/gerar_relatorio.py
Todos os números das tabelas vêm dos JSONs gerados pelo notebook; o texto fica em relatorio_texto.py.
"""
import json
import sys
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))
import relatorio_texto as T  # noqa: E402

RES = ROOT / "resultados"
N_EXP = len(list(RES.glob("E*.json")))
rows = [json.load(open(RES / f"E{i}.json"))["row"] for i in range(1, N_EXP + 1)]
best = json.load(open(RES / "melhor_modelo.json"))
final = json.load(open(RES / "final.json"))

ss = getSampleStyleSheet()
H1 = ParagraphStyle("h1", parent=ss["Heading1"], fontSize=14.5, spaceBefore=8, spaceAfter=5, keepWithNext=1)
H2 = ParagraphStyle("h2", parent=ss["Heading2"], fontSize=12, spaceBefore=8, spaceAfter=4, keepWithNext=1)
P = ParagraphStyle("p", parent=ss["BodyText"], fontSize=10, leading=13.2, alignment=TA_JUSTIFY)
SMALL = ParagraphStyle("s", parent=P, fontSize=8.2, leading=10.4, alignment=TA_LEFT)
CAP = ParagraphStyle("cap", parent=P, fontSize=8.5, alignment=TA_CENTER, textColor=colors.HexColor("#444444"))
TITLE = ParagraphStyle("t", parent=ss["Title"], fontSize=18, spaceAfter=4)

W = A4[0] - 4 * cm
VERDE = "#e6f4e6"


def para(txt, st=P):
    return Paragraph(txt, st)


def bullets(items):
    return [para(f"• {it}") for it in items]


def fig(name, caption, width=W):
    from PIL import Image as PI
    path = RES / name
    w, h = PI.open(path).size
    return KeepTogether([Image(str(path), width=width, height=width * h / w), para(caption, CAP), Spacer(1, 5)])


def table(data, col_widths, highlight_row=None, center_from=2, font=8.2):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    style = [("GRID", (0, 0), (-1, -1), 0.4, colors.grey),
             ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbe7f3")),
             ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
             ("FONTSIZE", (0, 0), (-1, -1), font),
             ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
             ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4),
             ("ALIGN", (center_from, 1), (-1, -1), "CENTER")]
    if highlight_row is not None:
        style.append(("BACKGROUND", (0, highlight_row), (-1, highlight_row), colors.HexColor(VERDE)))
    t.setStyle(TableStyle(style))
    t.spaceAfter = 7
    return t


pct = lambda v: f"{100 * v:.2f}%".replace(".", ",")
num = lambda v, n=3: f"{v:.{n}f}".replace(".", ",")
milhar = lambda v: f"{v:,}".replace(",", ".")
story = []

# ---------------- cabeçalho ----------------
story += [para("Classificação de Dígitos no Dataset SVHN", TITLE),
          para(T.AUTORES, CAP), para(T.DISCIPLINA, CAP), Spacer(1, 8)]

# ---------------- 1. solução ----------------
story.append(para("1. Solução desenvolvida", H1))
story += [para(t) for t in T.SOLUCAO]
story.append(para("Protocolo de avaliação", H2))
story += bullets(T.PROTOCOLO)

story.append(para("Modelos utilizados", H2))
mod = [["Modelo", "Experimentos", "Arquitetura", "Parâmetros"]]
for nome, exps, e_ref, arq in T.ARQUITETURAS:
    mod.append([para(nome, SMALL), exps, para(arq, SMALL), milhar(rows[e_ref - 1]["params"])])
story.append(table(mod, [3.0 * cm, 2.3 * cm, 9.5 * cm, 2.2 * cm], center_from=3))

story.append(para("Hiperparâmetros mais importantes", H2))
hp = [["Exp.", "Otimizador", "lr", "Batch", "Épocas", "Outros"]]
OPT = {"adam": "Adam", "adamw": "AdamW", "sgd": "SGD + momentum"}
for r in rows:
    c = r["config"]
    outros = [{"bruto": "entrada em [0,1]", "norm": "entrada normalizada",
               "norm_img": "entrada normalizada por canal e por imagem"}[r["dados"]]]
    if c.get("weight_decay"):
        outros.append(f"L2 (weight decay) {c['weight_decay']:g}")
    if c.get("scheduler"):
        outros.append({"cosine": "lr com decaimento cosseno", "onecycle": "lr one-cycle"}[c["scheduler"]])
    if c.get("augment_shift"):
        outros.append(f"translação ±{c['augment_shift']} px")
    if c.get("augment_rot"):
        outros.append(f"rotação ±{c['augment_rot']:g}° e escala ±{100 * c['augment_scale']:g}%")
    hp.append([f"E{r['exp']}", OPT[c.get("optimizer", "adam")], f"{c['lr']:g}", c["batch_size"], c["epochs"],
               para(", ".join(outros), SMALL)])
story.append(table(hp, [1.2 * cm, 3.0 * cm, 1.6 * cm, 1.4 * cm, 1.5 * cm, 8.3 * cm], center_from=1))
story += [para(t) for t in T.HIPER_NOTA]

# ---------------- 2. experimentos ----------------
story.append(para("2. Experimentos", H1))
story.append(para(T.EXPERIMENTOS_INTRO))
tab = [["Exp.", "Modificação", "Loss\ntreino", "Loss\nval.", "Acc.\ntreino", "Acc.\nval.", "Acc.\nteste",
        "Δ val.", "Resultado"]]
for r in rows:
    pai = r.get("pai")
    d = "—" if pai is None else f"{100 * (r['val_acc'] - rows[pai - 1]['val_acc']):+.2f} pp".replace(".", ",")
    tab.append([f"E{r['exp']}", para(r["modificacao"].replace("0.0001", "1e-4").replace("0.001", "1e-3"), SMALL), num(r["train_loss"]), num(r["val_loss"]),
                pct(r["train_acc"]), pct(r["val_acc"]), pct(r["test_acc"]), d, para(T.VEREDITO[r["exp"]], SMALL)])
story.append(table(tab, [1.0 * cm, 3.7 * cm, 1.2 * cm, 1.2 * cm, 1.5 * cm, 1.5 * cm, 1.5 * cm, 1.7 * cm, 3.7 * cm],
                   highlight_row=best["exp"]))
story.append(para(T.TABELA_LEGENDA, CAP))
story.append(Spacer(1, 4))
story += [para(t) for t in T.ANALISE_EXPERIMENTOS]

# ---------------- 3. curvas ----------------
story.append(para("3. Curvas de treinamento", H1))
story.append(para(T.CURVAS_INTRO))
n_fig = 0
for arquivo, legenda, txt, largura in T.CURVAS:
    n_fig += 1
    story.append(fig(arquivo, f"Figura {n_fig} — {legenda}", width=W * largura))
    if txt:
        story.append(para(txt))
        story.append(Spacer(1, 4))

# ---------------- 4. melhor solução ----------------
story.append(para("4. Melhor solução", H1))
story += [para(t) for t in T.MELHOR]
mb = [["Item", "Valor"],
      ["Modelo", f"E{best['exp']} — {best['nome']}"],
      ["Arquitetura", para(T.MELHOR_ARQ, SMALL)],
      ["Hiperparâmetros", para(T.MELHOR_HP, SMALL)],
      [f"Acurácia de validação (E{best['exp']}, seleção)", f"{pct(best['val_acc'])} (época {best['melhor_epoca']} de {best['epocas']})"],
      [f"Acurácia de teste (E{best['exp']}, treinado com 90%)", pct(best["test_acc"])],
      [para(f"Com média sobre {(2 * best['tta_raio'] + 1) ** 2} translações (E{best['exp']})", SMALL),
       f"validação {pct(best['val_acc_tta'])} · teste {pct(best['test_acc_tta'])}"],
      [para("<b>Acurácia de teste do modelo final</b> (retreinado com 100% do treino)", SMALL), para(f"<b>{pct(final['test_acc'])}</b> (sem a média sobre translações: {pct(final['test_acc_sem_tta'])})", SMALL)],
      ["Pesos do modelo final", "melhor_modelo.pt"]]
story.append(table(mb, [6.2 * cm, 10.8 * cm], center_from=9))
n_fig += 1
story.append(fig("final_confusao.png", f"Figura {n_fig} — Matriz de confusão do modelo final no conjunto de teste.",
                 width=W * 0.47))
story += [para(t) for t in T.MELHOR_ERROS]

# ---------------- 5. conclusão ----------------
story.append(para("5. Conclusão", H1))
story += [para(t) for t in T.CONCLUSAO]
story.append(para("Principais dificuldades", H2))
story += bullets(T.DIFICULDADES)
story.append(para("Entregáveis", H2))
story += bullets(T.ENTREGAVEIS)


def rodape(canvas, doc):
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
    canvas.restoreState()


SimpleDocTemplate(str(ROOT / "relatorio.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                  topMargin=1.7 * cm, bottomMargin=1.8 * cm, title="Relatório — Classificação de Dígitos no SVHN",
                  author=T.AUTORES).build(story, onFirstPage=rodape, onLaterPages=rodape)
print("relatorio.pdf gerado")
