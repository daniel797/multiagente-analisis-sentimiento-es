"""Genera notebooks/05_experimento_analisis.ipynb (análisis detallado del experimento, sin GPU).

Uso (desde la raíz):  python scripts/experimento/build_experimento_notebook.py
Luego ejecutar:       jupyter nbconvert --to notebook --execute --inplace notebooks/05_experimento_analisis.ipynb
"""
import nbformat as nbf

M = lambda s: nbf.v4.new_markdown_cell(s.strip())
C = lambda s: nbf.v4.new_code_cell(s.strip())
cells = []

cells.append(M("""
# 05 · Experimento: baseline y variantes de los datos, en detalle

Este cuaderno **no necesita GPU**. Lee las predicciones y tablas ya guardadas en el repositorio (el baseline clásico de `run_baseline_variantes.py` y el monoagente de `monoagente_salamandra.py`, corrido en el cuaderno 04) y muestra el detalle:

1. Qué cambia cada variante de datos y cuántos tuits toca.
2. Baseline clásico (TF-IDF + regresión logística), con la variante limpia y con cada componente.
3. Monoagente con Salamandra-7B-Instruct, con las mismas variantes.
4. Monoagente contra baseline: por variedad, qué clases predice y dónde se equivoca.
5. Verificación de que las tablas guardadas salen de las predicciones.

Métrica principal: Macro-F1 en *dev* (2.465 tuits), con IC 95% por bootstrap. Las variantes se comparan con **diferencia pareada** sobre los mismos tuits. No existe partición de prueba: todo se reporta en dev.
"""))

cells.append(C("""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

pd.set_option("display.width", 200, "display.max_colwidth", 90)
RAIZ = Path.cwd() if (Path.cwd() / "scripts").exists() else Path.cwd().parent
for sub in ("experimento", "eda", "features"):
    sys.path.append(str(RAIZ / "scripts" / sub))
import eda_utils as E
import variantes_datos as V
import evaluacion as EV

T_CLA = RAIZ / "docs/experimento/tablas"
T_LLM = RAIZ / "docs/experimento/llm/tablas"
AZUL, GRIS, ROJO = "#2a78d6", "#b8b7b0", "#c0392b"
CLASES = ["NEG", "NEU", "POS"]

corpus = E.cargar_corpus(RAIZ / "data")
dev = corpus[corpus.split == "dev"].reset_index(drop=True)
print("tuits en dev:", len(dev), "| variedades:", sorted(dev.variedad.unique()))
"""))

cells.append(M("""
## 1 · Qué cambia cada variante

Todas parten del texto **base** (el que ve el LLM: `@USUARIO`, `URL`, espacios colapsados). La variante principal es **limpio**; las demás cambian un solo componente, para saber cuál pesa. El conteo de abajo es sobre los 2.465 tuits de dev: cuántos textos cambian respecto de la base.
"""))
cells.append(C("""
base_txt = dev["texto"].map(V.base)
filas = []
for nombre in V.VARIANTES:
    t = dev["texto"].map(lambda x: V.aplicar(x, nombre))
    filas.append({"variante": nombre, "qué cambia": V.DESCRIPCION[nombre],
                  "tuits que cambian": int((t != base_txt).sum()),
                  "% de dev": round(100 * (t != base_txt).mean(), 1)})
tabla_var = pd.DataFrame(filas)
tabla_var
"""))
cells.append(M("""
Lectura: `minusculas` toca casi todos los textos (96%) y `sin_tildes` más de la mitad (58%). `sin_alargamientos` toca 5% y `sin_emojis` apenas 13 tuits (0,5%). Además, TF-IDF ya pasa todo a minúsculas y esos 13 emojis no cambian ninguna predicción, así que `minusculas` y `sin_emojis` no pueden mover nada en el baseline clásico (0 predicciones distintas en la tabla de la sección 2).
"""))

cells.append(C("""
ej = pd.read_csv(T_CLA / "ejemplos_variantes.csv")
for _, r in ej.iterrows():
    print(f"[{r.variedad}] base  : {r.base}\\n     limpio: {r.limpio}\\n")
"""))

cells.append(M("""
## 2 · Baseline clásico: TF-IDF + regresión logística

TF-IDF de n-gramas de palabras (1 a 2) con regresión logística balanceada, sin ajustar hiperparámetros. Se ajusta **solo con train** y se mide **solo en dev**. La variante se aplica igual a train y a dev.
"""))
cells.append(C("""
res_cla = pd.read_csv(T_CLA / "resultados_por_variante.csv")
dif_cla = pd.read_csv(T_CLA / "diferencias_pareadas_variantes.csv")
ancho = (res_cla[res_cla.variante.isin(["base", "limpio"])]
         .pivot(index="ambito", columns="variante", values="macro_f1")
         .reindex(["Todas", "CR", "ES", "MX", "PE", "UY"]))
d = dif_cla[dif_cla.variante == "limpio"].set_index("ambito")
ancho["diferencia"] = d["dif_macro_f1"]
ancho["IC 95%"] = d.apply(lambda r: f"{r.ic_inf:+.3f} a {r.ic_sup:+.3f}", axis=1)
ancho["cambian"] = d.apply(lambda r: f"{int(r.predicciones_distintas)} de {int(r.n_dev)}", axis=1)
ancho["conclusión"] = d["conclusion"]
ancho
"""))
cells.append(C("""
fig, ax = plt.subplots(figsize=(9, 4))
amb = list(ancho.index); x = np.arange(len(amb))
for off, var, col in ((-0.18, "base", GRIS), (0.18, "limpio", AZUL)):
    s = res_cla[res_cla.variante == var].set_index("ambito").loc[amb]
    ax.bar(x + off, s["macro_f1"], 0.34, color=col, label=var)
    ax.errorbar(x + off, s["macro_f1"], yerr=[s["macro_f1"] - s["ic_inf"], s["ic_sup"] - s["macro_f1"]], fmt="none", ecolor="k", capsize=3, lw=1)
ax.set_xticks(x, amb); ax.set_ylim(0, 0.75); ax.set_ylabel("Macro-F1 en dev")
ax.set_title("TF-IDF: base contra limpio, por variedad"); ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False); plt.show()
"""))

cells.append(M("""
### Cada componente por separado (baseline clásico)

Diferencia de Macro-F1 contra la base, en todo dev, con IC 95% pareado. Si el intervalo contiene 0, no se distingue de la base.
"""))
cells.append(C("""
c = dif_cla[dif_cla.ambito == "Todas"].copy()
c["IC 95%"] = c.apply(lambda r: f"{r.ic_inf:+.3f} a {r.ic_sup:+.3f}", axis=1)
c[["variante", "predicciones_distintas", "variante_acierta_y_base_no", "base_acierta_y_variante_no", "dif_macro_f1", "IC 95%", "p_bootstrap", "conclusion"]].reset_index(drop=True)
"""))
cells.append(C("""
c = c.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(9, 3.8)); y = np.arange(len(c))
ax.hlines(y, c.ic_inf, c.ic_sup, color=[AZUL if v == "limpio" else GRIS for v in c.variante], lw=2)
ax.scatter(c.dif_macro_f1, y, color=[AZUL if v == "limpio" else GRIS for v in c.variante], zorder=3)
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(y, c.variante)
ax.set_xlabel("Diferencia de Macro-F1 (variante menos base)"); ax.set_title("TF-IDF: ninguna variante se distingue de la base")
ax.spines[["top", "right"]].set_visible(False); plt.show()
"""))
cells.append(M("""
**Cómo leer las columnas.** *predicciones_distintas* es cuántos tuits cambian de predicción; *variante_acierta_y_base_no* y *base_acierta_y_variante_no* dicen en qué dirección. Cuando las dos cifras se parecen, el cambio sube y baja el resultado a la vez y el neto es casi cero.
"""))

cells.append(M("""
## 3 · Monoagente con Salamandra-7B-Instruct

`BSC-LT/salamandra-7b-instruct`, revisión `a3ed5452fafb3698a0423b1a18bfb5888f4e611b`, 4 bits, zero-shot, salida JSON, temperatura 0, hasta 2 reintentos. Prompt `B1-v1`, sin la variedad. Una salida que no se entiende cuenta como NEU.
"""))
cells.append(C("""
pred = pd.read_csv(RAIZ / "docs/experimento/llm/predicciones_llm.csv", dtype=str, keep_default_na=False)
print("filas:", len(pred), "| ids únicos por variante:", pred.groupby("variante").id.nunique().to_dict(),
      "| duplicados:", int(pred.duplicated(["id", "variante"]).sum()))
print("intentos:", pred.intentos.value_counts().to_dict())
print("salidas inválidas:"); display(pd.read_csv(T_LLM / "salidas_invalidas.csv"))
"""))
cells.append(C("""
res_llm = pd.read_csv(T_LLM / "resultados_llm_por_variante.csv")
dif_llm = pd.read_csv(T_LLM / "diferencias_llm_pareadas.csv")
a2 = (res_llm.pivot(index="ambito", columns="variante", values="macro_f1").reindex(["Todas", "CR", "ES", "MX", "PE", "UY"]))
d = dif_llm.set_index("ambito")
a2["diferencia"] = d["dif_macro_f1"]; a2["IC 95%"] = d.apply(lambda r: f"{r.ic_inf:+.3f} a {r.ic_sup:+.3f}", axis=1)
a2["cambian"] = d.apply(lambda r: f"{int(r.predicciones_distintas)} de {int(r.n_dev)}", axis=1)
a2["conclusión"] = d["conclusion"]
a2
"""))
cells.append(M("""
La variante limpia **tampoco** mueve al LLM: cambia 206 de 2.465 predicciones, 87 a favor y 84 en contra. Un LLM sí distingue mayúsculas, tildes y alargamientos, así que no era obvio de antemano; el resultado es que, con este prompt, no importa.
"""))

cells.append(M("""
## 4 · Monoagente contra baseline

Misma variante (`base`), mismos tuits. Diferencia = LLM menos TF-IDF.
"""))
cells.append(C("""
par = pd.read_csv(T_LLM / "llm_vs_clasico_pareado.csv")
par["conclusion"] = par["conclusion"].replace({"A mejor": "TF-IDF mejor", "B mejor": "LLM mejor"})
pb = par[par.variante.str.contains("base")].set_index("ambito").reindex(["Todas", "CR", "ES", "MX", "PE", "UY"])
fig, ax = plt.subplots(figsize=(9, 3.8)); y = np.arange(len(pb))[::-1]
col = [AZUL if a == "Todas" else GRIS for a in pb.index]
ax.hlines(y, pb.ic_inf, pb.ic_sup, color=col, lw=2.5); ax.scatter(pb.dif_macro_f1, y, color=col, zorder=3)
ax.axvline(0, color="k", lw=0.8); ax.set_yticks(y, pb.index)
ax.set_xlabel("Macro-F1: LLM menos TF-IDF  (← TF-IDF mejor | LLM mejor →)")
ax.set_title("El monoagente queda por debajo de TF-IDF"); ax.spines[["top", "right"]].set_visible(False); plt.show()
pb[["n_dev", "dif_macro_f1", "ic_inf", "ic_sup", "p_bootstrap", "conclusion"]]
"""))
cells.append(M("""
Con 5 variedades, una corrección de Bonferroni exige p < 0,01. La pasan ES y MX; CR (p = 0,012) queda en el límite; PE y UY no se distinguen.
"""))

cells.append(M("### Qué clases predice cada modelo"))
cells.append(C("""
dist = pd.read_csv(T_LLM / "distribucion_predicha.csv").set_index("clase")
dist = dist[["real", "clasico_base", "llm_base"]]
dist.columns = ["Real (dev)", "TF-IDF (base)", "LLM (base)"]
fig, ax = plt.subplots(figsize=(9, 3))
izq = np.zeros(3); colores = {"NEG": ROJO, "NEU": GRIS, "POS": AZUL}
for c_ in CLASES:
    v = dist.loc[c_].to_numpy()
    ax.barh(dist.columns, v, left=izq, color=colores[c_], label=c_, alpha=.85)
    for i, (l, w) in enumerate(zip(izq, v)): ax.text(l + w / 2, i, f"{w:.1f}%", ha="center", va="center", fontsize=9)
    izq += v
ax.invert_yaxis(); ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(.5, -.05))
ax.set_title("El LLM casi no predice NEG"); ax.spines[["top", "right"]].set_visible(False); plt.show()
dist
"""))
cells.append(C("""
f1 = pd.read_csv(T_LLM / "f1_por_clase.csv")
f1[f1.modelo.isin(["clasico_base", "llm_base"])].pivot(index="clase", columns="modelo", values=["precision", "recobrado", "f1"])
"""))
cells.append(C("""
mc = pd.read_csv(T_LLM / "matriz_confusion_llm.csv").set_index("real")
fig, ax = plt.subplots(figsize=(4.6, 3.8)); im = ax.imshow(mc.values, cmap="Blues")
ax.set_xticks(range(3), mc.columns); ax.set_yticks(range(3), mc.index)
ax.set_xlabel("Predicho (LLM, base)"); ax.set_ylabel("Real")
for i in range(3):
    for j in range(3):
        ax.text(j, i, mc.values[i, j], ha="center", va="center", color="white" if mc.values[i, j] > 400 else "black")
ax.set_title("Matriz de confusión del LLM"); plt.show()
print("NEG reales marcados como NEU:", mc.loc["NEG", "NEU"], "de", mc.loc["NEG"].sum())
"""))
cells.append(M("""
La fila NEG concentra el problema: de 960 tuits negativos, 748 salen como NEU. Cuando el LLM dice NEG casi siempre acierta (precisión 0,93), pero encuentra pocos (recobrado 0,14). En POS y NEU el LLM supera a TF-IDF en F1, así que la pérdida total viene de NEG.

**Hipótesis (no probada):** el prompt `B1-v1` empuja a NEU ante la duda. Perú, donde NEU es la clase mayoritaria (55,2%), es la variedad donde menos pierde.
"""))

cells.append(M("### Ejemplos de error: NEG real que el LLM marca NEU"))
cells.append(C("""
pb_ = pred[pred.variante == "base"].merge(dev[["id", "texto"]].assign(id=lambda d: d.id.astype(str)), on="id")
err = pb_[(pb_.etiqueta == "NEG") & (pb_.pred == "NEU")].sample(10, random_state=0)
err[["variedad", "texto", "salida_cruda"]].reset_index(drop=True)
"""))
cells.append(M("""
Estos ejemplos sirven para revisar a mano si el tuit es realmente negativo o si la etiqueta de TASS es discutible, antes de culpar solo al prompt.
"""))

cells.append(M("## 5 · Verificación: las tablas salen de las predicciones"))
cells.append(C("""
ver = pd.read_csv(T_LLM / "verificacion_tablas.csv")
print(f"{int(ver.coincide.sum())} de {len(ver)} valores de Macro-F1 recalculados coinciden con las tablas guardadas")
# recálculo independiente de un valor
p = pred[pred.variante == "base"].assign(pred_ok=lambda d: d.pred.where(d.pred.isin(CLASES), "NEU"))
print("Macro-F1 LLM base, todo dev:", round(EV.metricas(p.etiqueta, p.pred_ok, n_boot=200)["macro_f1"], 4))
ver
"""))

cells.append(M("""
## Conclusiones

- **Texto limpio:** no cambia el resultado ni con TF-IDF (+0,001) ni con Salamandra (+0,003). Ninguna variante de un solo componente se distingue de la base con TF-IDF.
- **Monoagente contra baseline:** 0,479 contra 0,556 de Macro-F1 (−0,076, IC 95% de −0,102 a −0,052). La causa visible es que casi no detecta NEG.
- **Lo que no se puede afirmar:** que el sesgo a NEU sea culpa del prompt, ni que otro prompt lo corrija; hay una sola corrida y un solo prompt.
- **Límites:** sin partición de prueba, sin GPU ni versiones de librerías registradas, y solo ES y MX resisten Bonferroni por variedad.
- **Siguiente experimento:** un prompt que corrija el sesgo a NEU, ajustado en desarrollo y reportado en un subconjunto reservado.
"""))

nb = nbf.v4.new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nbf.write(nb, "notebooks/05_experimento_analisis.ipynb")
print("escrito notebooks/05_experimento_analisis.ipynb")
