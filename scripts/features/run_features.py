"""
Genera el análisis de Feature Engineering: tablas (CSV) y figuras (PNG) en docs/features/.

Uso (desde la raíz del repositorio):
    python scripts/features/run_features.py
    python scripts/features/run_features.py --data-dir data --out-dir docs/features

No modifica los datos. No entrena el modelo del proyecto: los modelos de regresión logística de este
script son solo líneas base de referencia, ajustadas con train y medidas en dev.
Usa las mismas funciones que notebooks/03_feature_engineering.ipynb.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
sys.path.append(str(Path(__file__).resolve().parents[1] / "eda"))
import eda_utils as E  # noqa: E402
import feature_utils as F  # noqa: E402
import run_eda as R  # noqa: E402  (reutiliza el estilo gráfico y los helpers de guardado)

COLOR_GRUPO = {"estructural": "#2a78d6", "expresividad": "#eb6834", "conversacional": "#1baf7a", "fenomeno": "#e87ba4", "lexico": "#eda100"}


def fig_comparacion(res: pd.DataFrame, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.6))
    r = res.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(r))
    colores = ["#b8b7b0" if i == len(r) - 1 else ("#2a78d6" if "TF-IDF" in n else "#eda100") for i, n in enumerate(r["representacion"])]
    ax.barh(y, r["macro_f1_dev"], color=colores, height=0.62)
    ax.errorbar(r["macro_f1_dev"], y, xerr=[r["macro_f1_dev"] - r["macro_f1_ic_inf"], r["macro_f1_ic_sup"] - r["macro_f1_dev"]], fmt="none", ecolor=R.INK2, capsize=3)
    for yi, v, hi in zip(y, r["macro_f1_dev"], r["macro_f1_ic_sup"]):
        ax.text(hi + 0.012, yi, f"{v:.2f}", va="center", fontsize=9)
    ax.set_yticks(y, r["representacion"])
    ax.set_xlabel("Macro-F1 en dev (IC 95% por bootstrap)")
    ax.set_xlim(0, 0.75)
    ax.xaxis.grid(True, color=R.GRID)
    ax.set_axisbelow(True)
    ax.set_title("Qué representación aporta más señal (referencia, no es el experimento)")
    R._guardar(fig, ruta)


def fig_efecto(aso: pd.DataFrame, ruta: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6))
    lim = float(max(aso["efecto_etiqueta"].max(), aso["efecto_variedad"].max())) * 1.15
    ax.plot([0, lim], [0, lim], color=R.GRID, lw=1.5)
    for g, sub in aso.groupby("grupo"):
        ax.scatter(sub["efecto_variedad"], sub["efecto_etiqueta"], s=55, color=COLOR_GRUPO[g], label=g, zorder=3)
    for _, f in aso.iterrows():
        if max(f["efecto_etiqueta"], f["efecto_variedad"]) > 0.012:
            abajo = f["feature"] in ("n_exclamaciones", "n_intensificadores", "empieza_con_mencion")
            ax.annotate(f["feature"], (f["efecto_variedad"], f["efecto_etiqueta"]), xytext=(6, -12 if abajo else 5),
                        textcoords="offset points", fontsize=8.5, color=R.INK2)
    ax.text(lim * 0.98, lim * 0.93, "más ligada a la etiqueta", ha="right", fontsize=9, color=R.INK2)
    ax.text(lim * 0.98, lim * 0.03, "más ligada a la variedad", ha="right", fontsize=9, color=R.INK2)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Efecto sobre la variedad (epsilon², Kruskal-Wallis)")
    ax.set_ylabel("Efecto sobre la etiqueta (epsilon²)")
    ax.legend(loc="center right", bbox_to_anchor=(1, 0.6))
    ax.set_title("Cada característica: ¿delata el sentimiento o el país?")
    R._guardar(fig, ruta)


def fig_por_variedad(pv: pd.DataFrame, ruta: Path) -> None:
    t = pv.pivot(index="representacion", columns="variedad", values="macro_f1_dev")
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    im = ax.imshow(t.values, cmap=R.SEQ, vmin=0, vmax=0.7, aspect="auto")
    ax.set_xticks(range(t.shape[1]), t.columns)
    ax.set_yticks(range(t.shape[0]), t.index)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9, color="white" if v > 0.35 else R.INK)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Macro-F1 en dev por variedad (modelos entrenados con train de todas las variedades)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="Macro-F1")
    R._guardar(fig, ruta)


def fig_correlaciones(matriz: pd.DataFrame, ruta: Path) -> None:
    c = matriz.corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8.4, 7))
    im = ax.imshow(c.values, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(c)), c.columns, rotation=90, fontsize=8)
    ax.set_yticks(range(len(c)), c.columns, fontsize=8)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, shrink=0.75, label="Spearman")
    ax.set_title("Redundancia entre características")
    R._guardar(fig, ruta)


def fig_ablacion(ab: pd.DataFrame, ruta: Path) -> None:
    t = ab.iloc[1:].copy()
    t["tipo"] = t["configuracion"].str.split().str[0]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.2), sharex=True)
    for ax, tipo, titulo in ((axs[0], "Sin", "Quitar un grupo del modelo completo"), (axs[1], "Solo", "Usar un grupo solo")):
        d = t[t.tipo == tipo].iloc[::-1]
        y = np.arange(len(d))
        cols = ["#e34948" if hi < 0 else "#b8b7b0" for hi in d["ic_sup"]]
        ax.barh(y, d["cambio_vs_completo"], color=cols, height=0.6)
        ax.errorbar(d["cambio_vs_completo"], y, xerr=[d["cambio_vs_completo"] - d["ic_inf"], d["ic_sup"] - d["cambio_vs_completo"]], fmt="none", ecolor=R.INK2, capsize=3)
        ax.set_yticks(y, d["configuracion"].str.replace("Sin ", "").str.replace("Solo ", ""))
        ax.axvline(0, color=R.INK2, lw=0.8)
        ax.set_title(titulo)
        ax.set_xlabel("Cambio en Macro-F1 vs. modelo completo (IC 95%)")
        ax.xaxis.grid(True, color=R.GRID)
        ax.set_axisbelow(True)
    fig.suptitle("Ablación por grupo (rojo = caída distinguible de cero, gris = no se distingue)", x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    R._guardar(fig, ruta)


def fig_importancia(imp: pd.DataFrame, ruta: Path) -> None:
    d = imp.head(12).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.barh(d["feature"], d["caida_macro_f1"], xerr=d["desv"], color=[COLOR_GRUPO[g] for g in d["grupo"]], ecolor=R.INK2, capsize=2, height=0.62)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in COLOR_GRUPO.values()]
    ax.legend(handles, COLOR_GRUPO.keys(), loc="lower right")
    ax.set_xlabel("Caída del Macro-F1 al desordenar la característica (dev, media y desv. de 15 repeticiones)")
    ax.xaxis.grid(True, color=R.GRID)
    ax.set_axisbelow(True)
    ax.set_title("Importancia por permutación (modelo de 24 características)")
    R._guardar(fig, ruta)


def fig_estabilidad(est: pd.DataFrame, ruta: Path) -> None:
    t = est.set_index("feature")
    fig, ax = plt.subplots(figsize=(7.4, 7.2))
    im = ax.imshow(t.values, cmap=R.SEQ, vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(t.shape[1]), t.columns, rotation=30, ha="right")
    ax.set_yticks(range(t.shape[0]), t.index, fontsize=8.5)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            v = t.values[i, j]
            if v >= 0.5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8, color="white" if v > 55 else R.INK)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, shrink=0.7, label="% de tuits del dev donde cambia el valor")
    ax.set_title("Estabilidad de cada característica frente a las perturbaciones")
    R._guardar(fig, ruta)


def fig_robustez(rob: pd.DataFrame, ruta: Path) -> None:
    t = rob.pivot(index="perturbacion", columns="representacion", values="caida")
    fig, ax = plt.subplots(figsize=(9.2, 4))
    lim = float(np.abs(t.values).max())
    im = ax.imshow(t.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    etiquetas = {"Superficie + fenómenos + léxico (24 numéricas)": "24 numéricas", "TF-IDF de caracteres": "TF-IDF\ncaracteres", "TF-IDF de palabras": "TF-IDF\npalabras",
                 "TF-IDF palabras + caracteres": "TF-IDF\npal. + car.", "TF-IDF palabras + numéricas": "TF-IDF\npal. + num."}
    ax.set_xticks(range(t.shape[1]), [etiquetas.get(c, c) for c in t.columns], fontsize=9)
    ax.set_yticks(range(t.shape[0]), t.index)
    for i in range(t.shape[0]):
        for j in range(t.shape[1]):
            ax.text(j, i, f"{t.values[i, j]:+.3f}", ha="center", va="center", fontsize=9, color=R.INK)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    fig.colorbar(im, ax=ax, shrink=0.8, label="Cambio en Macro-F1")
    ax.set_title("Qué tanto cae el Macro-F1 al perturbar el texto (modelos entrenados con texto limpio)")
    R._guardar(fig, ruta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="docs/features")
    a = ap.parse_args()
    out = Path(a.out_dir)
    tablas, figs = out / "tablas", out / "figuras"
    tablas.mkdir(parents=True, exist_ok=True)

    corpus = E.cargar_corpus(a.data_dir)
    M = F.construir_matriz(corpus)
    print(f"Corpus: {len(corpus)} tuits · {M.shape[1]} características")

    resumen, por_var = F.comparar_representaciones(corpus)
    rep = F.Representaciones(corpus)
    y_dev = rep.dev["etiqueta"].values
    dif = F.diferencias_pareadas(y_dev, rep.predecir(), [("pal", "car"), ("pal", "pal+car"), ("pal", "pal+num"), ("car", "pal+car"), ("num", "pal"), ("num", "pal+num")])
    abl = F.ablacion_por_grupo(corpus)
    imp = F.importancia_permutacion(corpus)
    est = F.estabilidad_features(corpus)
    rob = F.robustez_modelos(corpus)
    aso = F.asociacion_univariada(M, corpus["etiqueta"], corpus["variedad"])
    salidas = {
        "catalogo_features": F.catalogo_features(),
        "estadisticos_features": M.describe().T.round(3).reset_index().rename(columns={"index": "feature"}),
        "medias_por_etiqueta": F.medias_por_etiqueta(M, corpus["etiqueta"]).reset_index().rename(columns={"index": "feature"}),
        "asociacion_univariada": aso,
        "correlaciones_altas": F.correlaciones_altas(M),
        "comparacion_representaciones": resumen,
        "macro_f1_por_variedad": por_var,
        "terminos_por_clase": F.terminos_por_clase(corpus),
        "prediccion_de_variedad": F.prediccion_de_variedad(corpus),
        "perfil_pistas": F.perfil_pistas(corpus, M),
        "diferencias_pareadas": dif,
        "ablacion_por_grupo": abl,
        "importancia_permutacion": imp,
        "estabilidad_features": est,
        "robustez_modelos": rob,
        "mapa_features_agentes": F.mapa_features_agentes(),
    }
    print("Tablas:")
    for nombre, t in salidas.items():
        t.to_csv(tablas / f"{nombre}.csv", index=False, encoding="utf-8")
        print(f"  tabla: {R._rel(tablas / (nombre + '.csv'))}")

    print("Figuras:")
    fig_comparacion(resumen, figs / "fig01_comparacion_representaciones.png")
    fig_efecto(aso, figs / "fig02_efecto_etiqueta_vs_variedad.png")
    fig_por_variedad(por_var, figs / "fig03_macro_f1_por_variedad.png")
    fig_correlaciones(M, figs / "fig04_correlaciones.png")
    fig_ablacion(abl, figs / "fig05_ablacion_por_grupo.png")
    fig_importancia(imp, figs / "fig06_importancia_permutacion.png")
    fig_estabilidad(est, figs / "fig07_estabilidad_features.png")
    fig_robustez(rob, figs / "fig08_robustez_modelos.png")
    print("Listo.")


if __name__ == "__main__":
    main()
