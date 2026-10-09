"""
Genera el EDA completo del corpus: tablas (CSV) y figuras (PNG) en docs/eda/.

Uso (desde la raíz del repositorio):
    python scripts/eda/run_eda.py
    python scripts/eda/run_eda.py --data-dir data --out-dir docs/eda

No modifica los datos. No entrena ningún modelo. Todas las cifras salen de
scripts/eda/eda_utils.py, las mismas funciones que usa notebooks/02_eda.ipynb.
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

sys.path.append(str(Path(__file__).resolve().parent))
import eda_utils as E  # noqa: E402

# --- Estilo: paleta categórica validada (azul, naranja, aqua, amarillo, magenta) ---
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
COLOR_VAR = {"CR": "#2a78d6", "ES": "#eb6834", "MX": "#1baf7a", "PE": "#eda100", "UY": "#e87ba4"}
COLOR_ET = {"NEG": "#e34948", "NEU": "#b8b7b0", "POS": "#2a78d6"}  # divergente rojo-gris-azul
SEQ = LinearSegmentedColormap.from_list("seq", ["#eef4fc", "#1d4f91"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": GRID, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "legend.frameon": False,
})


def _guardar(fig, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: {ruta}")


def fig_tamano(corpus: pd.DataFrame, ruta: Path) -> None:
    piv = corpus.groupby(["variedad", "split"]).size().unstack()[["train", "dev"]]
    x = np.arange(len(piv))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (split, col, nombre) in enumerate([("train", "#2a78d6", "train"), ("dev", "#9ec3f0", "dev")]):
        barras = ax.bar(x + (i - 0.5) * 0.38, piv[split], 0.36, color=col, edgecolor=SURFACE, linewidth=1.5, label=nombre)
        for b, v in zip(barras, piv[split]):
            ax.text(b.get_x() + b.get_width() / 2, v + 12, f"{v:,}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, piv.index)
    ax.set_ylabel("Tuits")
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("Tuits por variedad y partición")
    ax.legend(ncol=2, loc="upper right")
    _guardar(fig, ruta)


def fig_etiquetas(corpus: pd.DataFrame, ruta: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, split in zip(axes, E.SPLITS):
        sub = corpus[corpus.split == split]
        pct = pd.crosstab(sub["variedad"], sub["etiqueta"], normalize="index")[E.ETIQUETAS] * 100
        pct = pct.loc[::-1]
        izq = np.zeros(len(pct))
        for et in E.ETIQUETAS:
            barras = ax.barh(pct.index, pct[et], left=izq, color=COLOR_ET[et], edgecolor=SURFACE, linewidth=2, label=et, height=0.7)
            for b, v in zip(barras, pct[et]):
                ax.text(b.get_x() + b.get_width() / 2, b.get_y() + b.get_height() / 2, f"{v:.0f}%", ha="center", va="center",
                        fontsize=9, fontweight="bold", color=INK if et == "NEU" else "white")
            izq += pct[et].to_numpy()
        ax.set_title(split)
        ax.set_xlim(0, 100)
        ax.set_xlabel("% de tuits")
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.04))
    fig.suptitle("Perú es la única variedad con mayoría NEU; en las demás domina NEG", y=1.12, x=0.01, ha="left",
                 fontsize=11, fontweight="bold")
    _guardar(fig, ruta)


def fig_longitud(df: pd.DataFrame, ruta: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    for ax, col, orden, colores, titulo in [
        (axes[0], "variedad", sorted(df["variedad"].unique()), COLOR_VAR, "Por variedad"),
        (axes[1], "etiqueta", E.ETIQUETAS, COLOR_ET, "Por etiqueta"),
    ]:
        datos = [df.loc[df[col] == o, "longitud_car"] for o in orden]
        bp = ax.boxplot(datos, tick_labels=orden, patch_artist=True, widths=0.55, medianprops={"color": INK, "linewidth": 1.6},
                        whiskerprops={"color": INK2}, capprops={"color": INK2}, flierprops={"markersize": 2, "markeredgecolor": INK2})
        for patch, o in zip(bp["boxes"], orden):
            patch.set_facecolor(colores[o])
            patch.set_edgecolor(SURFACE)
        for i, d in enumerate(datos, 1):
            ax.text(i, d.max() + 3, f"{d.median():.0f}", ha="center", fontsize=9, color=INK)
        ax.set_title(titulo)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Caracteres por tuit (número = mediana)")
    fig.suptitle("Los tuits son cortos: máximo 146 caracteres, mediana ~92", y=1.04, x=0.01, ha="left", fontsize=11, fontweight="bold")
    _guardar(fig, ruta)


def fig_rasgos(df: pd.DataFrame, ruta: Path) -> None:
    tv, te = E.tabla_rasgos(df, "variedad"), E.tabla_rasgos(df, "etiqueta").loc[E.ETIQUETAS]
    vmax = max(tv.to_numpy().max(), te.to_numpy().max())
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.6), gridspec_kw={"height_ratios": [5, 3]})
    for ax, tabla in zip(axes, (tv, te)):
        ax.imshow(tabla.to_numpy(), cmap=SEQ, vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(tabla.shape[1]), tabla.columns, rotation=0, fontsize=8.5)
        ax.set_yticks(range(tabla.shape[0]), tabla.index)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        for i in range(tabla.shape[0]):
            for j in range(tabla.shape[1]):
                v = tabla.iat[i, j]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=9, color="white" if v > vmax * 0.68 else INK)
    axes[0].set_title("% de tuits que presentan cada rasgo, por variedad")
    axes[1].set_title("…y por etiqueta (la exclamación distingue POS; la pregunta, NEU)")
    fig.tight_layout()
    _guardar(fig, ruta)


def fig_fenomenos(df: pd.DataFrame, ruta: Path) -> None:
    t = E.tabla_fenomenos(df)
    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (col, nombre) in zip(axes.ravel(), E.FENOMENOS.items()):
        pct = t[f"{col}_pct"].loc[::-1]
        n = t[col].loc[::-1]
        ax.barh(pct.index, pct, color=[COLOR_VAR[v] for v in pct.index], edgecolor=SURFACE, linewidth=2, height=0.65)
        for i, (v, c) in enumerate(zip(pct, n)):
            ax.text(v + pct.max() * 0.02, i, f"{v:.1f}%  (n={c})", va="center", fontsize=9, color=INK)
        ax.set_xlim(0, pct.max() * 1.45)
        ax.set_title(nombre)
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0)
        ax.xaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
    fig.suptitle("Fenómenos lingüísticos según heurísticas simples (train + dev). Son estimaciones aproximadas, no anotaciones",
                 y=1.01, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _guardar(fig, ruta)


def fig_temporal(df: pd.DataFrame, ruta: Path) -> None:
    semanal = df.assign(semana=df["fecha"].dt.to_period("W").dt.start_time).groupby(["semana", "variedad"]).size().unstack(fill_value=0)
    vars_ = list(semanal.columns)
    fig, axes = plt.subplots(len(vars_), 1, figsize=(10, 7.2), sharex=True)
    for ax, v in zip(axes, vars_):
        ax.fill_between(semanal.index, semanal[v], color=COLOR_VAR[v], alpha=0.25, linewidth=0)
        ax.plot(semanal.index, semanal[v], color=COLOR_VAR[v], linewidth=2)
        pct = semanal[v].max() / semanal[v].sum() * 100
        ax.text(0.01, 0.93, f"pico: {semanal[v].max()} tuits en una semana ({pct:.0f}% del total del país)",
                transform=ax.transAxes, fontsize=8.5, color=INK, va="top")
        ax.set_ylabel(v, rotation=0, labelpad=18, va="center", fontweight="bold", color=INK)
        ax.set_ylim(0, semanal[v].max() * 1.35)
        ax.yaxis.grid(True, color=GRID)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", labelsize=8)
    fig.suptitle("Tuits por semana (cada panel con su propia escala): ES y MX se concentran en una sola ráfaga",
                 y=0.995, x=0.01, ha="left", fontsize=11, fontweight="bold")
    fig.tight_layout()
    _guardar(fig, ruta)


def fig_baseline(base: pd.DataFrame, ruta: Path) -> None:
    x = np.arange(len(base))
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    for i, (col, color, nombre) in enumerate([("accuracy_dev", "#9ec3f0", "Accuracy"), ("macro_f1_dev", "#1d4f91", "Macro-F1")]):
        barras = ax.bar(x + (i - 0.5) * 0.38, base[col], 0.36, color=color, edgecolor=SURFACE, linewidth=1.5, label=nombre)
        for b, v in zip(barras, base[col]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, base["variedad"])
    ax.set_ylim(0, 0.7)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(ncol=2, loc="upper left")
    ax.set_title("Predecir siempre la clase mayoritaria: la Accuracy engaña, el Macro-F1 no")
    _guardar(fig, ruta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="docs/eda")
    a = ap.parse_args()
    out = Path(a.out_dir)
    tablas = out / "tablas"
    tablas.mkdir(parents=True, exist_ok=True)

    corpus = E.cargar_corpus(a.data_dir)
    df = E.agregar_fecha(E.agregar_fenomenos(E.agregar_rasgos(corpus)))
    print(f"Corpus cargado: {len(df)} tuits, variedades {sorted(df['variedad'].unique())}")

    print("Tablas:")
    salidas = {
        "resumen_por_variedad": E.resumen_por_variedad(df),
        "calidad_datos": E.reporte_calidad(corpus),
        "distribucion_etiquetas": E.tabla_etiquetas(corpus),
        "asociacion_etiquetas": E.asociacion_etiqueta_variedad(corpus),
        "baseline_mayoritario": E.baseline_mayoritario(corpus),
        "rasgos_por_variedad": E.tabla_rasgos(df, "variedad").reset_index(),
        "rasgos_por_etiqueta": E.tabla_rasgos(df, "etiqueta").reset_index(),
        "fenomenos_train_dev": E.tabla_fenomenos(df).reset_index(),
        "fenomenos_solo_dev": E.tabla_fenomenos(df, "dev").reset_index(),
        "rango_fechas": E.tabla_fechas(df).reset_index(),
        "concentracion_temporal": E.concentracion_temporal(df),
        "top_terminos_por_etiqueta": E.top_terminos_por_etiqueta(df),
        "vocabulario_distintivo": E.vocabulario_distintivo(df),
        "muestra_validar_heuristicas": E.muestra_para_validar(df),
    }
    for nombre, t in salidas.items():
        t.to_csv(tablas / f"{nombre}.csv", index=False, encoding="utf-8")
        print(f"  tabla: {tablas / (nombre + '.csv')}")

    print("Figuras:")
    figs = out / "figuras"
    fig_tamano(corpus, figs / "fig01_tamano.png")
    fig_etiquetas(corpus, figs / "fig02_etiquetas.png")
    fig_longitud(df, figs / "fig03_longitud.png")
    fig_rasgos(df, figs / "fig04_rasgos.png")
    fig_fenomenos(df, figs / "fig05_fenomenos.png")
    fig_temporal(df, figs / "fig06_temporal.png")
    fig_baseline(salidas["baseline_mayoritario"], figs / "fig07_baseline_mayoritario.png")
    print("Listo.")


if __name__ == "__main__":
    main()
