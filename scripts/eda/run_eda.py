"""
Genera el EDA completo del corpus: tablas (CSV) y figuras (PNG) en docs/eda/.
Incluye la segunda etapa (scripts/eda/eda_extra.py): IC, época vs. país, línea base, potencia y casi duplicados.

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
import eda_extra as X  # noqa: E402
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


RAIZ_REPO = Path(__file__).resolve().parents[2]


def _rel(ruta: Path) -> str:
    """Ruta relativa a la raíz del repo, para no mostrar rutas absolutas de la máquina."""
    try:
        return Path(ruta).resolve().relative_to(RAIZ_REPO).as_posix()
    except ValueError:
        return Path(ruta).name


def _tiene_anotaciones(ruta: Path) -> bool:
    """True si la muestra de validación ya fue anotada a mano (para no borrar ese trabajo)."""
    if not ruta.exists():
        return False
    try:
        m = pd.read_csv(ruta, encoding="utf-8", dtype=str, keep_default_na=False)
        return bool(m[X.COL_VALIDACION].str.strip().ne("").any())
    except Exception:
        return False


def _guardar(fig, ruta: Path) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(ruta, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  figura: {_rel(ruta)}")


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
    ax.set_title("Tuits por variedad y partición (sin partición de prueba)")
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


def fig_neu_por_mes(df: pd.DataFrame, ruta: Path) -> None:
    """NEU por país y por mes: si Perú sigue arriba en cada mes, no es un efecto de la época."""
    t = X.etiquetas_por_mes(df, n_min=100)
    fig, ax = plt.subplots(figsize=(8.6, 4.4))
    meses = sorted(t["mes"].unique())
    for var, g in t.groupby("variedad"):
        g = g.set_index("mes").reindex(meses)
        ax.plot(range(len(meses)), g["NEU"], marker="o", lw=2.4 if var == "PE" else 1.4,
                color=COLOR_VAR[var], label=var, alpha=1 if var == "PE" else 0.8)
    ax.set_xticks(range(len(meses)), meses)
    ax.set_ylabel("% de tuits NEU")
    ax.set_ylim(0, 70)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(ncol=5, loc="upper left")
    ax.set_title("Perú tiene más NEU que los demás en cada mes comparable (meses con 100+ tuits)")
    _guardar(fig, ruta)


def fig_panel_resumen(df: pd.DataFrame, corpus: pd.DataFrame, ruta: Path) -> None:
    """Una sola lámina con lo esencial del EDA, pensada para la sustentación."""
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    paises = sorted(df["variedad"].unique())

    # (a) etiquetas con IC
    ax = axs[0, 0]
    ic = X.ic_etiquetas(df)
    for k, et in enumerate(E.ETIQUETAS):
        g = ic[ic.etiqueta == et].set_index("variedad").reindex(paises)
        x = np.arange(len(paises)) + (k - 1) * 0.22
        ax.errorbar(x, g["pct"], yerr=[g["pct"] - g["ic_inf"], g["ic_sup"] - g["pct"]], fmt="o",
                    color=COLOR_ET[et], capsize=3, label=et)
    ax.set_xticks(np.arange(len(paises)), paises)
    ax.set_ylabel("% de tuits (IC 95%)")
    ax.set_ylim(0, 65)
    ax.yaxis.grid(True, color=GRID)
    ax.legend(ncol=3, loc="upper left")
    ax.set_title("Etiquetas por país: Perú es distinto")

    # (b) piso de desempeño
    ax = axs[0, 1]
    may = E.baseline_mayoritario(corpus).set_index("variedad").reindex(paises)
    tf = X.baseline_tfidf(corpus)
    tf = tf[tf.modelo == "TF-IDF conjunto"].set_index("variedad").reindex(paises)
    x = np.arange(len(paises))
    ax.bar(x - 0.19, may["macro_f1_dev"], 0.36, color="#b8b7b0", label="Siempre la clase mayoritaria")
    ax.bar(x + 0.19, tf["macro_f1_dev"], 0.36, color="#2a78d6", label="TF-IDF + regresión logística")
    ax.errorbar(x + 0.19, tf["macro_f1_dev"], yerr=[tf["macro_f1_dev"] - tf["macro_f1_ic_inf"], tf["macro_f1_ic_sup"] - tf["macro_f1_dev"]],
                fmt="none", ecolor=INK2, capsize=3)
    ax.set_xticks(x, paises)
    ax.set_ylabel("Macro-F1 en dev")
    ax.set_ylim(0, 0.8)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left")
    ax.set_title("Pisos de referencia a superar")

    # (c) concentración temporal
    ax = axs[1, 0]
    vp = X.ventana_pico(df).set_index("variedad").reindex(paises)
    ax.bar(x, vp["pct_tuits_en_ventana"], color=[COLOR_VAR[v] for v in paises])
    for xi, v in zip(x, vp["pct_tuits_en_ventana"]):
        ax.text(xi, v + 1.5, f"{v:.0f}%", ha="center", fontsize=9)
    ax.set_xticks(x, paises)
    ax.set_ylabel("% de tuits en 14 días")
    ax.set_ylim(0, 100)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.set_title("Concentración temporal (ventana más densa)")

    # (d) potencia
    ax = axs[1, 1]
    n = np.arange(300, 2701, 50)
    z = 1.96 + 0.8416
    for d, ls in ((0.10, ":"), (0.20, "-"), (0.30, "--")):
        ax.plot(n, 100 * z * np.sqrt(d / n), ls, color="#1d4f91", label=f"discrepan en {int(d*100)}% de tuits")
    ax.axvspan(390, 581, color="#eda100", alpha=0.18, lw=0)
    ax.text(486, 11.2, "dev por país", ha="center", fontsize=9, color=INK2)
    ax.axvline(2465, color=INK2, lw=0.8)
    ax.text(2440, 11.2, "dev conjunto", ha="right", fontsize=9, color=INK2)
    ax.axhline(2, color="#e34948", lw=1)
    ax.text(1300, 0.9, "mejora de 2 pp (objetivo del proyecto)", ha="left", fontsize=9, color="#e34948")
    ax.set_xlabel("Tuits evaluados")
    ax.set_ylabel("Diferencia mínima detectable (pp)")
    ax.set_ylim(0, 12.5)
    ax.yaxis.grid(True, color=GRID)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", bbox_to_anchor=(0.2, 0.9))
    ax.set_title("Potencia: qué diferencias se pueden detectar")

    fig.suptitle("EDA del corpus: lo esencial en una lámina", x=0.01, ha="left", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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

    casi_resumen, casi_pares = X.casi_duplicados(corpus)

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
        # Segunda etapa (eda_extra.py)
        "ic_etiquetas": X.ic_etiquetas(df),
        "ic_fenomenos": X.ic_fenomenos(df),
        "etiquetas_por_mes": X.etiquetas_por_mes(df),
        "comparacion_misma_ventana": X.comparacion_misma_ventana(df),
        "ventana_pico": X.ventana_pico(df),
        "baseline_tfidf": X.baseline_tfidf(corpus),
        "potencia_minima": X.potencia_minima(corpus),
        "casi_duplicados_resumen": casi_resumen,
        "casi_duplicados_pares": casi_pares,
    }
    for nombre, t in salidas.items():
        destino = tablas / f"{nombre}.csv"
        if nombre == "muestra_validar_heuristicas" and _tiene_anotaciones(destino):
            print(f"  tabla: {_rel(destino)} (ya tiene anotaciones a mano, no se sobrescribe)")
            continue
        t.to_csv(destino, index=False, encoding="utf-8")
        print(f"  tabla: {_rel(destino)}")

    print("Figuras:")
    figs = out / "figuras"
    fig_tamano(corpus, figs / "fig01_tamano.png")
    fig_etiquetas(corpus, figs / "fig02_etiquetas.png")
    fig_longitud(df, figs / "fig03_longitud.png")
    fig_rasgos(df, figs / "fig04_rasgos.png")
    fig_fenomenos(df, figs / "fig05_fenomenos.png")
    fig_temporal(df, figs / "fig06_temporal.png")
    fig_baseline(salidas["baseline_mayoritario"], figs / "fig07_baseline_mayoritario.png")
    fig_neu_por_mes(df, figs / "fig08_neu_por_mes.png")
    fig_panel_resumen(df, corpus, figs / "fig09_panel_resumen.png")
    print("Listo.")


if __name__ == "__main__":
    main()
