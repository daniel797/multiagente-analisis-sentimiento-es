"""
Experimento de baseline clásico con variantes de los datos.

Compara, con el mismo modelo y los mismos tuits de dev, la línea base (el tuit como lo ve el LLM)
contra la variante `limpio` (minúsculas, sin tildes, sin alargamientos, sin emojis) y contra cada
componente por separado. El modelo es TF-IDF de palabras (1 a 2) + regresión logística, ajustado SOLO
con train y medido SOLO en dev. No se ajusta ningún hiperparámetro.

Esto NO es el experimento del proyecto (que usa un LLM sin entrenar). Es el piso clásico, y fija el
protocolo de variantes que después se repite con Salamandra-7B-Instruct (monoagente_salamandra.py).

Uso (desde la raíz del repositorio):
    python scripts/experimento/run_baseline_variantes.py
    python scripts/experimento/run_baseline_variantes.py --data-dir data --out-dir docs/experimento

Salida: docs/experimento/tablas/*.csv y docs/experimento/figuras/*.png
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
import regex as re  # noqa: E402

AQUI = Path(__file__).resolve().parent
sys.path.append(str(AQUI))
sys.path.append(str(AQUI.parents[0] / "features"))
sys.path.append(str(AQUI.parents[0] / "eda"))
import eda_utils as E  # noqa: E402
import evaluacion as EV  # noqa: E402
import feature_utils as F  # noqa: E402
import run_eda as R  # noqa: E402
import variantes_datos as V  # noqa: E402

GRIS = "#b8b7b0"
AZUL = "#2a78d6"


def ajustar_y_predecir(train: pd.DataFrame, dev: pd.DataFrame, variante: str) -> np.ndarray:
    """TF-IDF de palabras + regresión logística, ajustado solo con train."""
    x_tr = train["texto"].map(lambda t: V.aplicar(t, variante))
    x_dv = dev["texto"].map(lambda t: V.aplicar(t, variante))
    vec = F._tfidf_palabras().fit(x_tr)
    modelo = F._lr().fit(vec.transform(x_tr), train["etiqueta"])
    return modelo.predict(vec.transform(x_dv))


def ejemplos(dev: pd.DataFrame, n: int = 5) -> pd.DataFrame:
    """Tuits de dev donde las variantes cambian algo visible (alargamientos y tildes), con @USUARIO ya aplicado."""
    b = dev["texto"].map(V.base)
    cambia = b.map(lambda t: len(t) <= 110 and bool(re.search(r"(\p{L})\1{2,}", t)) and bool(re.search(r"[áéíóúÁÉÍÓÚ]", t)))
    cand = dev[cambia].assign(base=b[cambia]).sort_values("id")
    cand = cand.sample(min(n, len(cand)), random_state=0).sort_values("id")
    return pd.DataFrame({"variedad": cand["variedad"], "base": cand["base"], "limpio": cand["texto"].map(V.limpio)})


def fig_resultados(res: pd.DataFrame, ruta: Path) -> None:
    amb = list(res["ambito"].unique())
    x = np.arange(len(amb))
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    for off, var, col, lab in ((-0.18, "base", GRIS, "Base (tuit como lo ve el LLM)"), (0.18, "limpio", AZUL, "Limpio (minúsculas, sin tildes, sin alargamientos, sin emojis)")):
        d = res[res.variante == var].set_index("ambito").loc[amb]
        ax.bar(x + off, d["macro_f1"], 0.34, color=col, label=lab)
        ax.errorbar(x + off, d["macro_f1"], yerr=[d["macro_f1"] - d["ic_inf"], d["ic_sup"] - d["macro_f1"]], fmt="none", ecolor=R.INK, elinewidth=1, capsize=3)
        for xi, v in zip(x + off, d["macro_f1"]):
            ax.text(xi, 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=8.5, color="white" if var == "limpio" else R.INK)
    ax.set_xticks(x, amb)
    ax.set_ylim(0, 0.75)
    ax.set_ylabel("Macro-F1 en dev")
    ax.set_title("TF-IDF + regresión logística: el texto limpio no se distingue de la base")
    ax.legend(loc="upper center", ncol=1, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.1))
    ax.spines[["top", "right"]].set_visible(False)
    R._guardar(fig, ruta)


def fig_diferencias(dif: pd.DataFrame, ruta: Path) -> None:
    d = dif[dif.ambito == "Todas"].reset_index(drop=True)
    d = d.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9, 3.9))
    y = np.arange(len(d))
    cols = [AZUL if v == "limpio" else GRIS for v in d["variante"]]
    ax.hlines(y, d["ic_inf"], d["ic_sup"], color=cols, lw=2)
    ax.scatter(d["dif_macro_f1"], y, color=cols, zorder=3, s=36)
    ax.axvline(0, color=R.INK, lw=0.8)
    ax.set_yticks(y, [V.DESCRIPCION[v] for v in d["variante"]], fontsize=9)
    ax.set_xlabel("Diferencia de Macro-F1 (variante menos base), con IC 95% pareado")
    ax.set_title("Ninguna variante se distingue de la base")
    ax.spines[["top", "right"]].set_visible(False)
    R._guardar(fig, ruta)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="docs/experimento")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    out = Path(a.out_dir)
    (out / "tablas").mkdir(parents=True, exist_ok=True)
    (out / "figuras").mkdir(parents=True, exist_ok=True)

    corpus = E.cargar_corpus(a.data_dir)
    train, dev = corpus[corpus.split == "train"], corpus[corpus.split == "dev"].reset_index(drop=True)
    y = dev["etiqueta"].to_numpy()
    print(f"train={len(train)} dev={len(dev)} variantes={list(V.VARIANTES)}")

    preds = {v: ajustar_y_predecir(train, dev, v) for v in V.VARIANTES}
    amb = EV.ambitos(dev["variedad"])

    filas = []
    for v, p in preds.items():
        for nombre, m in amb.items():
            filas.append({"variante": v, "ambito": nombre, **EV.metricas(y[m], p[m], a.n_boot)})
    res = pd.DataFrame(filas).round(4)
    res.to_csv(out / "tablas" / "resultados_por_variante.csv", index=False, encoding="utf-8")

    filas = []
    for v in V.VARIANTES:
        if v != "base":
            filas += EV.comparar_con_base(y, preds["base"], preds[v], dev["variedad"], v, n_boot=a.n_boot)
    dif = pd.DataFrame(filas).round(4)
    dif.to_csv(out / "tablas" / "diferencias_pareadas_variantes.csv", index=False, encoding="utf-8")

    pd.DataFrame({"id": dev["id"], "variedad": dev["variedad"], "etiqueta": y, **{f"pred_{v}": p for v, p in preds.items()}}).to_csv(
        out / "tablas" / "predicciones_dev.csv", index=False, encoding="utf-8")
    ejemplos(dev).to_csv(out / "tablas" / "ejemplos_variantes.csv", index=False, encoding="utf-8")

    fig_resultados(res[res.variante.isin(["base", "limpio"])], out / "figuras" / "fig01_base_vs_limpio_por_variedad.png")
    fig_diferencias(dif, out / "figuras" / "fig02_diferencias_por_variante.png")

    print(res[res.ambito == "Todas"][["variante", "macro_f1", "ic_inf", "ic_sup"]].to_string(index=False))
    print(dif[dif.ambito == "Todas"][["variante", "predicciones_distintas", "variante_acierta_y_base_no", "base_acierta_y_variante_no", "dif_macro_f1", "ic_inf", "ic_sup", "conclusion"]].to_string(index=False))
    print("Listo.")


if __name__ == "__main__":
    main()
