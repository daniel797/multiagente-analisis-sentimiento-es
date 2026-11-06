"""
Optimización de hiperparámetros del baseline clásico (no del LLM).

Se busca la mejor configuración con validación cruzada de 5 pliegues SOLO sobre train. Dev no se toca durante
la búsqueda: se usa una única vez al final, para medir la configuración elegida contra la de referencia
(TF-IDF de palabras 1 a 2, C = 1, que es la del experimento baseline).

Tres familias de modelos:
  palabras   TF-IDF de palabras  (n-gramas, min_df) + regresión logística (C, class_weight)
  caracteres TF-IDF de caracteres, digramas y trigramas entre otros   + regresión logística (C, class_weight)
  numerico   las 24 variables estandarizadas + regresión logística (C, class_weight)

Uso (desde la raíz del repositorio):
    python scripts/experimento/optimizar_hiperparametros.py
Salida: docs/hiperparametros/tablas/*.csv
"""
from __future__ import annotations

import argparse
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

AQUI = Path(__file__).resolve().parent
for sub in ("experimento", "features", "eda"):
    sys.path.append(str(AQUI.parent / sub))
import eda_utils as E  # noqa: E402
import evaluacion as EV  # noqa: E402
import feature_utils as F  # noqa: E402
import variantes_datos as V  # noqa: E402

VALORES_C = [0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0]
PESOS = ["balanced", None]
N_PLIEGUES = 5
SEMILLA = 0

# Espacio de búsqueda: nombre de la familia -> lista de configuraciones del vectorizador
ESPACIO = {
    "palabras": [{"ngram_range": ng, "min_df": md} for ng in [(1, 1), (1, 2), (1, 3)] for md in [1, 2, 3]],
    "caracteres": [{"ngram_range": ng, "min_df": 3} for ng in [(2, 3), (3, 4), (2, 5)]],
    "numerico": [{}],
}


def cargar(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame]:
    corpus = E.cargar_corpus(data_dir)
    train = corpus[corpus.split == "train"].reset_index(drop=True)
    dev = corpus[corpus.split == "dev"].reset_index(drop=True)
    return train, dev


def _lr(c: float, peso) -> LogisticRegression:
    return LogisticRegression(C=c, class_weight=peso, max_iter=3000, random_state=0)


def _vectorizador(familia: str, cfg: dict) -> TfidfVectorizer:
    if familia == "palabras":
        return TfidfVectorizer(lowercase=True, sublinear_tf=True, **cfg)
    if familia == "caracteres":
        return TfidfVectorizer(lowercase=True, analyzer="char_wb", sublinear_tf=True, max_features=60000, **cfg)
    raise ValueError(familia)


def _textos(df: pd.DataFrame) -> pd.Series:
    """El tuit como lo ve el LLM (variante base)."""
    return df["texto"].map(lambda t: V.aplicar(t, "base"))


def pliegues(train: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray]]:
    """Pliegues estratificados por etiqueta y variedad a la vez."""
    estratos = train["etiqueta"] + "_" + train["variedad"]
    return list(StratifiedKFold(N_PLIEGUES, shuffle=True, random_state=SEMILLA).split(train, estratos))


def busqueda_cv(train: pd.DataFrame, familias=("palabras", "caracteres", "numerico"), verbose: bool = True) -> pd.DataFrame:
    """
    Una fila por configuración: familia, parámetros, Macro-F1 medio de los 5 pliegues y su desviación.
    El vectorizador y el escalador se ajustan DENTRO de cada pliegue, solo con sus datos de ajuste.
    """
    y = train["etiqueta"].to_numpy()
    textos = _textos(train)
    num = F.construir_matriz(train)[F._numericas(F.GRUPOS_NUM)] if "numerico" in familias else None
    folds = pliegues(train)
    filas = []
    for fam in familias:
        t0 = time.time()
        for cfg in ESPACIO[fam]:
            # se transforma una vez por pliegue y se reutiliza para todos los (C, peso)
            mats = []
            for tr, va in folds:
                if fam == "numerico":
                    sc = StandardScaler().fit(num.iloc[tr])
                    mats.append((sc.transform(num.iloc[tr]), sc.transform(num.iloc[va])))
                else:
                    vec = _vectorizador(fam, cfg).fit(textos.iloc[tr])
                    mats.append((vec.transform(textos.iloc[tr]), vec.transform(textos.iloc[va])))
            for c, peso in itertools.product(VALORES_C, PESOS):
                f1s = []
                for (tr, va), (Xtr, Xva) in zip(folds, mats):
                    p = _lr(c, peso).fit(Xtr, y[tr]).predict(Xva)
                    f1s.append(f1_score(y[va], p, average="macro"))
                filas.append({
                    "familia": fam, "ngram_range": str(cfg.get("ngram_range", "")), "min_df": cfg.get("min_df", ""),
                    "C": c, "class_weight": peso or "ninguno",
                    "macro_f1_cv": float(np.mean(f1s)), "desv_cv": float(np.std(f1s, ddof=1)),
                })
        if verbose:
            print(f"{fam}: {sum(1 for f in filas if f['familia'] == fam)} configuraciones en {time.time() - t0:.0f} s")
    return pd.DataFrame(filas).sort_values("macro_f1_cv", ascending=False).reset_index(drop=True)


def mejores_por_familia(res: pd.DataFrame) -> pd.DataFrame:
    return res.sort_values("macro_f1_cv", ascending=False).groupby("familia", sort=False).head(1).reset_index(drop=True)


def ajustar_y_predecir(train: pd.DataFrame, dev: pd.DataFrame, fila: pd.Series) -> np.ndarray:
    """Reajusta con TODO train la configuración de `fila` y predice dev."""
    peso = None if fila["class_weight"] == "ninguno" else fila["class_weight"]
    if fila["familia"] == "numerico":
        cols = F._numericas(F.GRUPOS_NUM)
        Mtr, Mdv = F.construir_matriz(train)[cols], F.construir_matriz(dev)[cols]
        sc = StandardScaler().fit(Mtr)
        return _lr(fila["C"], peso).fit(sc.transform(Mtr), train["etiqueta"]).predict(sc.transform(Mdv))
    cfg = {"ngram_range": eval(fila["ngram_range"]), "min_df": int(fila["min_df"])}
    vec = _vectorizador(fila["familia"], cfg).fit(_textos(train))
    return _lr(fila["C"], peso).fit(vec.transform(_textos(train)), train["etiqueta"]).predict(vec.transform(_textos(dev)))


def prediccion_referencia(train: pd.DataFrame, dev: pd.DataFrame) -> np.ndarray:
    """Configuración del experimento baseline: TF-IDF palabras (1,2), min_df=2, C=1, balanceado."""
    ref = pd.Series({"familia": "palabras", "ngram_range": "(1, 2)", "min_df": 2, "C": 1.0, "class_weight": "balanced"})
    return ajustar_y_predecir(train, dev, ref)


def evaluar_en_dev(train: pd.DataFrame, dev: pd.DataFrame, res: pd.DataFrame, n_boot: int = 2000):
    """
    Una sola medición en dev: la mejor configuración de cada familia y la de referencia.
    Devuelve (resultados, diferencias_contra_referencia, predicciones).
    """
    y = dev["etiqueta"].to_numpy()
    preds = {"referencia": prediccion_referencia(train, dev)}
    mejores = mejores_por_familia(res)
    for _, f in mejores.iterrows():
        preds[f"mejor_{f['familia']}"] = ajustar_y_predecir(train, dev, f)
    filas, difs = [], []
    for nombre, p in preds.items():
        for amb, m in EV.ambitos(dev["variedad"]).items():
            filas.append({"modelo": nombre, "ambito": amb, **EV.metricas(y[m], p[m], n_boot)})
        if nombre != "referencia":
            difs += [{**d, "variante": nombre} for d in EV.comparar_con_base(y, preds["referencia"], p, dev["variedad"], nombre, n_boot)]
    return pd.DataFrame(filas).round(4), pd.DataFrame(difs).round(4), preds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="docs/hiperparametros")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out_dir) / "tablas"
    out.mkdir(parents=True, exist_ok=True)

    train, dev = cargar(a.data_dir)
    print(f"train={len(train)} dev={len(dev)}")
    res = busqueda_cv(train)
    res.round(4).to_csv(out / "busqueda_cv.csv", index=False, encoding="utf-8")
    mejores_por_familia(res).round(4).to_csv(out / "mejor_por_familia.csv", index=False, encoding="utf-8")
    resultados, difs, _ = evaluar_en_dev(train, dev, res, a.n_boot)
    resultados.to_csv(out / "resultados_dev.csv", index=False, encoding="utf-8")
    difs.to_csv(out / "diferencias_contra_referencia.csv", index=False, encoding="utf-8")
    print(mejores_por_familia(res).round(4).to_string())
    print(resultados[resultados.ambito == "Todas"].to_string())


if __name__ == "__main__":
    main()
