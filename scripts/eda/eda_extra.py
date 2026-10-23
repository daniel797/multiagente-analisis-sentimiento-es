"""
Análisis adicionales del EDA (segunda etapa).

Complementa a eda_utils.py con lo que hace los hallazgos más defendibles:

- Intervalos de confianza para etiquetas y fenómenos.
- Separar el efecto de la época del efecto del país.
- Una línea base real (TF-IDF + regresión logística), solo como piso de referencia.
- Potencia estadística: qué diferencias se pueden detectar con estos tamaños.
- Textos casi duplicados (no solo idénticos), dentro y entre train/dev.
- Precisión de las heurísticas de fenómenos, una vez anotada la muestra a mano.

Nada de esto forma parte del experimento principal (monoagente vs. multiagente):
el clasificador TF-IDF solo se usa aquí para tener un piso más exigente que "la clase mayoritaria".
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, norm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

import eda_utils as E

ETIQUETAS = E.ETIQUETAS
_RE_MENCION = re.compile(r"@\w+")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")


# ----------------------------------------------------------------------------
# Intervalos de confianza
# ----------------------------------------------------------------------------
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Intervalo de Wilson para una proporción (mejor que el normal con n pequeño)."""
    if n == 0:
        return (np.nan, np.nan)
    p = k / n
    den = 1 + z**2 / n
    centro = (p + z**2 / (2 * n)) / den
    mitad = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / den
    return (centro - mitad, centro + mitad)


def ic_etiquetas(df: pd.DataFrame) -> pd.DataFrame:
    """Proporción de cada etiqueta por variedad (train+dev) con IC 95%."""
    filas = []
    for var, g in df.groupby("variedad"):
        n = len(g)
        for et in ETIQUETAS:
            k = int((g["etiqueta"] == et).sum())
            lo, hi = wilson(k, n)
            filas.append({"variedad": var, "etiqueta": et, "n": n, "pct": 100 * k / n, "ic_inf": 100 * lo, "ic_sup": 100 * hi})
    return pd.DataFrame(filas)


def ic_fenomenos(df: pd.DataFrame) -> pd.DataFrame:
    """Porcentaje de cada fenómeno por variedad (train+dev) con IC 95%."""
    filas = []
    for var, g in df.groupby("variedad"):
        n = len(g)
        for fen in E.FENOMENOS:
            k = int(g[fen].sum())
            lo, hi = wilson(k, n)
            filas.append({"variedad": var, "fenomeno": fen, "casos": k, "n": n, "pct": 100 * k / n, "ic_inf": 100 * lo, "ic_sup": 100 * hi})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# Época vs. país
# ----------------------------------------------------------------------------
def etiquetas_por_mes(df: pd.DataFrame, n_min: int = 50) -> pd.DataFrame:
    """Distribución de etiquetas por variedad y mes (solo celdas con al menos n_min tuits)."""
    d = df.assign(mes=df["fecha"].dt.to_period("M").astype(str))
    conteo = d.groupby(["variedad", "mes", "etiqueta"]).size().unstack(fill_value=0).reindex(columns=ETIQUETAS, fill_value=0)
    n = conteo.sum(axis=1)
    pct = conteo.div(n, axis=0) * 100
    out = pct.round(1).assign(n=n).reset_index()
    return out[out["n"] >= n_min].reset_index(drop=True)


def comparacion_misma_ventana(df: pd.DataFrame, n_min: int = 100) -> pd.DataFrame:
    """
    Compara etiquetas entre países DENTRO del mismo mes, así la época queda fija.
    Si Perú sigue siendo distinto cuando todos los países se miran en el mismo mes,
    la diferencia no se explica solo por la fecha.
    """
    d = df.assign(mes=df["fecha"].dt.to_period("M").astype(str))
    filas = []
    for mes, g in d.groupby("mes"):
        conteo = g.groupby(["variedad", "etiqueta"]).size().unstack(fill_value=0).reindex(columns=ETIQUETAS, fill_value=0)
        conteo = conteo[conteo.sum(axis=1) >= n_min]
        if len(conteo) < 2:
            continue
        chi2, p, _, _ = chi2_contingency(conteo.values)
        v = float(np.sqrt(chi2 / (conteo.values.sum() * (min(conteo.shape) - 1))))
        pct = conteo.div(conteo.sum(axis=1), axis=0) * 100
        for var in conteo.index:
            filas.append({
                "mes": mes, "variedad": var, "n": int(conteo.loc[var].sum()),
                "NEG_pct": pct.loc[var, "NEG"], "NEU_pct": pct.loc[var, "NEU"], "POS_pct": pct.loc[var, "POS"],
                "p_entre_paises_del_mes": p, "v_cramer_del_mes": v,
            })
    return pd.DataFrame(filas).round(4)


def ventana_pico(df: pd.DataFrame, dias: int = 14) -> pd.DataFrame:
    """
    Para cada variedad: ventana de `dias` días con más tuits, qué fracción concentra,
    y si la distribución de etiquetas dentro de la ventana difiere del resto (chi-cuadrado).
    """
    filas = []
    for var, g in df.groupby("variedad"):
        g = g.dropna(subset=["fecha"]).copy()
        g["dia"] = g["fecha"].dt.normalize()
        por_dia = g.groupby("dia").size()
        rango = pd.date_range(por_dia.index.min(), por_dia.index.max(), freq="D")
        serie = por_dia.reindex(rango, fill_value=0)
        ventana = serie.rolling(dias).sum()
        fin = ventana.idxmax()
        ini = fin - pd.Timedelta(days=dias - 1)
        dentro = g["dia"].between(ini, fin)
        tabla = pd.crosstab(dentro, g["etiqueta"]).reindex(columns=ETIQUETAS, fill_value=0)
        p = chi2_contingency(tabla.values)[1] if tabla.shape[0] == 2 else np.nan
        filas.append({
            "variedad": var, "ventana_desde": ini.date(), "ventana_hasta": fin.date(),
            "pct_tuits_en_ventana": 100 * dentro.mean(),
            "NEG_pct_dentro": 100 * (g.loc[dentro, "etiqueta"] == "NEG").mean(),
            "NEG_pct_fuera": 100 * (g.loc[~dentro, "etiqueta"] == "NEG").mean() if (~dentro).any() else np.nan,
            "p_etiquetas_dentro_vs_fuera": p,
        })
    return pd.DataFrame(filas).round(4)


# ----------------------------------------------------------------------------
# Línea base real (solo como piso de referencia)
# ----------------------------------------------------------------------------
def _preparar(texto: str) -> str:
    return _RE_URL.sub(" URL ", _RE_MENCION.sub(" USUARIO ", texto))


def _modelo() -> tuple[TfidfVectorizer, LogisticRegression]:
    vec = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", random_state=0)
    return vec, clf


def _ic_macro_f1(y, pred, n_boot: int = 500, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        vals.append(f1_score(y[i], pred[i], labels=ETIQUETAS, average="macro", zero_division=0))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def baseline_tfidf(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Piso de referencia más exigente que la clase mayoritaria: TF-IDF + regresión logística.
    Se entrena SOLO con el train y se mide SOLO en el dev, de dos formas:
    un modelo por variedad y un modelo conjunto con todas las variedades.
    Es una referencia del EDA, no parte del experimento (donde no se entrena nada).
    """
    t = corpus.assign(_x=corpus["texto"].map(_preparar))
    tr_all, dv_all = t[t.split == "train"], t[t.split == "dev"]

    vec, clf = _modelo()
    clf.fit(vec.fit_transform(tr_all["_x"]), tr_all["etiqueta"])
    pred_conjunto = pd.Series(clf.predict(vec.transform(dv_all["_x"])), index=dv_all.index)

    filas = []
    for var in sorted(t["variedad"].unique()):
        tr, dv = tr_all[tr_all.variedad == var], dv_all[dv_all.variedad == var]
        vec, clf = _modelo()
        clf.fit(vec.fit_transform(tr["_x"]), tr["etiqueta"])
        pred_pais = clf.predict(vec.transform(dv["_x"]))
        for nombre, pred in (("TF-IDF por variedad", pred_pais), ("TF-IDF conjunto", pred_conjunto[dv.index].values)):
            lo, hi = _ic_macro_f1(dv["etiqueta"], pred)
            filas.append({
                "variedad": var, "modelo": nombre, "n_dev": len(dv),
                "accuracy_dev": accuracy_score(dv["etiqueta"], pred),
                "macro_f1_dev": f1_score(dv["etiqueta"], pred, labels=ETIQUETAS, average="macro", zero_division=0),
                "macro_f1_ic_inf": lo, "macro_f1_ic_sup": hi,
            })
    return pd.DataFrame(filas).round(4)


# ----------------------------------------------------------------------------
# Potencia estadística
# ----------------------------------------------------------------------------
def potencia_minima(corpus: pd.DataFrame, discordancias=(0.10, 0.20, 0.30), alfa: float = 0.05, potencia: float = 0.80) -> pd.DataFrame:
    """
    Con un diseño pareado (los dos sistemas clasifican los mismos tuits), la diferencia mínima de
    Accuracy que se puede detectar es  (z_alfa/2 + z_potencia) * sqrt(d / n),  donde d es la
    proporción de tuits en los que los dos sistemas discrepan. d no se conoce todavía, por eso
    se muestran varios valores. Es una aproximación en Accuracy, no en Macro-F1.
    """
    z = norm.ppf(1 - alfa / 2) + norm.ppf(potencia)
    n_dev = corpus[corpus.split == "dev"].groupby("variedad").size()
    filas = []
    for var, n in list(n_dev.items()) + [("TODAS (conjunto)", int(n_dev.sum()))]:
        for d in discordancias:
            filas.append({
                "variedad": var, "n_dev": int(n), "discordancia_supuesta": d,
                "diferencia_minima_detectable_pp": 100 * z * np.sqrt(d / n),
                "n_necesario_para_2pp": int(np.ceil((z**2) * d / 0.02**2)),
            })
    return pd.DataFrame(filas).round(2)


# ----------------------------------------------------------------------------
# Textos casi duplicados
# ----------------------------------------------------------------------------
def _limpiar(texto: str) -> str:
    t = _RE_URL.sub(" ", _RE_MENCION.sub(" ", texto.lower()))
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", t)).strip()


def casi_duplicados(corpus: pd.DataFrame, umbral: float = 0.80, bloque: int = 1000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Pares de tuits con similitud coseno >= umbral (TF-IDF de palabras y bigramas).
    Devuelve (resumen, pares). Interesan sobre todo los pares train-dev: son fuga entre particiones.
    """
    c = corpus.reset_index(drop=True)
    X = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit_transform(c["texto"].map(_limpiar))
    pares = []
    for ini in range(0, X.shape[0], bloque):
        S = (X[ini:ini + bloque] @ X.T).toarray()
        filas = np.arange(ini, min(ini + bloque, X.shape[0]))
        for k, i in enumerate(filas):
            S[k, : i + 1] = 0  # solo la mitad superior, sin la diagonal
        ii, jj = np.nonzero(S >= umbral)
        pares += [(filas[a], b, S[a, b]) for a, b in zip(ii, jj)]
    p = pd.DataFrame(pares, columns=["i", "j", "similitud"])
    if p.empty:
        resumen = pd.DataFrame({"chequeo": ["Pares con similitud >= %.2f" % umbral], "casos": [0]})
        return resumen, pd.DataFrame(columns=["split_a", "split_b", "variedad_a", "variedad_b", "etiqueta_a", "etiqueta_b", "similitud", "texto_a", "texto_b"])
    p = p.assign(
        split_a=c.loc[p.i, "split"].values, split_b=c.loc[p.j, "split"].values,
        variedad_a=c.loc[p.i, "variedad"].values, variedad_b=c.loc[p.j, "variedad"].values,
        etiqueta_a=c.loc[p.i, "etiqueta"].values, etiqueta_b=c.loc[p.j, "etiqueta"].values,
        texto_a=c.loc[p.i, "texto"].map(E.anonimizar).values, texto_b=c.loc[p.j, "texto"].map(E.anonimizar).values,
    )
    entre = p["split_a"] != p["split_b"]
    resumen = pd.DataFrame([
        (f"Pares con similitud >= {umbral:.2f}", len(p)),
        ("  de ellos, entre train y dev (fuga entre particiones)", int(entre.sum())),
        ("  de ellos, con etiquetas distintas", int((p.etiqueta_a != p.etiqueta_b).sum())),
        ("  de ellos, de variedades distintas", int((p.variedad_a != p.variedad_b).sum())),
        ("Tuits del dev con un casi duplicado en train", int(len(set(np.where(p.split_a == "dev", p.i, p.j)[entre.values])))),
    ], columns=["chequeo", "casos"])
    pares_out = p.drop(columns=["i", "j"]).sort_values("similitud", ascending=False).reset_index(drop=True)
    return resumen, pares_out


# ----------------------------------------------------------------------------
# Validación de heurísticas (cuando la muestra ya fue anotada a mano)
# ----------------------------------------------------------------------------
COL_VALIDACION = "es_correcto (SI/NO)"


def precision_heuristicas(muestra: pd.DataFrame) -> pd.DataFrame:
    """Precisión de cada regla = fracción de marcas confirmadas a mano, con IC 95% de Wilson."""
    m = muestra.copy()
    m["_r"] = m[COL_VALIDACION].astype(str).str.strip().str.upper()
    m = m[m["_r"].isin(["SI", "SÍ", "NO"])]
    filas = []
    for fen, g in m.groupby("fenomeno"):
        k, n = int(g["_r"].isin(["SI", "SÍ"]).sum()), len(g)
        lo, hi = wilson(k, n)
        filas.append({"fenomeno": fen, "revisados": n, "correctos": k, "precision": k / n, "ic_inf": lo, "ic_sup": hi})
    return pd.DataFrame(filas).round(3)
