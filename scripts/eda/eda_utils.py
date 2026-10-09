"""
Funciones del análisis exploratorio (EDA) del corpus TASS 2020 por variedad.

Las usan tanto `scripts/eda/run_eda.py` (genera tablas y figuras en docs/eda/)
como el notebook `notebooks/02_eda.ipynb`, para que las cifras sean las mismas
en ambos lados.

Estructura de datos esperada (la que tiene el repositorio):

    data/train/<variedad>.tsv
    data/dev/<variedad>.tsv

Cada archivo es un TSV SIN encabezado con tres columnas: id del tuit, texto y
etiqueta (P / N / NEU). Los textos pueden contener comillas, por eso se lee con
quoting=QUOTE_NONE.
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re
from scipy.stats import chi2_contingency, ks_2samp
from sklearn.metrics import accuracy_score, f1_score

sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.data.audit_phenomena import contar_fenomenos  # noqa: E402

ETIQUETAS = ["NEG", "NEU", "POS"]  # orden de menos a más positivo
LABEL_MAP = {"N": "NEG", "NEU": "NEU", "P": "POS"}
SPLITS = ("train", "dev")

# Epoch de los ids de Twitter (snowflake): permite recuperar la fecha del tuit.
_TWITTER_EPOCH_MS = 1288834974657

_RE_MENCION = re.compile(r"@\w+")
_RE_HASHTAG = re.compile(r"#\w+")
_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_EMOJI = re.compile(r"\p{Extended_Pictographic}")
_RE_EMOTICON = re.compile(r"(?<!\w)[:;=][\-o]?[\)\(DPpd\\/]|<3")
_RE_ALARGA = re.compile(r"(\p{L})\1{2,}")
_RE_PALABRA = re.compile(r"[a-záéíóúñü]{3,}")

STOPWORDS = set(
    """
    que los las del con por para una uno unos unas como pero mas más muy sus
    este esta esto estos estas ese esa eso esos esas son fue ser era hay han
    has había tiene tengo tienes tener hace hacer solo sólo ya así asi sin
    sobre entre cuando donde porque pues bien también tambien todo todos toda
    todas nada algo les nos nuestro mis mío mia tus tuyo ahora aqui aquí allí
    ahí vez veces cada otro otra otros otras puede pueden poder voy vas van
    vamos estoy está están estamos estar estaba ver visto algun algún alguna
    mismo misma desde hasta durante tan tanto tanta mucho mucha muchos muchas
    poco poca pocos quien quién cual cuál cuales
    """.split()
)


# ----------------------------------------------------------------------------
# Carga
# ----------------------------------------------------------------------------
def cargar_corpus(data_dir: str | Path = "data") -> pd.DataFrame:
    """Lee data/<split>/<variedad>.tsv y devuelve un único DataFrame."""
    data_dir = Path(data_dir)
    frames = []
    for split in SPLITS:
        carpeta = data_dir / split
        if not carpeta.exists():
            raise FileNotFoundError(f"No existe {carpeta}. Se esperaba data/train y data/dev.")
        for archivo in sorted(carpeta.glob("*.tsv")):
            df = pd.read_csv(
                archivo, sep="\t", header=None, names=["id", "texto", "etiqueta_raw"],
                quoting=csv.QUOTE_NONE, dtype=str, keep_default_na=False, encoding="utf-8",
            )
            df["variedad"] = archivo.stem.upper()
            df["split"] = split
            frames.append(df)
    if not frames:
        raise FileNotFoundError(f"No se encontraron .tsv dentro de {data_dir}.")
    corpus = pd.concat(frames, ignore_index=True)
    corpus["etiqueta"] = corpus["etiqueta_raw"].str.strip().str.upper().map(LABEL_MAP)
    sin_mapear = corpus["etiqueta"].isna().sum()
    if sin_mapear:
        raise ValueError(f"{sin_mapear} filas con etiqueta fuera de {sorted(LABEL_MAP)}.")
    return corpus.drop(columns="etiqueta_raw")


def anonimizar(texto: str) -> str:
    """Reemplaza @usuarios por @USUARIO para no exponer cuentas en tablas o ejemplos."""
    return _RE_MENCION.sub("@USUARIO", texto)


def _normalizar(texto: str) -> str:
    t = _RE_URL.sub("", _RE_MENCION.sub("", texto.lower()))
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------------------
# Calidad de datos
# ----------------------------------------------------------------------------
def reporte_calidad(corpus: pd.DataFrame) -> pd.DataFrame:
    """Una fila por chequeo: qué se revisó y cuántos casos hay."""
    norm = corpus["texto"].map(_normalizar)
    etiquetas_por_texto = corpus.assign(_n=norm).groupby("_n")["etiqueta"].nunique()
    textos_en_conflicto = etiquetas_por_texto[etiquetas_por_texto > 1].index
    ids_train = set(corpus.loc[corpus.split == "train", "id"])
    ids_dev = set(corpus.loc[corpus.split == "dev", "id"])
    textos_train = set(norm[corpus.split == "train"])
    textos_dev = set(norm[corpus.split == "dev"])

    filas = [
        ("Filas totales", len(corpus)),
        ("Textos vacíos", int((corpus["texto"].str.strip() == "").sum())),
        ("Ids repetidos (filas extra)", int(corpus["id"].duplicated().sum())),
        ("Textos repetidos tras normalizar (filas extra)", int(norm.duplicated().sum())),
        ("Textos repetidos con etiquetas distintas (textos)", int(len(textos_en_conflicto))),
        ("Ids presentes en train y dev a la vez", len(ids_train & ids_dev)),
        ("Textos presentes en train y dev a la vez", len(textos_train & textos_dev)),
        ("Tuits con URL", int(corpus["texto"].map(lambda x: bool(_RE_URL.search(x))).sum())),
    ]
    return pd.DataFrame(filas, columns=["chequeo", "casos"])


# ----------------------------------------------------------------------------
# Etiquetas
# ----------------------------------------------------------------------------
def tabla_etiquetas(corpus: pd.DataFrame) -> pd.DataFrame:
    """Conteo y porcentaje de cada etiqueta por split y variedad."""
    conteo = corpus.groupby(["split", "variedad", "etiqueta"]).size().unstack(fill_value=0)[ETIQUETAS]
    pct = conteo.div(conteo.sum(axis=1), axis=0) * 100
    out = conteo.join(pct, rsuffix="_pct")
    out["n"] = conteo.sum(axis=1)
    return out.reset_index()


def cramers_v(tabla: pd.DataFrame) -> tuple[float, float, float]:
    """(chi2, p, V de Cramér) de una tabla de contingencia."""
    chi2, p, _, _ = chi2_contingency(tabla)
    n = tabla.to_numpy().sum()
    k = min(tabla.shape) - 1
    return float(chi2), float(p), float(np.sqrt(chi2 / (n * k)))


def asociacion_etiqueta_variedad(corpus: pd.DataFrame) -> pd.DataFrame:
    """¿Cambia la distribución de etiquetas entre variedades? ¿Y entre train y dev?"""
    filas = []
    tr = corpus[corpus.split == "train"]
    chi2, p, v = cramers_v(pd.crosstab(tr["variedad"], tr["etiqueta"]))
    filas.append(("Etiqueta vs. variedad (train)", "todas", chi2, p, v))
    for var, g in corpus.groupby("variedad"):
        chi2, p, v = cramers_v(pd.crosstab(g["split"], g["etiqueta"]))
        filas.append(("Etiqueta vs. split (train/dev)", var, chi2, p, v))
    return pd.DataFrame(filas, columns=["comparacion", "variedad", "chi2", "p_valor", "v_cramer"])


def baseline_mayoritario(corpus: pd.DataFrame) -> pd.DataFrame:
    """
    Piso de referencia: predecir siempre la clase más frecuente del train de esa
    variedad y medirlo en su dev. Cualquier sistema debe superar esto con holgura.
    """
    filas = []
    for var, g in corpus.groupby("variedad"):
        tr, dv = g[g.split == "train"], g[g.split == "dev"]
        mayoritaria = tr["etiqueta"].value_counts().idxmax()
        pred = np.full(len(dv), mayoritaria)
        filas.append({
            "variedad": var,
            "clase_mayoritaria_train": mayoritaria,
            "n_dev": len(dv),
            "accuracy_dev": accuracy_score(dv["etiqueta"], pred),
            "macro_f1_dev": f1_score(dv["etiqueta"], pred, labels=ETIQUETAS, average="macro", zero_division=0),
            "margen_error_max_pp": 1.96 * np.sqrt(0.25 / len(dv)) * 100,
        })
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# Rasgos del texto
# ----------------------------------------------------------------------------
def agregar_rasgos(corpus: pd.DataFrame) -> pd.DataFrame:
    """Agrega columnas con rasgos propios de texto de redes sociales."""
    df = corpus.copy()
    t = df["texto"]
    letras = t.map(lambda s: sum(c.isalpha() for c in s))
    mayus = t.map(lambda s: sum(c.isupper() for c in s))
    df["longitud_car"] = t.str.len()
    df["longitud_pal"] = t.str.split().str.len()
    df["n_menciones"] = t.map(lambda s: len(_RE_MENCION.findall(s)))
    df["n_hashtags"] = t.map(lambda s: len(_RE_HASHTAG.findall(s)))
    df["tiene_url"] = t.map(lambda s: bool(_RE_URL.search(s)))
    df["tiene_emoji"] = t.map(lambda s: bool(_RE_EMOJI.search(s)))
    df["tiene_emoticon"] = t.map(lambda s: bool(_RE_EMOTICON.search(s)))
    df["tiene_alargamiento"] = t.map(lambda s: bool(_RE_ALARGA.search(s)))
    df["tiene_exclamacion"] = t.str.contains("!", regex=False)
    df["tiene_interrogacion"] = t.str.contains("?", regex=False)
    df["mayusculas_altas"] = (letras >= 10) & ((mayus / letras.clip(lower=1)) > 0.5)
    return df


RASGOS_BOOL = {
    "tiene_url": "URL",
    "n_menciones": "@mención",
    "n_hashtags": "#hashtag",
    "tiene_emoji": "emoji",
    "tiene_emoticon": "emoticón :)",
    "tiene_alargamiento": "letras alargadas",
    "tiene_exclamacion": "¡exclamación!",
    "tiene_interrogacion": "¿pregunta?",
    "mayusculas_altas": "MAYÚSCULAS",
}


def tabla_rasgos(df: pd.DataFrame, por: str = "variedad") -> pd.DataFrame:
    """% de tuits que presentan cada rasgo, por variedad o por etiqueta."""
    out = {}
    for col, nombre in RASGOS_BOOL.items():
        serie = df[col] > 0 if df[col].dtype != bool else df[col]
        out[nombre] = serie.groupby(df[por]).mean() * 100
    return pd.DataFrame(out).round(1)


# ----------------------------------------------------------------------------
# Fenómenos lingüísticos (heurísticos)
# ----------------------------------------------------------------------------
# OJO: son las mismas heurísticas de scripts/data/audit_phenomena.py. La que ese
# script llama "sarcasmo_aparente" solo detecta risas ("jaja", "jeje") y algunos
# emojis, no sarcasmo: aquí se rebautiza "marcador_risa" para no sobrar-interpretar.
FENOMENOS = {
    "diminutivo_afectivo": "diminutivo (-ito/-ita)",
    "doble_negacion": "2+ negaciones",
    "code_switching": "inglés (lista corta)",
    "sarcasmo_aparente": "risa / marcador humor",
}


def agregar_fenomenos(df: pd.DataFrame) -> pd.DataFrame:
    flags = df["texto"].map(contar_fenomenos).apply(pd.Series)
    return pd.concat([df, flags[list(FENOMENOS)]], axis=1)


def tabla_fenomenos(df: pd.DataFrame, split: str | None = None) -> pd.DataFrame:
    """Conteo absoluto y % de cada fenómeno por variedad (opcionalmente en un split)."""
    base = df if split is None else df[df.split == split]
    conteo = base.groupby("variedad")[list(FENOMENOS)].sum().astype(int)
    conteo["n"] = base.groupby("variedad").size()
    pct = (conteo[list(FENOMENOS)].div(conteo["n"], axis=0) * 100).round(1)
    return conteo.join(pct, rsuffix="_pct")


def muestra_para_validar(df: pd.DataFrame, n_por_fenomeno: int = 30, seed: int = 42) -> pd.DataFrame:
    """
    Muestra anonimizada de tuits marcados por cada heurística, con una columna vacía
    para que el autor confirme a mano si la marca es correcta. Sirve para estimar la
    precisión real de las heurísticas antes de apoyarse en ellas.
    """
    partes = []
    for col, nombre in FENOMENOS.items():
        marcados = df[df[col]]
        m = marcados.sample(min(n_por_fenomeno, len(marcados)), random_state=seed)
        partes.append(pd.DataFrame({
            "fenomeno": nombre, "variedad": m["variedad"].values, "id": m["id"].values,
            "texto_anonimizado": m["texto"].map(anonimizar).values, "es_correcto (SI/NO)": "",
        }))
    return pd.concat(partes, ignore_index=True)


# ----------------------------------------------------------------------------
# Vocabulario
# ----------------------------------------------------------------------------
def tokens(texto: str) -> list[str]:
    limpio = _RE_URL.sub(" ", _RE_MENCION.sub(" ", texto.lower()))
    return [p for p in _RE_PALABRA.findall(limpio) if p not in STOPWORDS]


def top_terminos_por_etiqueta(df: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    """Términos más frecuentes en cada etiqueta (filtrando stopwords y menciones)."""
    filas = []
    for et in ETIQUETAS:
        conteo = pd.Series([t for s in df.loc[df.etiqueta == et, "texto"] for t in tokens(s)]).value_counts()
        for rank, (termino, c) in enumerate(conteo.head(k).items(), 1):
            filas.append((et, rank, termino, int(c)))
    return pd.DataFrame(filas, columns=["etiqueta", "rank", "termino", "frecuencia"])


def vocabulario_distintivo(df: pd.DataFrame, k: int = 10, min_count: int = 8, alpha: float = 0.5) -> pd.DataFrame:
    """
    Términos que una variedad usa mucho más que las demás (log-odds suavizado).
    Sirve para detectar peruanismos/mexicanismos... y también temas de actualidad
    de cada país, que son un posible factor de confusión entre dialecto y tema.
    """
    por_var = {v: pd.Series([t for s in g["texto"] for t in tokens(s)]).value_counts() for v, g in df.groupby("variedad")}
    vocab = sorted(set().union(*[set(c.index) for c in por_var.values()]))
    mat = pd.DataFrame({v: c.reindex(vocab).fillna(0) for v, c in por_var.items()})
    filas = []
    for v in mat.columns:
        a, resto = mat[v], mat.drop(columns=v).sum(axis=1)
        lo = np.log((a + alpha) / (a.sum() + alpha * len(vocab))) - np.log((resto + alpha) / (resto.sum() + alpha * len(vocab)))
        cand = lo[a >= min_count].sort_values(ascending=False).head(k)
        for rank, (termino, score) in enumerate(cand.items(), 1):
            filas.append((v, rank, termino, int(a[termino]), round(float(score), 2)))
    return pd.DataFrame(filas, columns=["variedad", "rank", "termino", "frecuencia", "log_odds"])


# ----------------------------------------------------------------------------
# Fechas (derivadas del id del tuit)
# ----------------------------------------------------------------------------
def agregar_fecha(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    ids = pd.to_numeric(out["id"], errors="coerce")
    ms = (ids.astype("Int64") // (2**22)) + _TWITTER_EPOCH_MS
    out["fecha"] = pd.to_datetime(ms.astype("float64"), unit="ms", errors="coerce")
    return out


def tabla_fechas(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("variedad")["fecha"]
    return pd.DataFrame({"desde": g.min().dt.date, "hasta": g.max().dt.date, "fechas_invalidas": df["fecha"].isna().groupby(df["variedad"]).sum()})


def concentracion_temporal(df: pd.DataFrame) -> pd.DataFrame:
    """
    ¿Los tuits de cada variedad están repartidos en el tiempo o concentrados en ráfagas?
    ¿Train y dev cubren las mismas fechas? (ks_D = 0 significa misma distribución de fechas;
    se reporta D y no el p-valor porque con miles de tuits el p-valor siempre sale ~0.)
    """
    d = df.assign(semana=df["fecha"].dt.to_period("W").dt.start_time)
    filas = []
    for var, g in d.groupby("variedad"):
        por_semana = g.groupby("semana").size().sort_values(ascending=False)
        tr = g.loc[g.split == "train", "fecha"]
        dv = g.loc[g.split == "dev", "fecha"]
        ks = ks_2samp(tr.astype("int64"), dv.astype("int64"))
        filas.append({
            "variedad": var,
            "semanas_con_tuits": len(por_semana),
            "semana_pico": por_semana.index[0].date(),
            "tuits_semana_pico": int(por_semana.iloc[0]),
            "pct_en_semana_pico": round(por_semana.iloc[0] / len(g) * 100, 1),
            "pct_en_2_semanas_top": round(por_semana.iloc[:2].sum() / len(g) * 100, 1),
            "mediana_fecha_train": tr.median().date(),
            "mediana_fecha_dev": dv.median().date(),
            "dias_dev_despues_de_train": int((dv.median() - tr.median()).days),
            "ks_D_train_vs_dev": round(float(ks.statistic), 3),
        })
    return pd.DataFrame(filas)


def resumen_por_variedad(df: pd.DataFrame) -> pd.DataFrame:
    piv = df.groupby(["variedad", "split"]).size().unstack(fill_value=0)
    piv["total"] = piv.sum(axis=1)
    piv["longitud_media_car"] = df.groupby("variedad")["longitud_car"].mean().round(1)
    piv["longitud_media_pal"] = df.groupby("variedad")["longitud_pal"].mean().round(1)
    return piv.reset_index()
