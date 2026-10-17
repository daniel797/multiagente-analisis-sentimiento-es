"""
Feature Engineering para el análisis de sentimiento en variedades del español.

El experimento principal (monoagente vs. multiagente) no entrena nada: un LLM recibe el
texto y devuelve una etiqueta. Aun así hay trabajo de ingeniería de características:

1. Representación del texto que ve el LLM (normalización mínima, sin pasar la variedad).
2. Catálogo de características lingüísticas y de superficie, calculadas con reglas, que sirven
   para (a) entender qué señales tiene el corpus, (b) contrastar lo que reporta el agente de
   Percepción, y (c) construir líneas base de referencia.
3. Representaciones TF-IDF (palabras y caracteres), usadas SOLO como piso de comparación.
4. Análisis de qué grupos de características aportan señal y cuáles delatan la variedad
   en lugar del sentimiento (posible atajo o confusión).

Regla de higiene: todo lo que se ajusta (vectorizadores, escalado, modelos de referencia) se ajusta
SOLO con train y se mide SOLO en dev. Nada de esto toca una partición de prueba.
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
import regex as re
from scipy.sparse import csr_matrix, hstack
from scipy.stats import kruskal
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.append(str(Path(__file__).resolve().parents[1] / "eda"))
import eda_extra as X  # noqa: E402
import eda_utils as E  # noqa: E402

ETIQUETAS = E.ETIQUETAS

# ----------------------------------------------------------------------------
# Patrones
# ----------------------------------------------------------------------------
_MENCION = re.compile(r"@\w+")
_HASHTAG = re.compile(r"#\w+")
_URL = re.compile(r"https?://\S+|www\.\S+")
_EMOJI = re.compile(r"\p{Extended_Pictographic}")
_EMOTICON = re.compile(r"(?<!\w)[:;=][\-o]?[\)\(DPpd\\/]|<3")
_ALARGA = re.compile(r"(\p{L})\1{2,}")
_PALABRA = re.compile(r"\p{L}+")
_PALABRA_MAYUS = re.compile(r"\b\p{Lu}{3,}\b")
_ORACION = re.compile(r"[^.!?¡¿]+[.!?]*")

_DIMINUTIVO = re.compile(r"\b\w+(ito|ita|itos|itas|cito|cita)\b", re.IGNORECASE)
_NEGACION = re.compile(r"\b(no|nunca|jamas|nada|nadie|ni)\b", re.IGNORECASE)
_INGLES = re.compile(r"\b(the|and|but|so|like|just|really|love|please|thanks|omg|lol|work)\b", re.IGNORECASE)
_RISA = re.compile(r"(?:ja|je|ji){2,}|\bxd+\b|\blol\b", re.IGNORECASE)
_INTENSIF = re.compile(
    r"\b(muy|tan|super|demasiado|mucho|mucha|muchos|muchas|bastante|totalmente|absolutamente|realmente)\b",
    re.IGNORECASE,
)

# Léxico mínimo propio (sin acentos). Es corto y transparente a propósito: no reemplaza a un léxico
# validado. Se incluye para medir cuánta señal aporta una lista pequeña antes de buscar una mejor.
LEXICO_POSITIVO = {
    "bien", "bueno", "buena", "buenos", "buenas", "feliz", "felices", "felicidades", "gracias", "genial",
    "excelente", "lindo", "linda", "hermoso", "hermosa", "amor", "amo", "encanta", "mejor", "gusta",
    "exito", "exitos", "divertido", "increible", "bonito", "bonita", "orgullo", "gracioso",
}
LEXICO_NEGATIVO = {
    "mal", "malo", "mala", "peor", "odio", "triste", "tristeza", "asco", "horrible", "pesimo", "pesima",
    "feo", "fea", "cansado", "cansada", "harto", "harta", "molesto", "molesta", "rabia", "dolor", "miedo",
    "problema", "problemas", "lamentable", "culpa", "verguenza", "estupido", "estupida",
}


def _sin_acentos(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


# ----------------------------------------------------------------------------
# Catálogo de características
# ----------------------------------------------------------------------------
CATALOGO = [
    # (nombre, grupo, tipo, definición, por qué podría importar)
    ("n_caracteres", "estructural", "conteo", "Longitud del tuit en caracteres.", "Tuits más largos suelen desarrollar más una opinión."),
    ("n_palabras", "estructural", "conteo", "Número de palabras (secuencias de letras).", "Redundante con n_caracteres, se mide para decidir si se conserva."),
    ("long_media_palabra", "estructural", "razón", "Caracteres de letras por palabra.", "Un registro más formal suele usar palabras más largas."),
    ("n_oraciones", "estructural", "conteo", "Segmentos terminados en . ! ? (mínimo 1).", "Cambios de oración pueden marcar contraste o ironía."),
    ("ratio_mayusculas", "estructural", "razón", "Fracción de letras en mayúscula.", "Mayúsculas sostenidas expresan énfasis o enojo."),
    ("n_digitos", "estructural", "conteo", "Cantidad de dígitos.", "Cifras y fechas suelen acompañar texto informativo (NEU)."),
    ("n_exclamaciones", "expresividad", "conteo", "Signos ! en el tuit.", "Asociado a emoción, sobre todo positiva."),
    ("n_interrogaciones", "expresividad", "conteo", "Signos ? en el tuit.", "Las preguntas tienden a ser neutras."),
    ("n_puntos_suspensivos", "expresividad", "conteo", "Secuencias de 3 o más puntos.", "Pueden marcar duda, ironía o tristeza."),
    ("n_alargamientos", "expresividad", "conteo", "Letras repetidas 3 o más veces (holaaa).", "Expresividad oral típica de redes."),
    ("n_palabras_mayuscula", "expresividad", "conteo", "Palabras de 3+ letras todas en mayúscula.", "Énfasis explícito."),
    ("n_emoticonos", "expresividad", "conteo", "Emoticonos de texto como :) o :(.", "Señal directa de polaridad, aunque escasa."),
    ("n_emojis", "expresividad", "conteo", "Emojis (símbolos pictográficos).", "Casi no hay en este corpus (menos del 0,5%)."),
    ("n_menciones", "conversacional", "conteo", "Menciones @usuario.", "Indica conversación directa."),
    ("n_hashtags", "conversacional", "conteo", "Hashtags #tema.", "Marca tema o campaña."),
    ("tiene_url", "conversacional", "binaria", "Contiene un enlace.", "Suele indicar contenido informativo."),
    ("empieza_con_mencion", "conversacional", "binaria", "El tuit empieza con @usuario (respuesta).", "Las respuestas son más personales."),
    ("n_diminutivos", "fenomeno", "conteo", "Palabras con terminación -ito/-ita y similares.", "Fenómeno del agente de Percepción (diminutivo afectivo). Regla aproximada."),
    ("n_negaciones", "fenomeno", "conteo", "no, nunca, jamás, nada, nadie, ni.", "Fenómeno del agente de Percepción. La doble negación sale de n_negaciones >= 2."),
    ("n_marcadores_ingles", "fenomeno", "conteo", "Palabras de una lista corta en inglés.", "Aproxima el code-switching, muy escaso aquí."),
    ("n_risas", "fenomeno", "conteo", "jaja, jeje, xd, lol.", "Marcador de humor. No mide sarcasmo."),
    ("n_intensificadores", "lexico", "conteo", "muy, tan, súper, demasiado, mucho...", "Refuerzan la polaridad que acompañan."),
    ("n_lex_positivo", "lexico", "conteo", "Palabras de un léxico mínimo propio de polaridad positiva.", "Léxico corto y transparente, no validado."),
    ("n_lex_negativo", "lexico", "conteo", "Palabras de un léxico mínimo propio de polaridad negativa.", "Léxico corto y transparente, no validado."),
]
NOMBRES = [c[0] for c in CATALOGO]
GRUPOS = {g: [c[0] for c in CATALOGO if c[1] == g] for g in dict.fromkeys(c[1] for c in CATALOGO)}


def catalogo_features() -> pd.DataFrame:
    return pd.DataFrame(CATALOGO, columns=["feature", "grupo", "tipo", "definicion", "por_que_importa"])


# ----------------------------------------------------------------------------
# Extracción
# ----------------------------------------------------------------------------
def extraer_features(texto: str) -> dict:
    t = texto or ""
    sa = _sin_acentos(t)
    palabras = _PALABRA.findall(t)
    letras = sum(c.isalpha() for c in t)
    n_pal = len(palabras)
    tokens_sa = set(_PALABRA.findall(sa.lower()))
    oraciones = [o for o in _ORACION.findall(t) if o.strip()]
    return {
        "n_caracteres": len(t),
        "n_palabras": n_pal,
        "long_media_palabra": (sum(len(p) for p in palabras) / n_pal) if n_pal else 0.0,
        "n_oraciones": max(1, len(oraciones)),
        "ratio_mayusculas": (sum(c.isupper() for c in t) / letras) if letras else 0.0,
        "n_digitos": sum(c.isdigit() for c in t),
        "n_exclamaciones": t.count("!"),
        "n_interrogaciones": t.count("?"),
        "n_puntos_suspensivos": len(re.findall(r"\.{3,}|…", t)),
        "n_alargamientos": len(_ALARGA.findall(t)),
        "n_palabras_mayuscula": len(_PALABRA_MAYUS.findall(t)),
        "n_emoticonos": len(_EMOTICON.findall(t)),
        "n_emojis": len(_EMOJI.findall(t)),
        "n_menciones": len(_MENCION.findall(t)),
        "n_hashtags": len(_HASHTAG.findall(t)),
        "tiene_url": int(bool(_URL.search(t))),
        "empieza_con_mencion": int(t.lstrip().startswith("@")),
        "n_diminutivos": len(_DIMINUTIVO.findall(sa)),
        "n_negaciones": len(_NEGACION.findall(sa)),
        "n_marcadores_ingles": len(_INGLES.findall(sa)),
        "n_risas": len(_RISA.findall(sa)),
        "n_intensificadores": len(_INTENSIF.findall(sa)),
        "n_lex_positivo": len(tokens_sa & LEXICO_POSITIVO),
        "n_lex_negativo": len(tokens_sa & LEXICO_NEGATIVO),
    }


def construir_matriz(df: pd.DataFrame) -> pd.DataFrame:
    """Una fila por tuit y una columna por característica del catálogo (mismo índice que df)."""
    filas = df["texto"].map(extraer_features)
    return pd.DataFrame(list(filas), index=df.index)[NOMBRES]


# ----------------------------------------------------------------------------
# Texto que ve el LLM
# ----------------------------------------------------------------------------
def normalizar_para_llm(texto: str) -> str:
    """
    Normalización mínima del tuit antes de ponerlo en un prompt:
    - @usuario -> @USUARIO y enlaces -> URL (privacidad y menos ruido),
    - espacios repetidos colapsados.
    NO se toca mayúsculas, puntuación, alargamientos, emojis ni tildes: son señales que el LLM debe ver.
    NO se agrega la variedad dialectal: el sistema no debe saber de qué país es el tuit.
    """
    t = _URL.sub("URL", _MENCION.sub("@USUARIO", texto or ""))
    return re.sub(r"\s+", " ", t).strip()


def pistas_percepcion(texto: str) -> dict:
    """
    Banderas heurísticas con la misma forma que la salida del agente de Percepción.
    Sirven para contrastar al agente (acuerdo entre agente y regla) y como piso de comparación,
    no como verdad. La regla de risa se llama marcador_risa y no sarcasmo.
    """
    f = extraer_features(texto)
    return {
        "diminutivo_afectivo": f["n_diminutivos"] > 0,
        "doble_negacion": f["n_negaciones"] >= 2,
        "code_switching": f["n_marcadores_ingles"] > 0,
        "marcador_risa": f["n_risas"] > 0,
    }


def perfil_pistas(corpus: pd.DataFrame, matriz: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada bandera heurística de Percepción: cuántos tuits la activan y cómo se reparten sus etiquetas
    frente al total. Muestra si la bandera, sola, dice algo del sentimiento.
    """
    banderas = {
        "diminutivo_afectivo": matriz["n_diminutivos"] > 0,
        "doble_negacion": matriz["n_negaciones"] >= 2,
        "code_switching": matriz["n_marcadores_ingles"] > 0,
        "marcador_risa": matriz["n_risas"] > 0,
    }
    base = corpus["etiqueta"].value_counts(normalize=True).reindex(ETIQUETAS) * 100
    filas = [{"bandera": "(todos los tuits)", "n": len(corpus), **{f"{e}_pct": round(base[e], 1) for e in ETIQUETAS}}]
    for nombre, mascara in banderas.items():
        sub = corpus.loc[mascara.values, "etiqueta"]
        pct = sub.value_counts(normalize=True).reindex(ETIQUETAS).fillna(0) * 100
        filas.append({"bandera": nombre, "n": int(mascara.sum()), **{f"{e}_pct": round(pct[e], 1) for e in ETIQUETAS}})
    return pd.DataFrame(filas)


# ----------------------------------------------------------------------------
# Asociación univariada y confusión con la variedad
# ----------------------------------------------------------------------------
def _epsilon2(valores: pd.Series, grupos: pd.Series) -> tuple[float, float]:
    """Kruskal-Wallis entre grupos. Devuelve (epsilon², p). epsilon² = H/(n-1): tamaño de efecto 0 a 1."""
    muestras = [valores[grupos == g].values for g in grupos.unique()]
    if all(np.all(m == muestras[0][0]) for m in muestras):
        return 0.0, 1.0
    h, p = kruskal(*muestras)
    return float(h / (len(valores) - 1)), float(p)


def asociacion_univariada(matriz: pd.DataFrame, etiquetas: pd.Series, variedades: pd.Series) -> pd.DataFrame:
    """
    Para cada característica: tamaño de efecto respecto a la etiqueta y respecto a la variedad.
    Si una característica se relaciona más con la variedad que con la etiqueta, puede actuar como
    atajo (el sistema aprendería el país en lugar del sentimiento) o reflejar el sesgo de recolección.
    """
    filas = []
    for f in matriz.columns:
        e_et, p_et = _epsilon2(matriz[f], etiquetas)
        e_va, p_va = _epsilon2(matriz[f], variedades)
        filas.append({
            "feature": f, "grupo": next(c[1] for c in CATALOGO if c[0] == f),
            "efecto_etiqueta": e_et, "p_etiqueta": p_et, "efecto_variedad": e_va, "p_variedad": p_va,
            "mas_ligada_a": "variedad" if e_va > e_et else "etiqueta",
        })
    out = pd.DataFrame(filas)
    return out.sort_values("efecto_etiqueta", ascending=False).reset_index(drop=True).round(5)


def medias_por_etiqueta(matriz: pd.DataFrame, etiquetas: pd.Series) -> pd.DataFrame:
    return matriz.groupby(etiquetas).mean().T[ETIQUETAS].round(3)


def correlaciones_altas(matriz: pd.DataFrame, umbral: float = 0.85) -> pd.DataFrame:
    """Pares de características con correlación de Spearman en valor absoluto >= umbral (redundancia)."""
    c = matriz.corr(method="spearman").abs()
    filas = []
    for i, a in enumerate(c.columns):
        for b in c.columns[i + 1:]:
            if c.loc[a, b] >= umbral:
                filas.append({"feature_a": a, "feature_b": b, "spearman_abs": round(float(c.loc[a, b]), 3)})
    return pd.DataFrame(filas, columns=["feature_a", "feature_b", "spearman_abs"])


# ----------------------------------------------------------------------------
# Modelos de referencia por grupos de características
# ----------------------------------------------------------------------------
def _lr() -> LogisticRegression:
    return LogisticRegression(max_iter=3000, class_weight="balanced", random_state=0)


def _tfidf_palabras() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)


def _tfidf_caracteres() -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=60000)


def _numericas(grupos: list[str]) -> list[str]:
    return [f for g in grupos for f in GRUPOS[g]]


def comparar_representaciones(corpus: pd.DataFrame, n_boot: int = 500) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compara qué representación aporta más señal. Todo se ajusta con train y se mide en dev.
    Devuelve (resumen_conjunto, macro_f1_por_variedad).
    """
    c = corpus.reset_index(drop=True)
    c["_txt"] = c["texto"].map(normalizar_para_llm)
    M = construir_matriz(c)
    tr, dv = (c.split == "train").values, (c.split == "dev").values
    y_tr, y_dv = c.loc[tr, "etiqueta"], c.loc[dv, "etiqueta"]

    def num(grupos):
        cols = _numericas(grupos)
        sc = StandardScaler().fit(M.loc[tr, cols])
        return sc.transform(M.loc[tr, cols]), sc.transform(M.loc[dv, cols])

    def tfidf(factory):
        v = factory().fit(c.loc[tr, "_txt"])
        return v.transform(c.loc[tr, "_txt"]), v.transform(c.loc[dv, "_txt"])

    A_tr, A_dv = num(["estructural", "expresividad", "conversacional"])
    B_tr, B_dv = num(["estructural", "expresividad", "conversacional", "fenomeno", "lexico"])
    W_tr, W_dv = tfidf(_tfidf_palabras)
    C_tr, C_dv = tfidf(_tfidf_caracteres)

    def j(*mats):
        return hstack([csr_matrix(m) for m in mats]).tocsr()

    modelos = {
        "0. Siempre la clase mayoritaria": None,
        "1. Superficie (estructural, expresividad, conversacional)": (A_tr, A_dv),
        "2. Superficie + fenómenos + léxico mínimo": (B_tr, B_dv),
        "3. TF-IDF de palabras": (W_tr, W_dv),
        "4. TF-IDF de caracteres": (C_tr, C_dv),
        "5. TF-IDF palabras + caracteres": (j(W_tr, C_tr), j(W_dv, C_dv)),
        "6. TF-IDF palabras + todas las numéricas": (j(W_tr, B_tr), j(W_dv, B_dv)),
    }
    resumen, por_var = [], []
    var_dv = c.loc[dv, "variedad"].values
    for nombre, mats in modelos.items():
        if mats is None:
            pred = np.full(dv.sum(), y_tr.value_counts().idxmax())
        else:
            m = _lr().fit(mats[0], y_tr)
            pred = m.predict(mats[1])
        lo, hi = X._ic_macro_f1(y_dv, pred, n_boot=n_boot)
        resumen.append({
            "representacion": nombre, "n_dev": int(dv.sum()),
            "accuracy_dev": accuracy_score(y_dv, pred),
            "macro_f1_dev": f1_score(y_dv, pred, labels=ETIQUETAS, average="macro", zero_division=0),
            "macro_f1_ic_inf": lo, "macro_f1_ic_sup": hi,
        })
        for v in sorted(set(var_dv)):
            idx = var_dv == v
            por_var.append({"representacion": nombre, "variedad": v,
                            "macro_f1_dev": f1_score(y_dv.values[idx], pred[idx], labels=ETIQUETAS, average="macro", zero_division=0)})
    return pd.DataFrame(resumen).round(4), pd.DataFrame(por_var).round(4)


def terminos_por_clase(corpus: pd.DataFrame, k: int = 12) -> pd.DataFrame:
    """Términos de TF-IDF de palabras con mayor peso por clase en una regresión logística (solo train)."""
    c = corpus[corpus.split == "train"]
    v = _tfidf_palabras().fit(c["texto"].map(normalizar_para_llm))
    m = _lr().fit(v.transform(c["texto"].map(normalizar_para_llm)), c["etiqueta"])
    nombres = np.array(v.get_feature_names_out())
    filas = []
    for i, cl in enumerate(m.classes_):
        top = np.argsort(m.coef_[i])[::-1][:k]
        filas += [{"clase": cl, "rank": r + 1, "termino": nombres[t], "peso": round(float(m.coef_[i][t]), 3)} for r, t in enumerate(top)]
    return pd.DataFrame(filas)


def prediccion_de_variedad(corpus: pd.DataFrame, n_boot: int = 300) -> pd.DataFrame:
    """
    ¿Se puede adivinar el país solo con el texto? Si sí, el LLM podría 'saber' de dónde es el tuit aunque
    no se le diga, y las diferencias entre países pueden mezclarse con diferencias de tema o época.
    """
    c = corpus
    txt = c["texto"].map(normalizar_para_llm)
    tr, dv = (c.split == "train").values, (c.split == "dev").values
    out = []
    for nombre, factory in (("TF-IDF de palabras", _tfidf_palabras), ("TF-IDF de caracteres", _tfidf_caracteres)):
        v = factory().fit(txt[tr])
        m = _lr().fit(v.transform(txt[tr]), c.loc[tr, "variedad"])
        pred = m.predict(v.transform(txt[dv]))
        y = c.loc[dv, "variedad"].values
        rng = np.random.default_rng(0)
        vals = []
        for _ in range(n_boot):
            i = rng.integers(0, len(y), len(y))
            vals.append(f1_score(y[i], pred[i], average="macro"))
        out.append({
            "representacion": nombre, "n_dev": int(dv.sum()), "n_variedades": c["variedad"].nunique(),
            "azar_aprox": round(1 / c["variedad"].nunique(), 3),
            "accuracy_dev": round(accuracy_score(y, pred), 4), "macro_f1_dev": round(f1_score(y, pred, average="macro"), 4),
            "macro_f1_ic_inf": round(float(np.percentile(vals, 2.5)), 4), "macro_f1_ic_sup": round(float(np.percentile(vals, 97.5)), 4),
        })
    return pd.DataFrame(out)
