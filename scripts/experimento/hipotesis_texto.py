"""Variantes por hipótesis lingüística: ¿qué pasa si quito cierta clase de palabras o la puntuación?"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np, pandas as pd, spacy

AQUI = Path(__file__).resolve().parent
sys.path[:0] = [str(AQUI)]
import validacion_cruzada as VC
import optimizar_hiperparametros as H
from sklearn.metrics import f1_score

NEGACIONES = {"no", "ni", "nunca", "jamás", "jamas", "tampoco", "nada", "nadie", "ningún", "ninguno", "ninguna", "sin"}
FUNCIONALES = {"DET", "ADP", "CCONJ", "SCONJ", "PRON", "AUX", "PART", "PUNCT", "SYM"}
CONTENIDO = {"NOUN", "PROPN", "ADJ", "VERB", "ADV", "INTJ"}

VARIANTES = {
    "sin_articulos": ("Los artículos no aportan información, agregan ruido", lambda x, p: p != "DET"),
    "sin_puntuacion": ("La puntuación no aporta información", lambda x, p: p not in ("PUNCT", "SYM")),
    "sin_articulos_ni_puntuacion": ("Artículos y puntuación juntos", lambda x, p: p not in ("DET", "PUNCT", "SYM")),
    "sin_funcionales": ("Las palabras funcionales (artículos, preposiciones, conjunciones, pronombres, auxiliares) son ruido; se conservan las negaciones", lambda x, p: p not in FUNCIONALES or x.lower() in NEGACIONES),
    "solo_contenido": ("Solo sustantivos, adjetivos, verbos, adverbios e interjecciones (más negaciones) llevan el sentimiento", lambda x, p: p in CONTENIDO or x.lower() in NEGACIONES),
}


def textos(train: pd.DataFrame) -> dict[str, list[str]]:
    nlp = spacy.load("es_core_news_md", disable=["ner", "parser", "lemmatizer"])
    base = train.texto.map(lambda t: VC.V.aplicar(t, "base")).tolist()
    out = {"control": [], **{k: [] for k in VARIANTES}}
    for d in nlp.pipe(base, batch_size=256):
        tk = [(t.text, t.pos_) for t in d if not t.is_space]
        out["control"].append(" ".join(x for x, _ in tk))
        for k, (_, f) in VARIANTES.items():
            out[k].append(" ".join(x for x, p in tk if f(x, p)))
    return out


def correr(train, txt: list[str], familia: str):
    y = train.etiqueta.to_numpy(); s = pd.Series(txt); oof = np.empty(len(train), dtype=object); f1s = []
    for tr, va in VC.pliegues(train):
        v = (VC._vec_palabras() if familia == "palabras" else VC._vec_caracteres()).fit(s.iloc[tr])
        p = VC._lr().fit(v.transform(s.iloc[tr]), y[tr]).predict(v.transform(s.iloc[va]))
        oof[va] = p; f1s.append(f1_score(y[va], p, average="macro"))
    return np.array(f1s), oof


def main() -> None:
    train, _ = H.cargar("data")
    VC.Representacion.precalcular(train, _)
    y = train.etiqueta.to_numpy(); T = textos(train); filas = []
    for fam in ("palabras", "caracteres"):
        f_c, oof_c = correr(train, T["control"], fam)
        filas.append({"familia": fam, "variante": "control", "hipotesis": "Texto sin modificar", "media": f_c.mean(), "desv": f_c.std(ddof=1),
                      **{f"fold{i+1}": v for i, v in enumerate(f_c)}, "dif": 0.0, "ic_inf": 0.0, "ic_sup": 0.0, "folds_a_favor": 0,
                      "palabras_por_tuit": np.mean([len(t.split()) for t in T["control"]])})
        for k, (hip, _f) in VARIANTES.items():
            f, oof = correr(train, T[k], fam)
            d = VC.dif_pareada_oof(y, oof_c, oof)
            filas.append({"familia": fam, "variante": k, "hipotesis": hip, "media": f.mean(), "desv": f.std(ddof=1),
                          **{f"fold{i+1}": v for i, v in enumerate(f)}, "dif": d["dif"], "ic_inf": d["ic_inf"], "ic_sup": d["ic_sup"],
                          "folds_a_favor": int((f > f_c).sum()), "palabras_por_tuit": np.mean([len(t.split()) for t in T[k]])})
            print(fam, k, round(f.mean(), 4), round(d["dif"], 4), flush=True)
    out = AQUI.parent.parent / "docs" / "validacion" / "tablas" / "variantes_hipotesis_cv.csv"
    pd.DataFrame(filas).to_csv(out, index=False); print("escrito", out)


if __name__ == "__main__":
    main()
