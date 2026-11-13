"""
Validación cruzada del baseline y dos variantes, elección de la mejor, optimización de sus hiperparámetros y
ablaciones. Es la base de las cifras del informe parcial.

Todo se ajusta y se valida SOLO con train (5 pliegues estratificados por etiqueta y variedad, semilla 0). Dev
(2.465 tuits) se mide una única vez al final, con las configuraciones ya elegidas.

Candidatos (baseline + 2 variantes):
  M0  baseline   TF-IDF de palabras (1,2), min_df 2 + regresión logística (C = 1, clases balanceadas)
  M1  variante 1 TF-IDF de caracteres (2,5), min_df 3, máx. 60.000 + regresión logística (C = 1, balanceadas)
  M2  variante 2 TF-IDF de palabras (1,2) + las 24 variables de ingeniería de atributos, regresión logística
  M3  variante 3 TF-IDF de caracteres (2,5) + las 24 variables de ingeniería de atributos, regresión logística
Referencias simples (no son candidatos; sirven para ver cuánto aporta cada señal por sí sola):
  V1  solo las 24 variables (estandarizadas) + regresión logística
  V2  regla de léxico sin entrenamiento: POS si hay más palabras positivas que negativas, NEG si hay más
      negativas, NEU en otro caso

Uso (desde la raíz del repositorio):
    python scripts/experimento/validacion_cruzada.py
Salida: docs/validacion/tablas/*.csv y docs/validacion/figuras/*.png
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.sparse import csr_matrix, hstack  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

AQUI = Path(__file__).resolve().parent
for sub in ("experimento", "features", "eda"):
    sys.path.append(str(AQUI.parent / sub))
import evaluacion as EV  # noqa: E402
import feature_utils as F  # noqa: E402
import optimizar_hiperparametros as H  # noqa: E402
import variantes_datos as V  # noqa: E402

CLASES = ["NEG", "NEU", "POS"]
SEMILLA = 0
AZUL, GRIS, ROJO, VERDE = "#2a78d6", "#b8b7b0", "#d9503f", "#3a9a5b"

MODELOS = {
    "M0": "TF-IDF de palabras (baseline)",
    "M1": "TF-IDF de caracteres (variante 1)",
    "M2": "Palabras + 24 variables (variante 2)",
    "M3": "Caracteres + 24 variables (variante 3)",
    "V1": "24 variables solas (referencia)",
    "V2": "Regla de léxico sin entrenamiento (referencia)",
}


def predecir_regla(df: pd.DataFrame) -> np.ndarray:
    pos, neg = df["n_lex_positivo"].to_numpy(), df["n_lex_negativo"].to_numpy()
    return np.where(pos > neg, "POS", np.where(neg > pos, "NEG", "NEU")).astype(object)


# ----------------------------------------------------------------------------------------------------------
# Representaciones
# ----------------------------------------------------------------------------------------------------------
def _lr(c: float = 1.0, peso="balanced") -> LogisticRegression:
    return LogisticRegression(C=c, class_weight=peso, max_iter=3000, random_state=SEMILLA)


def _vec_palabras(**cfg) -> TfidfVectorizer:
    base = {"lowercase": True, "sublinear_tf": True, "ngram_range": (1, 2), "min_df": 2}
    return TfidfVectorizer(**{**base, **cfg})


def _vec_caracteres(ngram=(2, 5), min_df=3, analyzer="char_wb", max_features=60000) -> TfidfVectorizer:
    return TfidfVectorizer(lowercase=True, sublinear_tf=True, analyzer=analyzer, ngram_range=ngram, min_df=min_df, max_features=max_features)


class Representacion:
    """Ajusta vectorizador/escalador con el conjunto de ajuste y transforma ajuste y validación."""

    def __init__(self, nombre: str, variante_texto: str = "base", excluir_grupos: tuple = (), cfg_car: dict | None = None, cfg_pal: dict | None = None):
        self.nombre, self.var, self.excluir, self.cfg_car, self.cfg_pal = nombre, variante_texto, excluir_grupos, cfg_car or {}, cfg_pal or {}

    def _num_cols(self) -> list[str]:
        return F._numericas([g for g in F.GRUPOS_NUM if g not in self.excluir])

    @staticmethod
    def precalcular(*dfs: pd.DataFrame) -> None:
        """Las 24 variables dependen solo del texto de cada tuit: se calculan una vez y se reutilizan en todos los pliegues."""
        cols = F._numericas(F.GRUPOS_NUM)
        for df in dfs:
            df[cols] = F.construir_matriz(df)[cols].to_numpy()

    def ajustar_transformar(self, tr: pd.DataFrame, va: pd.DataFrame):
        t_tr, t_va = tr["texto"].map(lambda t: V.aplicar(t, self.var)), va["texto"].map(lambda t: V.aplicar(t, self.var))
        partes_tr, partes_va = [], []
        if self.nombre in ("M0", "M2"):
            v = _vec_palabras(**self.cfg_pal).fit(t_tr)
            partes_tr.append(v.transform(t_tr)); partes_va.append(v.transform(t_va))
        if self.nombre in ("M1", "M3"):
            v = _vec_caracteres(**self.cfg_car).fit(t_tr)
            partes_tr.append(v.transform(t_tr)); partes_va.append(v.transform(t_va))
        if self.nombre in ("M2", "M3", "V1") and self._num_cols():
            cols = self._num_cols()
            Mtr, Mva = tr[cols], va[cols]
            sc = StandardScaler().fit(Mtr)
            partes_tr.append(csr_matrix(sc.transform(Mtr))); partes_va.append(csr_matrix(sc.transform(Mva)))
        return hstack(partes_tr).tocsr(), hstack(partes_va).tocsr()


def pliegues(train: pd.DataFrame):
    return H.pliegues(train)


def correr_cv(train: pd.DataFrame, rep: Representacion, c: float = 1.0, peso="balanced", guardar_pred: bool = True) -> dict:
    """Una pasada de 5 pliegues. Devuelve métricas por pliegue y las predicciones fuera de muestra (OOF)."""
    y = train["etiqueta"].to_numpy()
    oof = np.empty(len(train), dtype=object)
    filas = []
    for k, (tr, va) in enumerate(pliegues(train), start=1):
        t0 = time.time()
        if rep.nombre == "V2":   # regla de léxico: no se ajusta nada
            t_ajuste = time.time() - t0
            t1 = time.time()
            p = predecir_regla(train.iloc[va])
            t_pred = time.time() - t1
            p_aj = predecir_regla(train.iloc[tr])
        else:
            Xtr, Xva = rep.ajustar_transformar(train.iloc[tr], train.iloc[va])
            m = _lr(c, peso).fit(Xtr, y[tr])
            t_ajuste = time.time() - t0
            t1 = time.time()
            p = m.predict(Xva)
            t_pred = time.time() - t1
            p_aj = m.predict(Xtr)
        oof[va] = p
        filas.append({
            "fold": k, "n_ajuste": len(tr), "n_valid": len(va),
            "macro_f1_valid": f1_score(y[va], p, average="macro"), "accuracy_valid": accuracy_score(y[va], p),
            "macro_f1_ajuste": f1_score(y[tr], p_aj, average="macro"),
            "f1_NEG": f1_score(y[va], p, labels=["NEG"], average="macro"), "f1_NEU": f1_score(y[va], p, labels=["NEU"], average="macro"),
            "f1_POS": f1_score(y[va], p, labels=["POS"], average="macro"),
            "seg_ajuste": t_ajuste, "ms_por_tuit": 1000 * t_pred / len(va),
        })
    return {"folds": pd.DataFrame(filas), "oof": oof}


def resumen(folds: pd.DataFrame, col: str = "macro_f1_valid") -> tuple[float, float]:
    return float(folds[col].mean()), float(folds[col].std(ddof=1))


def dif_pareada_oof(y, p_a, p_b, n_boot: int = 2000, seed: int = SEMILLA) -> dict:
    """Diferencia de Macro-F1 (B menos A) sobre las predicciones fuera de muestra, con IC 95% por bootstrap pareado."""
    d = F.diferencias_pareadas(y, {"A": p_a, "B": p_b}, [("A", "B")], n_boot=n_boot, seed=seed).iloc[0]
    return {"dif": d["dif_macro_f1_B_menos_A"], "ic_inf": d["ic_inf"], "ic_sup": d["ic_sup"]}


# ----------------------------------------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--out-dir", default="docs/validacion")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    out = Path(a.out_dir)
    (out / "tablas").mkdir(parents=True, exist_ok=True)
    (out / "figuras").mkdir(parents=True, exist_ok=True)
    T = out / "tablas"

    train, dev = H.cargar(a.data_dir)
    Representacion.precalcular(train, dev)
    y = train["etiqueta"].to_numpy()
    print(f"train={len(train)} dev={len(dev)}")

    # --- Particiones ---------------------------------------------------------------------------------------
    fl = pliegues(train)
    part = []
    for k, (tr, va) in enumerate(fl, start=1):
        d = train.iloc[va]
        part.append({"fold": k, "n_ajuste": len(tr), "n_valid": len(va),
                     **{f"pct_{c}": round(100 * (d.etiqueta == c).mean(), 1) for c in CLASES},
                     **{f"n_{v}": int((d.variedad == v).sum()) for v in sorted(train.variedad.unique())}})
    pd.DataFrame(part).to_csv(T / "particiones.csv", index=False)

    # --- 1. Baseline y variantes (por fold) ----------------------------------------------------------------
    res, oof = {}, {}
    for nombre in ("M0", "M1", "M2", "M3", "V1", "V2"):
        r = correr_cv(train, Representacion(nombre))
        res[nombre], oof[nombre] = r["folds"], r["oof"]
        print(nombre, "macro-F1 CV = %.4f ± %.4f" % resumen(r["folds"]))
    filas = []
    for n, f in res.items():
        m, s = resumen(f)
        gap = float((f["macro_f1_ajuste"] - f["macro_f1_valid"]).mean())
        filas.append({"modelo": n, "descripcion": MODELOS[n], **{f"fold{k}": round(v, 4) for k, v in zip(f["fold"], f["macro_f1_valid"])},
                      "media": round(m, 4), "desv": round(s, 4), "accuracy_media": round(f["accuracy_valid"].mean(), 4),
                      "f1_NEG": round(f["f1_NEG"].mean(), 4), "f1_NEU": round(f["f1_NEU"].mean(), 4), "f1_POS": round(f["f1_POS"].mean(), 4),
                      "macro_f1_ajuste": round(f["macro_f1_ajuste"].mean(), 4), "brecha_ajuste_valid": round(gap, 4),
                      "seg_ajuste_por_fold": round(f["seg_ajuste"].mean(), 2), "ms_por_tuit": round(f["ms_por_tuit"].mean(), 3)})
    comp = pd.DataFrame(filas)
    comp.to_csv(T / "cv_por_fold_baseline_variantes.csv", index=False)

    filas = []
    for b, ref_ in (("M1", "M0"), ("M2", "M0"), ("M3", "M0"), ("V1", "M0"), ("V2", "M0"), ("M3", "M1"), ("M3", "M2"), ("M1", "M2")):
        d = dif_pareada_oof(y, oof[ref_], oof[b], a.n_boot)
        porf = res[b]["macro_f1_valid"].to_numpy() - res[ref_]["macro_f1_valid"].to_numpy()
        filas.append({"comparacion": f"{b} menos {ref_}", "dif_media_folds": round(porf.mean(), 4), "folds_a_favor": int((porf > 0).sum()),
                      "dif_oof": round(d["dif"], 4), "ic_inf": round(d["ic_inf"], 4), "ic_sup": round(d["ic_sup"], 4)})
    pd.DataFrame(filas).to_csv(T / "cv_diferencias_contra_baseline.csv", index=False)

    elegido = comp[comp.modelo.isin(["M0", "M1", "M2", "M3"])].sort_values("media", ascending=False).iloc[0]["modelo"]
    print("mejor de los tres en CV:", elegido)
    with open(T / "modelo_elegido.txt", "w") as fh:
        fh.write(elegido + "\n")

    # --- Por clase y por variedad (OOF) --------------------------------------------------------------------
    filas = []
    for n in ("M0", "M1", "M2", "M3", "V1", "V2"):
        for v in ["Todas"] + sorted(train.variedad.unique()):
            m = np.ones(len(train), bool) if v == "Todas" else (train.variedad == v).to_numpy()
            r = EV.metricas(y[m], oof[n][m], a.n_boot)
            filas.append({"modelo": n, "ambito": v, "n": int(m.sum()), "macro_f1_oof": round(r["macro_f1"], 4), "ic_inf": round(r["ic_inf"], 4), "ic_sup": round(r["ic_sup"], 4)})
    pd.DataFrame(filas).to_csv(T / "cv_por_variedad_oof.csv", index=False)
    cm = pd.DataFrame(confusion_matrix(y, oof[elegido], labels=CLASES), index=[f"real_{c}" for c in CLASES], columns=[f"pred_{c}" for c in CLASES])
    cm.to_csv(T / "cv_matriz_confusion_elegido.csv")

    # --- 2. Optimización de hiperparámetros del modelo elegido --------------------------------------------
    if elegido not in ("M0", "M1", "M3"):
        sys.exit(f"El elegido es {elegido}: la búsqueda implementada es la de M0, M1 o M3.")
    if elegido == "M3":   # caracteres + 24 variables: malla propia con la misma estructura que la de H.busqueda_cv
        filas_b = []
        for ng in [(2, 3), (3, 4), (2, 5)]:
            for c_, p_ in itertools.product(H.VALORES_C, H.PESOS):
                f_ = correr_cv(train, Representacion("M3", cfg_car={"ngram": ng}), c=c_, peso=p_)["folds"]
                m_, s_ = resumen(f_)
                filas_b.append({"familia": "caracteres+variables", "ngram_range": str(ng), "min_df": 3, "C": c_, "class_weight": p_ or "ninguno", "macro_f1_cv": m_, "desv_cv": s_})
        busq = pd.DataFrame(filas_b).sort_values("macro_f1_cv", ascending=False).reset_index(drop=True)
    else:
        familia = {"M0": "palabras", "M1": "caracteres"}[elegido]
        busq = H.busqueda_cv(train, familias=(familia,), verbose=False)
    busq.round(4).to_csv(T / "optimizacion_busqueda_cv.csv", index=False)
    mejor = busq.iloc[0]
    print("mejor de la malla (etapa 1):", dict(mejor))
    peso = None if mejor["class_weight"] == "ninguno" else mejor["class_weight"]
    c_best = float(mejor["C"])
    if elegido in ("M1", "M3"):
        cfg = {"ngram": eval(mejor["ngram_range"]), "min_df": int(mejor["min_df"]), "analyzer": "char_wb", "max_features": 60000}
        # Etapa 2: refinamiento alrededor del mejor punto (tipo de n-grama, tamaño del vocabulario y min_df)
        ref = []
        for analyzer, mf, md in [("char_wb", 60000, 3), ("char_wb", 20000, 3), ("char_wb", None, 3), ("char", 60000, 3), ("char", 20000, 3), ("char", None, 3), ("char_wb", 60000, 2), ("char_wb", 60000, 5)]:
            c2 = {**cfg, "analyzer": analyzer, "max_features": mf, "min_df": md}
            f2 = correr_cv(train, Representacion(elegido, cfg_car=c2), c=c_best, peso=peso)["folds"]
            m2, s2 = resumen(f2)
            ref.append({"etapa": 2, "analyzer": analyzer, "max_features": mf if mf else "sin tope", "min_df": md, "macro_f1_cv": round(m2, 4), "desv_cv": round(s2, 4)})
        ref = pd.DataFrame(ref)
        ref.to_csv(T / "optimizacion_etapa2.csv", index=False)
        mejor2 = ref.sort_values("macro_f1_cv", ascending=False).iloc[0]
        print("mejor de la etapa 2:", dict(mejor2))
        if mejor2["macro_f1_cv"] > float(mejor["macro_f1_cv"]) + 1e-9:
            cfg = {**cfg, "analyzer": mejor2["analyzer"], "max_features": None if mejor2["max_features"] == "sin tope" else int(mejor2["max_features"]), "min_df": int(mejor2["min_df"])}
        rep_opt = Representacion(elegido, cfg_car=cfg)
        fila_cfg = {"analyzer": cfg["analyzer"], "ngram_range": str(cfg["ngram"]), "min_df": cfg["min_df"], "max_features": cfg["max_features"] or "sin tope"}
    else:
        cfg = {"ngram_range": eval(mejor["ngram_range"]), "min_df": int(mejor["min_df"])}
        ref = []
        for sub, mf, acc, binar in [(True, None, None, False), (False, None, None, False), (True, 20000, None, False), (True, 5000, None, False),
                                     (True, None, "unicode", False), (False, None, "unicode", False), (False, None, None, True), (True, 10000, "unicode", False)]:
            c2 = {**cfg, "sublinear_tf": sub, "max_features": mf, "strip_accents": acc, "binary": binar}
            f2 = correr_cv(train, Representacion("M0", cfg_pal=c2), c=c_best, peso=peso)["folds"]
            m2, s2 = resumen(f2)
            ref.append({"etapa": 2, "sublinear_tf": sub, "max_features": mf if mf else "sin tope", "strip_accents": acc or "no", "binary": binar, "macro_f1_cv": round(m2, 4), "desv_cv": round(s2, 4)})
        ref = pd.DataFrame(ref)
        ref.to_csv(T / "optimizacion_etapa2.csv", index=False)
        cfg = {**cfg, "sublinear_tf": True, "max_features": None, "strip_accents": None, "binary": False}
        rep_opt = Representacion("M0", cfg_pal=cfg)
        fila_cfg = {"ngram_range": str(cfg["ngram_range"]), "min_df": cfg["min_df"]}
    r_opt = correr_cv(train, rep_opt, c=c_best, peso=peso)
    res["opt"], oof["opt"] = r_opt["folds"], r_opt["oof"]
    m_o, s_o = resumen(r_opt["folds"])
    d = dif_pareada_oof(y, oof[elegido], oof["opt"], a.n_boot)
    porf = r_opt["folds"]["macro_f1_valid"].to_numpy() - res[elegido]["macro_f1_valid"].to_numpy()
    pd.DataFrame([{
        "configuracion": "optimizada", **fila_cfg, "C": c_best, "class_weight": mejor["class_weight"],
        **{f"fold{k}": round(v, 4) for k, v in zip(r_opt["folds"]["fold"], r_opt["folds"]["macro_f1_valid"])},
        "media": round(m_o, 4), "desv": round(s_o, 4), "macro_f1_ajuste": round(r_opt["folds"]["macro_f1_ajuste"].mean(), 4),
        "dif_contra_por_defecto": round(d["dif"], 4), "ic_inf": round(d["ic_inf"], 4), "ic_sup": round(d["ic_sup"], 4),
        "folds_a_favor": int((porf > 0).sum()), "seg_ajuste_por_fold": round(r_opt["folds"]["seg_ajuste"].mean(), 2),
    }]).to_csv(T / "optimizacion_resultado_cv.csv", index=False)
    # efecto marginal de cada hiperparámetro (mejor valor alcanzable con ese valor)
    marg = []
    for col in ("C", "class_weight", "ngram_range", "min_df"):
        for val, g in busq.groupby(col):
            marg.append({"hiperparametro": col, "valor": val, "mejor_macro_f1_cv": round(g.macro_f1_cv.max(), 4)})
    pd.DataFrame(marg).to_csv(T / "optimizacion_efecto_marginal.csv", index=False)

    # --- 3. Ablaciones (por fold, con el modelo elegido) --------------------------------------------------
    ab = []
    def registrar(componente, control, nombre_cfg, folds_cfg, oof_cfg, folds_ctrl, oof_ctrl):
        dd = dif_pareada_oof(y, oof_ctrl, oof_cfg, a.n_boot)
        m, s = resumen(folds_cfg)
        porf = folds_cfg["macro_f1_valid"].to_numpy() - folds_ctrl["macro_f1_valid"].to_numpy()
        ab.append({"componente": componente, "configuracion": nombre_cfg, "control": control,
                   **{f"fold{k}": round(v, 4) for k, v in zip(folds_cfg["fold"], folds_cfg["macro_f1_valid"])},
                   "media": round(m, 4), "desv": round(s, 4), "dif_contra_control": round(dd["dif"], 4), "ic_inf": round(dd["ic_inf"], 4),
                   "ic_sup": round(dd["ic_sup"], 4), "folds_a_favor": int((porf > 0).sum())})

    # 3a. variantes de los datos (modelo elegido, hiperparámetros por defecto)
    ctrl_f, ctrl_o = res[elegido], oof[elegido]
    for var in ["limpio", "minusculas", "sin_tildes", "sin_alargamientos", "sin_emojis", "sin_usuario_url"]:
        r = correr_cv(train, Representacion(elegido, variante_texto=var))
        registrar("Texto de entrada", "base", var, r["folds"], r["oof"], ctrl_f, ctrl_o)
        print("variante de datos", var, "%.4f" % resumen(r["folds"])[0])
    # 3b. grupos de variables en M2 (palabras + 24 variables)
    for mm in ("M2", "M3"):
        for g in F.GRUPOS_NUM:
            r = correr_cv(train, Representacion(mm, excluir_grupos=(g,)))
            registrar(f"Grupo de variables (en {mm})", f"{mm} completo", f"{mm} sin {g}", r["folds"], r["oof"], res[mm], oof[mm])
        r = correr_cv(train, Representacion(mm, excluir_grupos=tuple(F.GRUPOS_NUM)))
        registrar(f"Grupo de variables (en {mm})", f"{mm} completo", f"{mm} sin ninguna variable", r["folds"], r["oof"], res[mm], oof[mm])
    # 3c. solo un grupo de variables (modelo de 24 variables)
    for g in F.GRUPOS_NUM:
        r = correr_cv(train, Representacion("V1", excluir_grupos=tuple(x for x in F.GRUPOS_NUM if x != g)))
        registrar("Solo un grupo (modelo de variables)", "V1: 24 variables", f"solo {g}", r["folds"], r["oof"], res["V1"], oof["V1"])
    # 3d. peso de clases en el modelo elegido
    r = correr_cv(train, Representacion(elegido), peso=None)
    registrar("Peso de clases", "balanceado", "sin balancear", r["folds"], r["oof"], ctrl_f, ctrl_o)
    # 3e. hiperparámetros optimizados contra los de defecto
    registrar("Hiperparámetros", "por defecto", "optimizados", res["opt"], oof["opt"], ctrl_f, ctrl_o)
    pd.DataFrame(ab).to_csv(T / "ablaciones_cv.csv", index=False)

    # --- 4. Curva de aprendizaje del modelo optimizado ---------------------------------------------------
    rng = np.random.default_rng(SEMILLA)
    filas = []
    for frac in (0.2, 0.4, 0.6, 0.8, 1.0):
        vt, vv = [], []
        for k, (tr, va) in enumerate(pliegues(train), start=1):
            sub = np.sort(rng.choice(tr, size=int(len(tr) * frac), replace=False)) if frac < 1 else tr
            Xtr, Xva = rep_opt.ajustar_transformar(train.iloc[sub], train.iloc[va])
            m = _lr(c_best, peso).fit(Xtr, y[sub])
            vt.append(f1_score(y[sub], m.predict(Xtr), average="macro"))
            vv.append(f1_score(y[va], m.predict(Xva), average="macro"))
        filas.append({"fraccion_train": frac, "n_ajuste_aprox": int(len(train) * 0.8 * frac), "macro_f1_ajuste": round(np.mean(vt), 4), "macro_f1_valid": round(np.mean(vv), 4), "desv_valid": round(np.std(vv, ddof=1), 4)})
    pd.DataFrame(filas).to_csv(T / "curva_aprendizaje.csv", index=False)

    # --- 5. Medición única en dev -------------------------------------------------------------------------
    yd = dev["etiqueta"].to_numpy()
    preds = {}
    preds["V2"] = predecir_regla(dev)
    for n, rep, c, p_ in (("M0", Representacion("M0"), 1.0, "balanced"), ("M1", Representacion("M1"), 1.0, "balanced"), ("V1", Representacion("V1"), 1.0, "balanced"),
                          ("M2", Representacion("M2"), 1.0, "balanced"), ("M3", Representacion("M3"), 1.0, "balanced"), ("opt", rep_opt, c_best, peso)):
        t0 = time.time()
        Xtr, Xdv = rep.ajustar_transformar(train, dev)
        m = _lr(c, p_).fit(Xtr, y)
        preds[n] = m.predict(Xdv)
        preds[n + "_seg"] = time.time() - t0
    # Salamandra monoagente (B1), mismos tuits de dev
    llm = pd.read_csv(AQUI.parents[1] / "docs/experimento/llm/predicciones_llm.csv", dtype=str, keep_default_na=False)
    llm = llm[llm.variante == "base"].drop_duplicates("id").set_index("id")
    pl = llm["pred"].where(llm["pred"].isin(CLASES), "NEU")
    preds["LLM"] = pl.reindex(dev["id"]).to_numpy()
    if pd.isna(preds["LLM"]).any():
        sys.exit("Hay tuits de dev sin predicción del LLM.")
    filas, difs = [], []
    nombres = {"M0": "M0 baseline", "M1": "M1 caracteres", "M2": "M2 palabras + 24 variables", "M3": "M3 caracteres + 24 variables", "V1": "V1 24 variables", "V2": "V2 regla de léxico", "opt": f"{elegido} optimizado", "LLM": "Salamandra monoagente (B1)"}
    for n in ("M0", "M1", "M2", "M3", "V1", "V2", "opt", "LLM"):
        for amb, mk in EV.ambitos(dev["variedad"]).items():
            r = EV.metricas(yd[mk], preds[n][mk], a.n_boot)
            filas.append({"modelo": nombres[n], "ambito": amb, **{k: round(v, 4) for k, v in r.items()}})
    pd.DataFrame(filas).to_csv(T / "dev_resultados.csv", index=False)
    for n in ("M1", "M2", "M3", "V1", "V2", "opt", "LLM"):
        for amb, mk in EV.ambitos(dev["variedad"]).items():
            dd = F.diferencias_pareadas(yd[mk], {"A": preds["M0"][mk], "B": preds[n][mk]}, [("A", "B")], n_boot=a.n_boot, seed=SEMILLA).iloc[0]
            difs.append({"comparacion": f"{nombres[n]} menos M0", "ambito": amb, "dif": round(dd["dif_macro_f1_B_menos_A"], 4), "ic_inf": round(dd["ic_inf"], 4), "ic_sup": round(dd["ic_sup"], 4), "conclusion": dd["conclusion"]})
    for amb, mk in EV.ambitos(dev["variedad"]).items():
        dd = F.diferencias_pareadas(yd[mk], {"A": preds["opt"][mk], "B": preds["LLM"][mk]}, [("A", "B")], n_boot=a.n_boot, seed=SEMILLA).iloc[0]
        difs.append({"comparacion": f"Salamandra B1 menos {elegido} optimizado", "ambito": amb, "dif": round(dd["dif_macro_f1_B_menos_A"], 4), "ic_inf": round(dd["ic_inf"], 4), "ic_sup": round(dd["ic_sup"], 4), "conclusion": dd["conclusion"]})
    pd.DataFrame(difs).to_csv(T / "dev_diferencias.csv", index=False)
    f1c = []
    for n in ("M0", "M1", "M2", "M3", "V1", "V2", "opt", "LLM"):
        f = f1_score(yd, preds[n], labels=CLASES, average=None, zero_division=0)
        f1c.append({"modelo": nombres[n], **{f"F1_{c}": round(v, 4) for c, v in zip(CLASES, f)}})
    pd.DataFrame(f1c).to_csv(T / "dev_f1_por_clase.csv", index=False)

    # --- 6. Registro de corridas (Anexo D) ----------------------------------------------------------------
    h = hashlib.sha1()
    for p in sorted(Path(a.data_dir).glob("*/*.tsv")):
        h.update(p.read_bytes())
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        commit = "sin git"
    reg = []
    for i, (n, f) in enumerate({**res}.items(), start=1):
        m, _ = resumen(f)
        reg.append({"run_id": f"cv-{n}", "commit": commit, "datos_sha1": h.hexdigest()[:10], "seed_split": f"semilla {SEMILLA} · 5 pliegues estratificados (etiqueta, variedad)", "metrica": f"Macro-F1 CV {m:.4f}", "artefacto": "docs/validacion/tablas/cv_por_fold_baseline_variantes.csv" if n in MODELOS else "docs/validacion/tablas/optimizacion_resultado_cv.csv"})
    pd.DataFrame(reg).to_csv(T / "registro_corridas.csv", index=False)

    # --- Figuras ----------------------------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ancho = 0.16
    for i, (n, col) in enumerate((("M0", GRIS), ("M1", AZUL), ("M2", VERDE), ("M3", "#e0a030"), ("opt", ROJO))):
        ax.bar(np.arange(1, 6) + (i - 2) * ancho, res[n]["macro_f1_valid"], ancho, label={"opt": f"{elegido} optimizado"}.get(n, n), color=col)
    ax.set_xlabel("Pliegue"); ax.set_ylabel("Macro-F1 de validación"); ax.set_ylim(0.45, 0.62); ax.set_xticks(range(1, 6))
    ax.set_title("Macro-F1 en cada pliegue"); ax.legend(ncol=4, fontsize=8, loc="upper center"); plt.tight_layout(); plt.savefig(out / "figuras" / "fig01_folds.png", dpi=170); plt.close()

    ab_df = pd.DataFrame(ab)
    sel = ab_df[ab_df.componente.isin(["Texto de entrada", "Grupo de variables (en M2)", "Peso de clases", "Hiperparámetros"])].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 0.34 * len(sel) + 1.2))
    ax.barh(range(len(sel)), sel["dif_contra_control"], xerr=[sel["dif_contra_control"] - sel["ic_inf"], sel["ic_sup"] - sel["dif_contra_control"]], color=AZUL, capsize=2)
    ax.axvline(0, color="black", lw=0.8); ax.set_yticks(range(len(sel)), sel["configuracion"]); ax.invert_yaxis()
    ax.set_xlabel("Cambio en Macro-F1 contra su control (IC 95%, predicciones fuera de muestra)"); ax.set_title("Ablaciones principales")
    plt.tight_layout(); plt.savefig(out / "figuras" / "fig02_ablaciones.png", dpi=170); plt.close()

    cu = pd.read_csv(T / "curva_aprendizaje.csv")
    fig, ax = plt.subplots(figsize=(6.2, 3.8))
    ax.plot(cu.n_ajuste_aprox, cu.macro_f1_ajuste, marker="o", color=GRIS, label="ajuste")
    ax.errorbar(cu.n_ajuste_aprox, cu.macro_f1_valid, yerr=cu.desv_valid, marker="o", color=AZUL, capsize=3, label="validación")
    ax.set_xlabel("Tuits de ajuste"); ax.set_ylabel("Macro-F1"); ax.set_title("Curva de aprendizaje (modelo optimizado)"); ax.legend()
    plt.tight_layout(); plt.savefig(out / "figuras" / "fig03_curva_aprendizaje.png", dpi=170); plt.close()
    print("listo")


if __name__ == "__main__":
    main()
