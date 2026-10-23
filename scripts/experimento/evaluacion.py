"""
Evaluación común del experimento: Macro-F1 con intervalo por bootstrap, por variedad y en conjunto,
y diferencia pareada entre una variante y la línea base (mismos tuits).

La usan el baseline clásico (run_baseline_variantes.py) y el monoagente con LLM
(monoagente_salamandra.py), así las cifras de los dos son comparables.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).resolve().parents[1] / "features"))
import feature_utils as F  # noqa: E402


def metricas(y, p, n_boot: int = 2000, seed: int = 0) -> dict:
    """Macro-F1 (con IC 95% por bootstrap) y accuracy. `y` y `p` son listas de 'POS', 'NEG' o 'NEU'."""
    y, p = np.asarray(y), np.asarray(p)
    yc, pc = F._codificar(y), F._codificar(p)
    rng = np.random.default_rng(seed)
    vals = [F._macro_f1_rapido(yc[i], pc[i]) for i in (rng.integers(0, len(yc), len(yc)) for _ in range(n_boot))]
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return {"n_dev": len(y), "accuracy": accuracy_score(y, p), "macro_f1": F._macro_f1_rapido(yc, pc), "ic_inf": lo, "ic_sup": hi}


def ambitos(variedades: pd.Series) -> dict:
    """Máscaras booleanas: 'Todas' y una por variedad."""
    out = {"Todas": np.ones(len(variedades), dtype=bool)}
    for v in sorted(variedades.unique()):
        out[v] = (variedades == v).to_numpy()
    return out


def comparar_con_base(y, pred_base, pred_var, variedades: pd.Series, nombre: str, n_boot: int = 2000) -> list[dict]:
    """Diferencia de Macro-F1 (variante menos base) por ámbito, con IC pareado y cuántas predicciones cambian."""
    y, pred_base, pred_var = np.asarray(y), np.asarray(pred_base), np.asarray(pred_var)
    filas = []
    for amb, m in ambitos(variedades).items():
        d = F.diferencias_pareadas(y[m], {"base": pred_base[m], nombre: pred_var[m]}, [("base", nombre)], n_boot=n_boot).iloc[0]
        filas.append({
            "variante": nombre, "ambito": amb, "n_dev": int(m.sum()),
            "predicciones_distintas": int((pred_base[m] != pred_var[m]).sum()),
            "variante_acierta_y_base_no": int(((pred_var[m] == y[m]) & (pred_base[m] != y[m])).sum()),
            "base_acierta_y_variante_no": int(((pred_base[m] == y[m]) & (pred_var[m] != y[m])).sum()),
            "dif_macro_f1": d["dif_macro_f1_B_menos_A"], "ic_inf": d["ic_inf"], "ic_sup": d["ic_sup"],
            "p_bootstrap": d["p_bootstrap"], "conclusion": d["conclusion"],
        })
    return filas
