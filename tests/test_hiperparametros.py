import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "experimento"))
import optimizar_hiperparametros as H  # noqa: E402


def _mini():
    rng = np.random.default_rng(0)
    pos = ["me encanta esto", "qué bueno todo", "excelente día", "muy feliz hoy"]
    neg = ["odio esto", "qué malo todo", "pésimo día", "muy triste hoy"]
    neu = ["salgo a las ocho", "la reunión es mañana", "hay clase hoy", "nos vemos luego"]
    filas = []
    for i in range(60):
        e, t = [("POS", pos), ("NEG", neg), ("NEU", neu)][i % 3]
        filas.append({"texto": rng.choice(t) + f" {i}", "etiqueta": e, "variedad": ["ES", "MX"][i % 2]})
    return pd.DataFrame(filas)


def test_pliegues_cubren_todo_sin_solaparse():
    df = _mini()
    folds = H.pliegues(df)
    assert len(folds) == H.N_PLIEGUES
    todos = np.concatenate([va for _, va in folds])
    assert sorted(todos) == list(range(len(df)))
    for tr, va in folds:
        assert not set(tr) & set(va)


def test_mejores_por_familia_toma_el_maximo():
    res = pd.DataFrame({"familia": ["a", "a", "b"], "macro_f1_cv": [0.5, 0.6, 0.4]})
    m = H.mejores_por_familia(res)
    assert dict(zip(m.familia, m.macro_f1_cv)) == {"a": 0.6, "b": 0.4}
