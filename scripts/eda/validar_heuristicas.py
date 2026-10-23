"""
Calcula la precisión de las heurísticas de fenómenos a partir de la muestra anotada a mano.

Pasos:
  1. Abrir docs/eda/tablas/muestra_validar_heuristicas.csv (Excel o similar).
  2. En la columna "es_correcto (SI/NO)" escribir SI si la marca es correcta (el tuit sí tiene el
     fenómeno) y NO si no lo es. Guardar como CSV (UTF-8).
  3. Correr:
        python scripts/eda/validar_heuristicas.py
     o con otra ruta:
        python scripts/eda/validar_heuristicas.py --input mi_muestra.csv

Salida: docs/eda/tablas/precision_heuristicas.csv y un resumen por pantalla.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))
import eda_extra as X  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="docs/eda/tablas/muestra_validar_heuristicas.csv")
    ap.add_argument("--out", default="docs/eda/tablas/precision_heuristicas.csv")
    a = ap.parse_args()

    muestra = pd.read_csv(a.input, encoding="utf-8")
    res = X.precision_heuristicas(muestra)
    if res.empty:
        sys.exit(f'No hay filas anotadas en la columna "{X.COL_VALIDACION}". Escribe SI o NO en cada fila y vuelve a correr.')
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    res.to_csv(a.out, index=False, encoding="utf-8")
    print(res.to_string(index=False))
    print(f"\nGuardado en {a.out}")


if __name__ == "__main__":
    main()
