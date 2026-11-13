"""Genera la partición de desarrollo / validación / prueba bloqueada, estratificada por"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.utils.io_utils import get_logger, read_table_any, write_csv

logger = get_logger("split_dataset")

DEV_SIZE = 0.60
VAL_SIZE = 0.15
TEST_SIZE = 0.25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV consolidado (salida de consolidate_dataset.py).")
    parser.add_argument("--out-dir", required=True, help="Carpeta donde se guardan dev.csv, val.csv y test.csv.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para que la partición sea reproducible.")
    args = parser.parse_args()

    df = read_table_any(args.input)
    df = df.dropna(subset=["etiqueta"]).reset_index(drop=True)


    estrato = df["variedad"].astype(str) + "_" + df["etiqueta"].astype(str)

    dev, resto = train_test_split(
        df, train_size=DEV_SIZE, stratify=estrato, random_state=args.seed
    )
    estrato_resto = resto["variedad"].astype(str) + "_" + resto["etiqueta"].astype(str)
    val, test = train_test_split(
        resto,
        train_size=VAL_SIZE / (VAL_SIZE + TEST_SIZE),
        stratify=estrato_resto,
        random_state=args.seed,
    )

    logger.info("Desarrollo: %d filas", len(dev))
    logger.info("Validación: %d filas", len(val))
    logger.info("Prueba (bloqueada): %d filas", len(test))

    for nombre, parte in (("dev", dev), ("val", val), ("test", test)):
        logger.info("  %s por variedad:\n%s", nombre, parte["variedad"].value_counts().to_string())

    write_csv(dev, Path(args.out_dir) / "dev.csv")
    write_csv(val, Path(args.out_dir) / "val.csv")
    write_csv(test, Path(args.out_dir) / "test.csv")

    logger.warning(
        "El archivo test.csv queda bloqueado a partir de aquí: no debe inspeccionarse "
        "ni usarse para ajustar prompts o umbrales, solo para la evaluación final."
    )


if __name__ == "__main__":
    main()
