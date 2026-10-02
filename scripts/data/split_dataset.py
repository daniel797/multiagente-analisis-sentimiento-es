"""
Genera la partición de desarrollo / validación / prueba bloqueada, estratificada por
variedad dialectal y por etiqueta, para evitar que el conjunto de prueba se use por
error durante el diseño de los prompts o la selección del umbral de confianza.

Protocolo (documentado en el avance de la Semana 2):
    - desarrollo: se usa para analizar errores del monoagente y diseñar los prompts
      de los agentes especializados.
    - validación: se usa para fijar el umbral de confianza (tau) y max_iter del
      meta-agente.
    - prueba: queda bloqueada y solo se usa una vez, para la evaluación final de
      todas las condiciones (B1, B3, B4, B5, B6).

Uso:
    python scripts/data/split_dataset.py \
        --input data/processed/tass2020_consolidado.csv \
        --out-dir data/processed \
        --seed 42
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.utils.io_utils import get_logger, read_table_any, write_csv  # noqa: E402

logger = get_logger("split_dataset")

DEV_SIZE = 0.60
VAL_SIZE = 0.15
TEST_SIZE = 0.25  # dev + val + test = 1.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV consolidado (salida de consolidate_dataset.py).")
    parser.add_argument("--out-dir", required=True, help="Carpeta donde se guardan dev.csv, val.csv y test.csv.")
    parser.add_argument("--seed", type=int, default=42, help="Semilla para que la partición sea reproducible.")
    args = parser.parse_args()

    df = read_table_any(args.input)
    df = df.dropna(subset=["etiqueta"]).reset_index(drop=True)

    # Estrato combinado: variedad + etiqueta, para que cada partición mantenga
    # proporciones similares de país y de polaridad.
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
