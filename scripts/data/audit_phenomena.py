"""
Auditoría previa del corpus: cuenta, con heurísticas simples (no con un modelo),
cuántos tuits del conjunto de desarrollo contienen indicios de los cuatro fenómenos
lingüísticos que motivan la especialización del agente de Percepción:

    - diminutivo afectivo (terminaciones -ito/-ita/-itos/-itas)
    - doble negación (dos marcadores de negación en la misma oración)
    - code-switching (mezcla aparente de palabras en inglés dentro del texto)
    - sarcasmo / ironía (marcadores superficiales: risas, signos de exclamación
      repetidos junto a emojis invertidos, expresiones irónicas típicas)

Esto NO reemplaza una anotación manual por expertos (que sigue siendo necesaria
para contrastar la hipótesis específica sobre fenómenos lingüísticos). Sirve para
decidir, antes de diseñar los prompts, si hay evidencia mínima de cada fenómeno en
el corpus o si conviene replantear esa hipótesis como exploratoria.

Uso:
    python scripts/data/audit_phenomena.py \
        --input data/processed/dev.csv \
        --out data/processed/auditoria_fenomenos.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import regex as re

sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.utils.io_utils import get_logger, read_table_any, write_csv  # noqa: E402

logger = get_logger("audit_phenomena")

DIMINUTIVO_RE = re.compile(r"\b\w+(ito|ita|itos|itas|cito|cita)\b", re.IGNORECASE)
NEGACION_RE = re.compile(r"\b(no|nunca|jamás|nada|nadie|ni)\b", re.IGNORECASE)
INGLES_COMUN_RE = re.compile(
    r"\b(the|and|but|so|like|just|really|love|please|thanks|omg|lol|work|love it)\b",
    re.IGNORECASE,
)
SARCASMO_RE = re.compile(r"(jaja+|jeje+|🙃|😏|😑|claro que sí|por supuesto\.{3}|qué bien\.{3})", re.IGNORECASE)


def contar_fenomenos(texto: str) -> dict:
    texto = texto or ""
    negaciones = len(NEGACION_RE.findall(texto))
    return {
        "diminutivo_afectivo": bool(DIMINUTIVO_RE.search(texto)),
        "doble_negacion": negaciones >= 2,
        "code_switching": bool(INGLES_COMUN_RE.search(texto)),
        "sarcasmo_aparente": bool(SARCASMO_RE.search(texto)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV con columna 'texto' y 'variedad' (ej. dev.csv).")
    parser.add_argument("--out", required=True, help="Ruta del reporte de auditoría en CSV.")
    args = parser.parse_args()

    df = read_table_any(args.input)
    flags = df["texto"].apply(contar_fenomenos).apply(pd.Series)
    df_flags = pd.concat([df[["id", "variedad"]], flags], axis=1)

    resumen = df_flags.groupby("variedad")[
        ["diminutivo_afectivo", "doble_negacion", "code_switching", "sarcasmo_aparente"]
    ].sum()
    total = df_flags[["diminutivo_afectivo", "doble_negacion", "code_switching", "sarcasmo_aparente"]].sum()

    logger.info("Conteo aproximado de fenómenos por variedad (heurístico, no definitivo):\n%s", resumen.to_string())
    logger.info("Total general:\n%s", total.to_string())
    logger.warning(
        "Estos conteos son heurísticos y sobreestiman o subestiman según el caso. "
        "Si algún fenómeno aparece en muy pocos casos, conviene replantear su hipótesis "
        "específica como exploratoria antes de diseñar un prompt dedicado a él."
    )

    write_csv(df_flags, args.out)
    logger.info("Reporte detallado guardado en %s", args.out)


if __name__ == "__main__":
    main()
