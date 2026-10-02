"""
Consolida los archivos crudos de TASS 2020 (uno por variedad dialectal) en un único
archivo tabular con columnas estandarizadas: id, texto, variedad, etiqueta.

Este script NO descarga el corpus: el corpus se obtiene aparte, con registro y
licencia, en tass.sepln.org (ver README.md). Lo que hace es tomar lo que el
usuario ya descargó y colocó en data/raw/<VARIEDAD>/ y dejarlo en un formato
único y predecible para los scripts siguientes.

Uso:
    python scripts/data/consolidate_dataset.py \
        --raw-dir data/raw \
        --out data/processed/tass2020_consolidado.csv

Estructura esperada de entrada (ejemplo):
    data/raw/
        PE/  archivo(s) .xml o .tsv de InterTASS PE-Perú
        MX/  archivo(s) .xml o .tsv de la variante MX
        ES/  ...
        CL/  ...
        CR/  ...
        UY/  ...

Nota sobre el esquema de etiquetas: ediciones previas de InterTASS usan
P / N / NEU / NONE; TASS 2020 documenta un esquema de tres clases donde NEU
incorpora los textos sin polaridad clara. Este script normaliza ambos casos
a {POS, NEG, NEU} y dejan constancia del mapeo aplicado en los logs, para que
quede documentado qué transformación se usó (correction del plan: fijar el
esquema de etiquetas exactamente).
"""
from __future__ import annotations

import argparse
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[2]))
from scripts.utils.io_utils import get_logger, read_table_any, write_csv  # noqa: E402

logger = get_logger("consolidate_dataset")

# Mapeo de esquemas de etiquetas antiguos al esquema de tres clases usado en este proyecto.
# Ajustar aquí si la versión descargada trae otro esquema; dejar el cambio documentado.
LABEL_MAP = {
    "P": "POS", "POS": "POS", "POSITIVE": "POS",
    "N": "NEG", "NEG": "NEG", "NEGATIVE": "NEG",
    "NEU": "NEU", "NEUTRAL": "NEU",
    "NONE": "NEU",  # NONE se incorpora a NEU, siguiendo el esquema de tres clases de TASS 2020
}


def parse_xml_tweets(path: Path) -> pd.DataFrame:
    """Parsea el formato XML clásico de InterTASS (<tweets><tweet><tweetid>...)."""
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for tweet in root.findall(".//tweet"):
        tid = tweet.findtext("tweetid", default="")
        content = tweet.findtext("content", default="")
        sentiment_node = tweet.find("sentiment")
        label_raw = ""
        if sentiment_node is not None:
            polarity_node = sentiment_node.find("polarity")
            if polarity_node is not None:
                label_raw = (polarity_node.findtext("value") or "").strip().upper()
        rows.append({"id": tid, "texto": content, "etiqueta_raw": label_raw})
    return pd.DataFrame(rows)


def load_variety_folder(folder: Path, variedad: str) -> pd.DataFrame:
    """Carga y concatena todos los archivos reconocidos dentro de la carpeta de una variedad."""
    frames = []
    for file in sorted(folder.iterdir()):
        if file.suffix.lower() == ".xml":
            df = parse_xml_tweets(file)
        elif file.suffix.lower() in (".tsv", ".csv", ".xlsx", ".xls"):
            df = read_table_any(file)
            # Se esperan columnas 'id'/'tweetid', 'texto'/'content' y 'etiqueta'/'polarity' bajo
            # distintos nombres según la fuente; se normalizan los más comunes.
            rename_map = {
                "tweetid": "id", "content": "texto", "text": "texto",
                "polarity": "etiqueta_raw", "label": "etiqueta_raw", "sentiment": "etiqueta_raw",
                "etiqueta": "etiqueta_raw",
            }
            df = df.rename(columns={c: rename_map.get(c.lower(), c) for c in df.columns})
            if "etiqueta_raw" not in df.columns:
                logger.warning("No se encontró columna de etiqueta en %s, revisar manualmente.", file.name)
                df["etiqueta_raw"] = ""
        else:
            continue
        frames.append(df)

    if not frames:
        logger.warning("No se encontraron archivos reconocidos en %s", folder)
        return pd.DataFrame(columns=["id", "texto", "etiqueta_raw"])

    consolidado = pd.concat(frames, ignore_index=True)
    consolidado["variedad"] = variedad
    return consolidado


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", required=True, help="Carpeta con una subcarpeta por variedad (PE, MX, ES, CL, CR, UY).")
    parser.add_argument("--out", required=True, help="Ruta del CSV consolidado de salida.")
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    if not raw_dir.exists():
        logger.error("No existe %s. Coloca ahí el corpus descargado de TASS 2020 (ver README.md).", raw_dir)
        sys.exit(1)

    variedades = [p for p in raw_dir.iterdir() if p.is_dir()]
    if not variedades:
        logger.error("No hay subcarpetas de variedad dentro de %s (se esperaba PE/, MX/, ES/, etc.).", raw_dir)
        sys.exit(1)

    todas = []
    for carpeta in variedades:
        variedad = carpeta.name.upper()
        logger.info("Procesando variedad %s desde %s", variedad, carpeta)
        df = load_variety_folder(carpeta, variedad)
        logger.info("  %d filas leídas para %s", len(df), variedad)
        todas.append(df)

    consolidado = pd.concat(todas, ignore_index=True)

    antes = consolidado["etiqueta_raw"].astype(str).str.upper().str.strip()
    consolidado["etiqueta"] = antes.map(LABEL_MAP)
    sin_mapear = consolidado["etiqueta"].isna().sum()
    if sin_mapear:
        logger.warning("%d filas con etiqueta no reconocida, quedarán como NaN y conviene revisarlas.", sin_mapear)

    consolidado = consolidado[["id", "texto", "variedad", "etiqueta"]]
    logger.info("Total consolidado: %d filas de %d variedades.", len(consolidado), consolidado["variedad"].nunique())
    logger.info("Distribución por variedad:\n%s", consolidado["variedad"].value_counts().to_string())
    logger.info("Distribución por etiqueta:\n%s", consolidado["etiqueta"].value_counts(dropna=False).to_string())

    write_csv(consolidado, args.out)
    logger.info("Guardado en %s", args.out)


if __name__ == "__main__":
    main()
