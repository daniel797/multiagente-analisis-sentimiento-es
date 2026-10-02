"""
Funciones compartidas de lectura/escritura y logging para los scripts de datos.

No depende de ningún modelo ni de la data real: solo maneja archivos.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd


def get_logger(name: str) -> logging.Logger:
    """Logger simple con formato consistente para todos los scripts."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s", "%H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def ensure_dir(path: str | Path) -> Path:
    """Crea el directorio si no existe y lo devuelve como Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_table_any(path: str | Path) -> pd.DataFrame:
    """
    Lee CSV, TSV o XLSX según la extensión del archivo.
    Pensado para los distintos formatos en que puede llegar cada variedad de TASS.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".tsv":
        return pd.read_csv(path, sep="\t", encoding="utf-8")
    if suffix in (".csv",):
        return pd.read_csv(path, encoding="utf-8")
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Formato no soportado todavía: {path.name}")


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    df.to_csv(path, index=False, encoding="utf-8")
