"""
Variantes de los DATOS de entrada para el experimento.

El experimento compara el mismo sistema (hoy un clasificador de referencia, luego el monoagente con
Salamandra-7B-Instruct) cambiando SOLO el texto que recibe. Hay una línea base y una variante principal:

- base:   el tuit como lo ve el LLM hoy (`normalizar_para_llm`): @USUARIO, URL, espacios colapsados.
          Conserva mayúsculas, tildes, alargamientos, emojis y puntuación.
- limpio: la variante principal. Minúsculas, sin tildes (la ñ se conserva), sin letras alargadas
          y sin emojis. Mide cuánto del resultado depende de la información de "tono" del texto.

Además hay variantes de un solo cambio, que sirven para saber qué componente de `limpio` pesa.
Las mismas funciones se usan para el clasificador de referencia y para el LLM, así las variantes
son idénticas en los dos casos.
"""
from __future__ import annotations

import sys
import unicodedata
from pathlib import Path

import regex as re

sys.path.append(str(Path(__file__).resolve().parents[1] / "features"))
import feature_utils as F  # noqa: E402

_ALARGA = re.compile(r"(\p{L})\1{2,}")
_EMOJI = re.compile(r"[\p{Extended_Pictographic}‍️]")
_USUARIO_URL = re.compile(r"@USUARIO|\bURL\b")
_ESPACIOS = re.compile(r"\s+")


def _colapsar(t: str) -> str:
    return _ESPACIOS.sub(" ", t).strip()


def base(texto: str) -> str:
    return F.normalizar_para_llm(texto)


def minusculas(texto: str) -> str:
    return base(texto).lower()


def _quitar_tildes(t: str) -> str:
    """Quita tildes y diéresis, pero conserva la ñ (en 'año' la ñ cambia el significado)."""
    t = t.replace("ñ", "\0n").replace("Ñ", "\0N")
    t = "".join(c for c in unicodedata.normalize("NFD", t) if unicodedata.category(c) != "Mn")
    return t.replace("\0n", "ñ").replace("\0N", "Ñ")


def sin_tildes(texto: str) -> str:
    return _quitar_tildes(base(texto))


def sin_alargamientos(texto: str) -> str:
    """'holaaaa' pasa a 'hola'. Solo toca letras repetidas 3 o más veces."""
    return _ALARGA.sub(r"\1", base(texto))


def sin_emojis(texto: str) -> str:
    return _colapsar(_EMOJI.sub("", base(texto)))


def limpio(texto: str) -> str:
    t = sin_emojis(texto)
    t = _ALARGA.sub(r"\1", t)
    return _colapsar(_quitar_tildes(t).lower())


def sin_usuario_url(texto: str) -> str:
    """Extra: quita los marcadores @USUARIO y URL. Sirve para ver cuánto pesa el rasgo conversacional."""
    return _colapsar(_USUARIO_URL.sub("", base(texto)))


VARIANTES = {
    "base": base,
    "limpio": limpio,
    "minusculas": minusculas,
    "sin_tildes": sin_tildes,
    "sin_alargamientos": sin_alargamientos,
    "sin_emojis": sin_emojis,
    "sin_usuario_url": sin_usuario_url,
}

DESCRIPCION = {
    "base": "Tuit como lo ve el LLM (@USUARIO, URL, espacios colapsados)",
    "limpio": "Minúsculas + sin tildes + sin alargamientos + sin emojis",
    "minusculas": "Solo minúsculas",
    "sin_tildes": "Solo sin tildes (conserva ñ)",
    "sin_alargamientos": "Solo sin letras alargadas",
    "sin_emojis": "Solo sin emojis",
    "sin_usuario_url": "Extra: sin los marcadores @USUARIO y URL",
}

COMPONENTES = ["minusculas", "sin_tildes", "sin_alargamientos", "sin_emojis", "sin_usuario_url"]


def aplicar(texto: str, variante: str) -> str:
    if variante not in VARIANTES:
        raise ValueError(f"Variante desconocida: {variante}. Opciones: {sorted(VARIANTES)}")
    return VARIANTES[variante](texto)
