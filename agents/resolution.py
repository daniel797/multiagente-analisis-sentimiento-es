"""
Agente de Resolución (llamadas 3 en adelante al LLM).

Responsabilidad: convertir la hipótesis de Razonamiento en una etiqueta candidata.
La confianza NO es una cifra autodeclarada por el modelo: se calcula como la
proporción de acuerdo entre k muestras del propio modelo (self-consistency),
con temperatura > 0 (ver configs/model_config.yaml -> self_consistency).

Esqueleto pendiente de implementación.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from agents.reasoning import ReasoningOutput


@dataclass
class ResolutionOutput:
    etiqueta: str          # "POS" | "NEG" | "NEU"
    confianza: float       # proporción de acuerdo entre las k muestras, no autodeclarada
    muestras: list[str]    # las k etiquetas individuales obtenidas


def calcular_confianza_por_acuerdo(muestras: list[str]) -> tuple[str, float]:
    """
    Dado un listado de k etiquetas (una por muestra de self-consistency),
    devuelve la etiqueta mayoritaria y la proporción de acuerdo.
    Esta función sí está implementada porque no depende de llamar al LLM.
    """
    if not muestras:
        raise ValueError("Se necesita al menos una muestra para calcular confianza.")
    conteo = Counter(muestras)
    etiqueta, votos = conteo.most_common(1)[0]
    confianza = votos / len(muestras)
    return etiqueta, confianza


def run_resolution(texto: str, hipotesis: ReasoningOutput, k_muestras: int = 5) -> ResolutionOutput:
    """
    TODO (siguiente etapa):
        - llamar al modelo k_muestras veces con temperatura_resolucion > 0
        - usar calcular_confianza_por_acuerdo() sobre las k respuestas
    """
    raise NotImplementedError("Pendiente de implementar en la etapa de desarrollo de agentes.")
