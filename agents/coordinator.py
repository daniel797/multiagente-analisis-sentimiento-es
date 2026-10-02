"""
Meta-agente / Coordinador.

Mantiene el estado compartido (texto, flags de percepción, hipótesis, etiquetas
parciales, confianza) y decide, con la condición de terminación, si la etiqueta
queda cerrada o si conviene una segunda ronda de discusión entre Razonamiento y
Resolución (como máximo una, según el diseño de la Semana 2).

Esqueleto pendiente de implementación.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from agents.perception import PerceptionOutput, run_perception
from agents.reasoning import ReasoningOutput, run_reasoning
from agents.resolution import ResolutionOutput, run_resolution


@dataclass
class EstadoCompartido:
    texto: str
    variedad: str
    percepcion: PerceptionOutput | None = None
    razonamientos: list[ReasoningOutput] = field(default_factory=list)
    resoluciones: list[ResolutionOutput] = field(default_factory=list)
    iteracion: int = 0


def condicion_terminacion(confianza: float, iteracion: int, tau: float, max_iter: int) -> bool:
    """
    True si el flujo debe terminar: confianza >= tau, o se alcanzó max_iter.
    Esta función sí está implementada, por no depender de llamar al LLM.
    """
    return confianza >= tau or iteracion >= max_iter


def run_pipeline(texto: str, variedad: str, tau: float = 0.6, max_iter: int = 2) -> EstadoCompartido:
    """
    Orquesta Percepción -> Razonamiento -> Resolución, con el bucle de verificación.

    TODO (siguiente etapa):
        - implementar run_perception / run_reasoning / run_resolution (ver sus TODO)
        - implementar la segunda ronda de discusión explícita cuando no se cumple
          la condición de terminación
        - registrar tokens, latencia y trazas por agente (ver README.md, sección de salida esperada)
    """
    raise NotImplementedError(
        "Pendiente: este coordinador orquesta a los tres agentes, que todavía no "
        "están implementados (ver agents/perception.py, reasoning.py, resolution.py)."
    )
