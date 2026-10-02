"""
Agente de Razonamiento (2.ª llamada al LLM).

Responsabilidad: a partir del texto y de los indicios del agente de Percepción,
construir una hipótesis de polaridad con su justificación.

Esqueleto pendiente de implementación (ver agents/perception.py para el porqué).
"""
from __future__ import annotations

from dataclasses import dataclass

from agents.perception import PerceptionOutput


@dataclass
class ReasoningOutput:
    hipotesis_polaridad: str  # "POS" | "NEG" | "NEU"
    justificacion: str


def run_reasoning(texto: str, percepcion: PerceptionOutput) -> ReasoningOutput:
    """
    TODO (siguiente etapa):
        - construir el prompt condicionado a los flags de percepción
        - llamar al modelo (temperatura_razonamiento, ver configs/model_config.yaml)
    """
    raise NotImplementedError("Pendiente de implementar en la etapa de desarrollo de agentes.")
