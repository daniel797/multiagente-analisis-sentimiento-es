"""
Agente de Percepción (1.ª llamada al LLM).

Responsabilidad: leer el texto y devolver un JSON con los fenómenos lingüísticos
detectados (sarcasmo, diminutivo afectivo, doble negación, code-switching) junto
con la evidencia textual de cada uno. No clasifica polaridad.

Este archivo es un esqueleto: todavía no implementa la llamada real al LLM.
Se completa en la siguiente etapa del proyecto, una vez cerrada la auditoría
del corpus (ver scripts/data/audit_phenomena.py) y fijado el modelo base
(ver configs/model_config.yaml).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PerceptionOutput:
    sarcasmo: bool
    sarcasmo_evidencia: str
    diminutivo_afectivo: bool
    diminutivo_evidencia: str
    doble_negacion: bool
    doble_negacion_evidencia: str
    code_switching: bool
    code_switching_evidencia: str


def run_perception(texto: str) -> PerceptionOutput:
    """
    Llama al LLM en modo Percepción sobre `texto` y devuelve los indicios detectados.

    TODO (siguiente etapa):
        - construir el prompt de especialización en español
        - llamar al modelo según configs/model_config.yaml (temperatura_percepcion = 0.0)
        - parsear la respuesta JSON, con reintento ante formato inválido
    """
    raise NotImplementedError("Pendiente de implementar en la etapa de desarrollo de agentes.")
