"""
Condición B1 (monoagente) con Salamandra-7B-Instruct, repetida para cada variante de los datos.

ESTADO: el código de este archivo se probó con un backend de reglas (`--backend stub`) para verificar
el flujo completo (prompt, lectura de la salida, reintentos, reanudación, evaluación). NO se ejecutó
con el modelo real, porque el entorno donde se escribió no tiene GPU ni acceso a Hugging Face. La
primera corrida con el modelo real debe hacerse con `--limite 50` para revisar a mano las salidas.

Qué hace
  1. Lee data/<split>/<variedad>.tsv (por defecto el split `dev`).
  2. Para cada variante de los datos (base, limpio, ...), arma un prompt zero-shot con el tuit
     transformado y le pide al modelo un JSON {"label": "POS" | "NEG" | "NEU"}.
  3. Guarda cada predicción en un CSV apenas la obtiene, y al reanudar salta lo ya hecho
     (útil en Colab, donde la sesión puede cortarse).
  4. `resumir` calcula Macro-F1 por variedad con intervalo y la diferencia pareada contra la base.

El modelo se usa tal como se publica, sin fine-tuning. Los parámetros salen de configs/model_config.yaml.
Al monoagente no se le dice la variedad del tuit.

Uso (desde la raíz del repositorio)
    # prueba del flujo sin modelo
    python scripts/experimento/monoagente_salamandra.py ejecutar --backend stub --limite 40 --out docs/experimento/prueba_stub
    # corrida real (GPU), primero una muestra pequeña
    python scripts/experimento/monoagente_salamandra.py ejecutar --variantes base limpio --limite 50 --out docs/experimento/llm
    # corrida completa y resumen
    python scripts/experimento/monoagente_salamandra.py ejecutar --variantes base limpio --out docs/experimento/llm
    python scripts/experimento/monoagente_salamandra.py resumir --out docs/experimento/llm

Dependencias del modelo real: torch, transformers, accelerate, bitsandbytes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import regex as re
import yaml

AQUI = Path(__file__).resolve().parent
sys.path.append(str(AQUI))
sys.path.append(str(AQUI.parents[0] / "eda"))
sys.path.append(str(AQUI.parents[0] / "features"))
import eda_utils as E  # noqa: E402
import evaluacion as EV  # noqa: E402
import feature_utils as F  # noqa: E402
import variantes_datos as V  # noqa: E402

ETIQUETAS = ["POS", "NEG", "NEU"]
VERSION_PROMPT = "B1-v1"

PROMPT_SISTEMA = (
    "Eres un analista de sentimiento de tuits en español. Clasifica el sentimiento del tuit en una de tres etiquetas:\n"
    "- POS: el tuit expresa una opinión o emoción positiva.\n"
    "- NEG: el tuit expresa una opinión o emoción negativa.\n"
    "- NEU: el tuit es neutro, informativo o no expresa una opinión clara.\n"
    "El tuit puede traer expresiones coloquiales propias de cualquier país hispanohablante. "
    'Responde únicamente con un JSON de esta forma: {"label": "POS"}, {"label": "NEG"} o {"label": "NEU"}.'
)


def construir_mensajes(texto_variante: str) -> list[dict]:
    """Mensajes en formato de chat. Solo lleva el tuit: ni la variedad ni ningún otro dato del corpus."""
    return [
        {"role": "system", "content": PROMPT_SISTEMA},
        {"role": "user", "content": f"Tuit: {texto_variante}"},
    ]


def leer_etiqueta(salida: str) -> str | None:
    """Extrae POS, NEG o NEU de la salida. Primero intenta JSON y después busca la etiqueta suelta."""
    if not salida:
        return None
    m = re.search(r"\{.*?\}", salida, flags=re.S)
    if m:
        try:
            valor = json.loads(m.group(0)).get("label")
            if isinstance(valor, str) and valor.strip().upper() in ETIQUETAS:
                return valor.strip().upper()
        except (json.JSONDecodeError, AttributeError):
            pass
    sueltas = re.findall(r"\b(POS|NEG|NEU)\b", salida.upper())
    return sueltas[0] if sueltas else None


# ----------------------------------------------------------------------------
# Backends
# ----------------------------------------------------------------------------
class BackendStub:
    """Backend de reglas para probar el flujo sin modelo. NO es un clasificador serio."""

    def generar(self, lotes: list[list[dict]], temperatura: float, semilla: int, **_) -> list[str]:  # acepta top_p y top_k y los ignora
        salidas = []
        for mensajes in lotes:
            texto = mensajes[-1]["content"].lower()
            palabras = set(re.findall(r"\p{L}+", texto))
            pos, neg = len(palabras & F.LEXICO_POSITIVO), len(palabras & F.LEXICO_NEGATIVO)
            etiqueta = "POS" if pos > neg else ("NEG" if neg > pos else "NEU")
            salidas.append(json.dumps({"label": etiqueta}))
        return salidas


class BackendTransformers:
    """Salamandra-7B-Instruct con transformers (y 4 bits con bitsandbytes). Sin fine-tuning."""

    def __init__(self, nombre: str, revision: str, cuantizacion: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        kw = {}
        if "4-bit" in cuantizacion:
            kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
        self.tok = AutoTokenizer.from_pretrained(nombre, revision=revision or None, padding_side="left")
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.modelo = AutoModelForCausalLM.from_pretrained(nombre, revision=revision or None, device_map="auto", torch_dtype=torch.float16, **kw)
        self.modelo.eval()

    def generar(self, lotes: list[list[dict]], temperatura: float, semilla: int, top_p: float = 0.9, top_k: int = 40, max_new_tokens: int = 24) -> list[str]:
        import torch

        textos = [self.tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in lotes]
        enc = self.tok(textos, return_tensors="pt", padding=True, add_special_tokens=False).to(self.modelo.device)
        torch.manual_seed(semilla)
        gen = {"max_new_tokens": max_new_tokens, "pad_token_id": self.tok.pad_token_id}
        if temperatura > 0:
            gen.update(do_sample=True, temperature=temperatura, top_p=top_p, top_k=top_k)
        else:
            gen.update(do_sample=False)
        with torch.no_grad():
            salida = self.modelo.generate(**enc, **gen)
        nuevos = salida[:, enc["input_ids"].shape[1]:]
        return self.tok.batch_decode(nuevos, skip_special_tokens=True)


# ----------------------------------------------------------------------------
# Ejecución
# ----------------------------------------------------------------------------
def cargar_config(ruta: str = "configs/model_config.yaml") -> dict:
    with open(ruta, encoding="utf-8") as f:
        return yaml.safe_load(f)


def clasificar_lote(backend, textos_variante: list[str], temperatura: float, semilla: int, reintentos: int, dec: dict) -> list[dict]:
    """Un intento con la temperatura pedida y hasta `reintentos` más (con muestreo) si la salida no se entiende."""
    kw = {"top_p": dec.get("top_p", 0.9), "top_k": dec.get("top_k", 40)}
    lotes = [construir_mensajes(t) for t in textos_variante]
    res = [{"salida_cruda": s, "pred": leer_etiqueta(s), "intentos": 1} for s in backend.generar(lotes, temperatura=temperatura, semilla=semilla, **kw)]
    for intento in range(2, reintentos + 2):
        pend = [i for i, r in enumerate(res) if r["pred"] is None]
        if not pend:
            break
        nuevas = backend.generar([lotes[i] for i in pend], temperatura=max(temperatura, 0.3), semilla=semilla + intento, **kw)
        for i, s in zip(pend, nuevas):
            res[i] = {"salida_cruda": s, "pred": leer_etiqueta(s), "intentos": intento}
    return res


def ejecutar(a: argparse.Namespace) -> None:
    cfg = cargar_config(a.config)
    dec, modelo = cfg["decodificacion"], cfg["modelo"]
    reintentos = cfg["formato_salida"]["reintentos_maximos"]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ruta = out / "predicciones_llm.csv"

    corpus = E.cargar_corpus(a.data_dir)
    datos = corpus[corpus.split == a.split].sort_values("id").reset_index(drop=True)
    if a.limite:
        datos = datos.groupby("variedad", group_keys=False).head(max(1, a.limite // datos["variedad"].nunique())).reset_index(drop=True)

    hechas = set()
    if ruta.exists():
        previo = pd.read_csv(ruta, dtype=str, keep_default_na=False)
        hechas = set(zip(previo["id"], previo["variante"]))

    if a.backend == "stub":
        backend = BackendStub()
    else:
        if not (a.revision or modelo.get("revision")):
            print("AVISO: configs/model_config.yaml no fija `revision`. Fíjala antes de la corrida final para que sea reproducible.")
        backend = BackendTransformers(modelo["nombre"], a.revision or modelo.get("revision", ""), modelo.get("cuantizacion", ""))

    temperatura = a.temperatura
    for variante in a.variantes:
        pendientes = datos[[(i, variante) not in hechas for i in datos["id"]]]
        print(f"variante={variante}: {len(pendientes)} tuits por clasificar ({len(datos) - len(pendientes)} ya hechos)")
        for ini in range(0, len(pendientes), a.lote):
            trozo = pendientes.iloc[ini:ini + a.lote]
            res = clasificar_lote(backend, [V.aplicar(t, variante) for t in trozo["texto"]], temperatura, dec.get("semilla", 42), reintentos, dec)
            filas = pd.DataFrame({
                "id": trozo["id"].values, "variedad": trozo["variedad"].values, "etiqueta": trozo["etiqueta"].values,
                "variante": variante, "version_prompt": VERSION_PROMPT, "pred": [r["pred"] or "" for r in res],
                "intentos": [r["intentos"] for r in res], "salida_cruda": [r["salida_cruda"].replace("\n", " ")[:200] for r in res],
            })
            filas.to_csv(ruta, mode="a", header=not ruta.exists(), index=False, encoding="utf-8")
            print(f"  {min(ini + a.lote, len(pendientes))}/{len(pendientes)}", end="\r")
        print()
    print(f"Predicciones en {ruta}")


def resumir(a: argparse.Namespace) -> None:
    out = Path(a.out)
    pred = pd.read_csv(out / "predicciones_llm.csv", dtype=str, keep_default_na=False)
    if "base" not in set(pred["variante"]):
        sys.exit("Falta la variante `base` en las predicciones: se necesita para las diferencias pareadas.")
    pred["valida"] = pred["pred"].isin(ETIQUETAS)
    # Una salida que no se entendió cuenta como NEU, y la tasa de salidas inválidas se reporta aparte.
    pred["pred_ok"] = pred["pred"].where(pred["valida"], "NEU")

    ancho = {v: g.set_index("id") for v, g in pred.groupby("variante")}
    ids = sorted(set.intersection(*[set(g.index) for g in ancho.values()]))
    filas, difs, invalidas = [], [], []
    for v, g in ancho.items():
        g = g.loc[ids]
        for amb, m in EV.ambitos(g["variedad"].reset_index(drop=True)).items():
            filas.append({"variante": v, "ambito": amb, **EV.metricas(g["etiqueta"].to_numpy()[m], g["pred_ok"].to_numpy()[m], a.n_boot)})
        invalidas.append({"variante": v, "n": len(g), "salidas_invalidas": int((~g["valida"]).sum()), "pct_invalidas": round(100 * (~g["valida"]).mean(), 2)})
        if v != "base":
            b = ancho["base"].loc[ids]
            difs += EV.comparar_con_base(g["etiqueta"].to_numpy(), b["pred_ok"].to_numpy(), g["pred_ok"].to_numpy(), g["variedad"].reset_index(drop=True), v, a.n_boot)
    (out / "tablas").mkdir(exist_ok=True)
    pd.DataFrame(filas).round(4).to_csv(out / "tablas" / "resultados_llm_por_variante.csv", index=False, encoding="utf-8")
    pd.DataFrame(difs).round(4).to_csv(out / "tablas" / "diferencias_llm_pareadas.csv", index=False, encoding="utf-8")
    pd.DataFrame(invalidas).to_csv(out / "tablas" / "salidas_invalidas.csv", index=False, encoding="utf-8")
    print(pd.DataFrame(filas)[lambda d: d.ambito == "Todas"][["variante", "macro_f1", "ic_inf", "ic_sup"]].round(4).to_string(index=False))
    print(pd.DataFrame(invalidas).to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("ejecutar", help="clasifica los tuits con el LLM para cada variante")
    e.add_argument("--data-dir", default="data")
    e.add_argument("--config", default="configs/model_config.yaml")
    e.add_argument("--split", default="dev", choices=["dev", "train"])
    e.add_argument("--variantes", nargs="+", default=["base", "limpio"], choices=sorted(V.VARIANTES))
    e.add_argument("--backend", default="transformers", choices=["transformers", "stub"])
    e.add_argument("--revision", default="")
    e.add_argument("--temperatura", type=float, default=0.0, help="0 = decodificación determinista (recomendado para B1)")
    e.add_argument("--lote", type=int, default=16)
    e.add_argument("--limite", type=int, default=0, help="tuits en total, repartidos por variedad (0 = todos)")
    e.add_argument("--out", default="docs/experimento/llm")
    r = sub.add_parser("resumir", help="Macro-F1 por variante y diferencias pareadas contra la base")
    r.add_argument("--out", default="docs/experimento/llm")
    r.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()
    ejecutar(a) if a.cmd == "ejecutar" else resumir(a)


if __name__ == "__main__":
    main()
