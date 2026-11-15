# Fábrica do LLM usando LangChain (como na aula 01: ChatOpenAI/ChatGoogleGenerativeAI são Runnables).
# Se existir chave dos dois provedores, o segundo vira fallback (aula 02: with_fallbacks).
from __future__ import annotations

import os

from src import config


def _gemini(temperature: float):
    from langchain_google_genai import ChatGoogleGenerativeAI

    key = os.getenv("GEMINI_API_KEY")
    if not key:
        raise RuntimeError("Defina GEMINI_API_KEY no .env")
    return ChatGoogleGenerativeAI(model=config.GEMINI_MODEL, temperature=temperature,
                                  google_api_key=key)


def _openai(temperature: float):
    from langchain_openai import ChatOpenAI

    key = os.getenv("OPENAI_API_KEY")
    if not key:
        raise RuntimeError("Defina OPENAI_API_KEY no .env")
    return ChatOpenAI(model=config.OPENAI_MODEL, temperature=temperature, api_key=key)


def get_llm(temperature: float = 0.1, schema=None):
    """Devolve um Runnable. Se `schema` (Pydantic) vier preenchido, a saída já vem
    validada nesse formato (with_structured_output), sem parsear JSON na mão."""
    ordem = [_openai, _gemini] if config.LLM_PROVIDER == "openai" else [_gemini, _openai]

    modelos = []
    for criar in ordem:
        try:
            m = criar(temperature)
        except RuntimeError:
            continue  # sem chave desse provedor, pula
        modelos.append(m.with_structured_output(schema) if schema else m)

    if not modelos:
        raise RuntimeError("Nenhuma chave de LLM encontrada no .env")
    # o primeiro é o principal; se der erro (rate limit, 500...), tenta o próximo
    return modelos[0].with_fallbacks(modelos[1:]) if len(modelos) > 1 else modelos[0]
