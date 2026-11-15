# Busca por palavras (BM25), a metade "lexical" da recuperação híbrida da aula de RAG.
# O vetorial acha sinônimo ("demorou" ~ "atrasou"); o BM25 acha a palavra exata
# (nome de produto, "Correios"...). Juntando os dois com RRF a gente pega o melhor de cada.
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd

from src.config import KB_FILE

# stopwords bem básicas, só pra o BM25 não dar peso a "de", "o", "que"...
_STOP = set("""a o as os um uma de do da dos das em no na nos nas por para com sem e ou que se
mas ao aos foi ser sao esta estao tem ter muito mais ja nao me meu minha eu voce lhe isso este esse
essa pra pelo pela como quando so""".split())

META_COLS = ["doc_id", "review_id", "order_id", "nota", "categoria", "ano_compra",
             "mes_compra", "order_status", "dias_atraso", "entrega_atrasada"]


def tokenize(text: str) -> list[str]:
    t = unicodedata.normalize("NFD", text.lower())
    t = "".join(c for c in t if unicodedata.category(c) != "Mn")  # tira acentos
    return [w for w in re.findall(r"\w+", t) if w not in _STOP and len(w) > 1]


@lru_cache(maxsize=1)
def _load():
    from rank_bm25 import BM25Okapi

    df = pd.read_csv(KB_FILE)
    # monta o índice BM25 uma vez só (35 mil textos, leva alguns segundos)
    return df, BM25Okapi([tokenize(t) for t in df["texto"]])


def _mask(df: pd.DataFrame, f) -> np.ndarray:
    # mesmos filtros que o Chroma aplica, só que feitos na mão em cima do DataFrame
    m = np.ones(len(df), dtype=bool)
    if f.nota_min:
        m &= (df["nota"] >= f.nota_min).to_numpy()
    if f.nota_max:
        m &= (df["nota"] <= f.nota_max).to_numpy()
    if f.categorias:
        m &= df["categoria"].isin(f.categorias).to_numpy()
    if f.anos:
        m &= df["ano_compra"].isin(f.anos).to_numpy()
    if f.entrega_atrasada:
        m &= (df["entrega_atrasada"] == 1).to_numpy()
    return m


def search(consulta: str, filtros, n: int = 40) -> list[dict]:
    """Devolve até n avaliações ordenadas pelo score BM25 (só as com score > 0)."""
    df, bm25 = _load()
    scores = bm25.get_scores(tokenize(consulta))
    scores = np.where(_mask(df, filtros), scores, 0.0)  # filtrados ficam com 0
    top = np.argsort(-scores)[:n]
    out = []
    for i in top:
        if scores[i] <= 0:
            continue
        row = df.iloc[i]
        meta = {k: (int(row[k]) if k in ("nota", "ano_compra", "dias_atraso", "entrega_atrasada")
                    else str(row[k])) for k in META_COLS}
        out.append({"doc_id": meta["doc_id"], "texto": row["texto"], "meta": meta})
    return out
