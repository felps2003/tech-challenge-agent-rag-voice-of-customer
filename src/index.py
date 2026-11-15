# Cria o índice vetorial (Chroma via LangChain) a partir da base já limpa.
# Rodar:  python -m src.index            (tudo)
#         python -m src.index --limit 2000   (teste rápido com amostra)
from __future__ import annotations

import argparse
import time

import chromadb
import pandas as pd
from langchain_chroma import Chroma

from src.config import CHROMA_DIR, COLLECTION, KB_FILE
from src.embeddings import E5Embeddings

# colunas que viram metadados (servem de filtro e aparecem na evidência)
INT_COLS = {"nota", "ano_compra", "dias_atraso", "entrega_atrasada"}
META_COLS = ["doc_id", "review_id", "order_id", "nota", "categoria", "ano_compra",
             "mes_compra", "order_status", "dias_atraso", "entrega_atrasada"]


def get_vectorstore(reset: bool = False) -> Chroma:
    if reset:
        # apaga a coleção antiga pra reindexar do zero
        try:
            chromadb.PersistentClient(path=str(CHROMA_DIR)).delete_collection(COLLECTION)
        except Exception:
            pass  # se não existia, tudo bem
    return Chroma(
        collection_name=COLLECTION,
        embedding_function=E5Embeddings(),
        persist_directory=str(CHROMA_DIR),
        # distância cosseno (o padrão do Chroma seria L2)
        collection_metadata={"hnsw:space": "cosine"},
    )


def index_all(limit: int | None = None, batch: int = 256) -> None:
    if not KB_FILE.exists():
        raise FileNotFoundError("Rode antes: python -m src.prepare")
    df = pd.read_csv(KB_FILE)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=42)  # amostra reprodutível

    vs = get_vectorstore(reset=True)
    t0 = time.time()
    # manda em lotes pra não estourar memória e poder mostrar o progresso
    for i in range(0, len(df), batch):
        part = df.iloc[i : i + batch]
        metadatas = [
            {k: (int(v) if k in INT_COLS else str(v)) for k, v in row.items()}
            for row in part[META_COLS].to_dict("records")
        ]
        # add_texts calcula os embeddings (embed_documents) e grava no Chroma
        vs.add_texts(texts=part["texto"].tolist(), metadatas=metadatas,
                     ids=part["doc_id"].tolist())
        print(f"\r{min(i + batch, len(df))}/{len(df)} indexados ({time.time() - t0:.0f}s)",
              end="", flush=True)
    print(f"\nPronto: {vs._collection.count()} documentos em {CHROMA_DIR}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    index_all(ap.parse_args().limit)
