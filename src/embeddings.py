# Embeddings locais com o E5 multilíngue, embrulhado na interface do LangChain
# pra poder plugar direto no Chroma (como na aula de Vector Databases).
from __future__ import annotations

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from src.config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def _model():
    # carrega o modelo uma vez só (demora uns segundos); o cache evita recarregar
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(EMBEDDING_MODEL)


class E5Embeddings(Embeddings):
    """O E5 foi treinado com prefixos: 'query: ' na pergunta e 'passage: ' no documento.
    Sem eles a busca piora, por isso uma classe própria em vez do HuggingFaceEmbeddings pronto."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = _model().encode(
            [f"passage: {t}" for t in texts],  # prefixo dos documentos
            batch_size=64,
            normalize_embeddings=True,  # vetor de tamanho 1 -> cosseno vira produto escalar
            show_progress_bar=False,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        # a pergunta usa o prefixo 'query: '
        return _model().encode(f"query: {text}", normalize_embeddings=True).tolist()
