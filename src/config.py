# Configurações do projeto num lugar só, pra não ficar número solto espalhado pelo código.
import os
from pathlib import Path

from dotenv import load_dotenv

# pasta raiz do projeto (um nível acima de src/)
ROOT = Path(__file__).resolve().parent.parent
# carrega o arquivo .env (as chaves de API ficam lá, nunca no código)
load_dotenv(ROOT / ".env")

# pastas e arquivos que o pipeline usa
RAW_DIR = ROOT / "data" / "raw"              # CSVs originais da Olist
PROCESSED_DIR = ROOT / "data" / "processed"  # base já limpa
# onde o ChromaDB salva o índice. Dá pra mudar com CHROMA_DIR no .env: se o projeto estiver numa pasta
# sincronizada (OneDrive, Dropbox), o Chroma trava ao gravar, então use uma pasta local fora dela
CHROMA_DIR = Path(os.getenv("CHROMA_DIR") or ROOT / "data" / "chroma")
KB_FILE = PROCESSED_DIR / "reviews_kb.csv"
COLLECTION = "olist_reviews"

# modelo de embeddings: multilíngue porque as avaliações são em português
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")

# qual LLM usar; o outro vira reserva (fallback) se existir chave dele
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# quantos candidatos a busca traz e quantos entram no contexto do LLM
TOP_K_CANDIDATES = int(os.getenv("TOP_K_CANDIDATES", "40"))
TOP_K_CONTEXT = int(os.getenv("TOP_K_CONTEXT", "12"))
# abaixo dessa similaridade (cosseno) a avaliação não conta como evidência
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.905"))
# se sobrar menos evidências que isso, o sistema recusa responder
MIN_EVIDENCES = int(os.getenv("MIN_EVIDENCES", "3"))
