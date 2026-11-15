# Monta a base de conhecimento a partir dos CSVs da Olist.
# Rodar:  python -m src.prepare
#
# Resumo do que é feito aqui (as justificativas completas estão no README):
#  - só ficam avaliações que têm texto (sem texto não há o que recuperar)
#  - título e mensagem viram um documento só
#  - limpeza leve; NÃO tiro stopwords nem faço stemming, porque o modelo de embeddings
#    entende o contexto da frase inteira e isso estragaria o sentido
#  - texto duplicado e texto muito curto ("ok", "bom") saem
#  - junto metadados do pedido, do produto e da entrega pra poder filtrar depois
from __future__ import annotations

import re

import pandas as pd

from src.config import KB_FILE, PROCESSED_DIR, RAW_DIR

MIN_CHARS = 15  # menos que isso não serve de evidência

_URL = re.compile(r"https?://\S+|www\.\S+")
_SPACES = re.compile(r"\s+")
_REPEAT = re.compile(r"(.)\1{3,}")  # "ótimooooo" vira "ótimooo"
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_CPF = re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b")
_PHONE = re.compile(r"(?:\(\d{2}\)\s?|\b\d{2}\s?)?\b9?\d{4}[-\s]?\d{4}\b")


def mask_pii(t: str) -> str:
    # esconde dado pessoal (LGPD: só guardo o que preciso). É regex, então não pega tudo.
    t = _EMAIL.sub("[EMAIL]", t)
    t = _CPF.sub("[CPF]", t)  # CPF antes do telefone porque os dois são só dígitos
    return _PHONE.sub("[TELEFONE]", t)


def clean_text(text) -> str:
    # valor ausente (None ou NaN) vira texto vazio
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    t = str(text)
    t = _URL.sub(" ", t)             # tira links
    t = mask_pii(t)                  # mascara dados pessoais
    t = _REPEAT.sub(r"\1\1\1", t)    # reduz letras repetidas
    return _SPACES.sub(" ", t).strip()  # espaços e quebras de linha viram um espaço só


def build_document(title: str, message: str) -> str:
    # o título costuma resumir o tema ("Não recomendo"), então junto com a mensagem
    title, message = clean_text(title), clean_text(message)
    if title and message:
        sep = "" if title[-1] in ".!?" else "."  # só coloca ponto se o título não terminar em pontuação
        return f"{title}{sep} {message}"
    return title or message  # se um dos dois faltar, usa o que tiver


def _read(name: str, **kw) -> pd.DataFrame:
    path = RAW_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}\n"
            "Baixe o dataset da Olist no Kaggle e extraia os CSVs em data/raw/."
        )
    return pd.read_csv(path, **kw)


def build_kb() -> pd.DataFrame:
    reviews = _read("olist_order_reviews_dataset.csv")
    n_total = len(reviews)

    # 1) cria o texto do documento e descarta os curtos/vazios
    reviews["texto"] = [
        build_document(t, m)
        for t, m in zip(reviews["review_comment_title"], reviews["review_comment_message"])
    ]
    reviews = reviews[reviews["texto"].str.len() >= MIN_CHARS].copy()
    n_text = len(reviews)

    # 2) tira duplicatas exatas (ignorando maiúscula/minúscula), mantendo a primeira
    key = reviews["texto"].str.lower()
    reviews = reviews[~key.duplicated()].copy()
    n_dedup = len(reviews)

    # 3) dados do pedido (data da compra, status, datas de entrega)
    orders = _read(
        "olist_orders_dataset.csv",
        parse_dates=["order_purchase_timestamp", "order_delivered_customer_date",
                     "order_estimated_delivery_date"],
    )[["order_id", "order_status", "order_purchase_timestamp",
       "order_delivered_customer_date", "order_estimated_delivery_date"]]
    df = reviews.merge(orders, on="order_id", how="left")  # left: não perde avaliação sem pedido

    # 4) categoria do produto. Um pedido pode ter vários itens, então pego a mais frequente
    items = _read("olist_order_items_dataset.csv")[["order_id", "product_id"]]
    prods = _read("olist_products_dataset.csv")[["product_id", "product_category_name"]]
    tr = _read("product_category_name_translation.csv")  # português -> inglês
    cat = items.merge(prods, on="product_id", how="left").merge(
        tr, on="product_category_name", how="left")
    cat["categoria"] = cat["product_category_name_english"].fillna(cat["product_category_name"])
    cat = (
        cat.dropna(subset=["categoria"])
        .groupby("order_id")["categoria"]
        .agg(lambda s: s.value_counts().index[0])  # a mais comum do pedido
        .reset_index()
    )
    df = df.merge(cat, on="order_id", how="left")

    # 5) colunas derivadas que viram filtro
    df["ano_compra"] = df["order_purchase_timestamp"].dt.year
    df["mes_compra"] = df["order_purchase_timestamp"].dt.strftime("%Y-%m")
    # atraso = entrega real - entrega prometida (positivo = chegou depois do prazo)
    delay = (df["order_delivered_customer_date"] - df["order_estimated_delivery_date"]).dt.days
    df["dias_atraso"] = delay
    df["entrega_atrasada"] = delay.gt(0).astype("Int64").where(delay.notna())

    # 6) valores ausentes: não descarto a avaliação, só marco que o dado não existe
    df["categoria"] = df["categoria"].fillna("desconhecida")
    df["ano_compra"] = df["ano_compra"].fillna(-1).astype(int)
    df["mes_compra"] = df["mes_compra"].fillna("desconhecido")
    df["order_status"] = df["order_status"].fillna("desconhecido")
    df["dias_atraso"] = df["dias_atraso"].fillna(-999).astype(int)       # -999 = sem data de entrega
    df["entrega_atrasada"] = df["entrega_atrasada"].fillna(-1).astype(int)  # -1 = não sei

    # id único por avaliação (o review_id sozinho se repete em alguns pedidos)
    df["doc_id"] = df["review_id"].astype(str) + "|" + df["order_id"].astype(str)
    df = df.drop_duplicates("doc_id")
    df = df.rename(columns={"review_score": "nota"})

    cols = ["doc_id", "review_id", "order_id", "texto", "nota", "categoria",
            "ano_compra", "mes_compra", "order_status", "dias_atraso", "entrega_atrasada"]
    out = df[cols].reset_index(drop=True)

    # resumo do funil, útil pra explicar no vídeo
    print(f"Avaliações no CSV ........ {n_total:>7}")
    print(f"Com texto (>= {MIN_CHARS} chars) .. {n_text:>7}")
    print(f"Após remover duplicatas .. {n_dedup:>7}")
    print(f"Documentos finais ........ {len(out):>7}")
    return out


if __name__ == "__main__":
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    kb = build_kb()
    kb.to_csv(KB_FILE, index=False)
    print(f"Base salva em {KB_FILE}")
