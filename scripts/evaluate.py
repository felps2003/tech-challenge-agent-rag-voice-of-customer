# Avaliação do RAG com as métricas da aula: precisão da recuperação, recusa, citações,
# latência e fidelidade (LLM-as-a-judge).
# Rodar:  python -m scripts.evaluate            (recuperação + fim a fim)
#         python -m scripts.evaluate --judge    (acrescenta a fidelidade)
#         python -m scripts.evaluate --no-llm   (só recuperação, não gasta API)
#
# Não tenho avaliações rotuladas à mão, então "relevante" = contém algum termo-chave do tema.
# É um proxy simples, serve para comparar configurações, não como verdade absoluta.
import argparse
import statistics

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src import config
from src.lexical import tokenize
from src.llm import get_llm
from src.rag import Filters, answer, retrieve, retrieve_hybrid, select_evidence

# (pergunta, termos-chave do tema). termos = None quer dizer pergunta FORA do escopo.
GOLDEN = [
    ("O que os clientes relatam sobre atraso na entrega?", ["atras", "demor", "prazo", "entreg"]),
    ("Quais problemas de produto com defeito aparecem?", ["defeit", "quebr", "estrag", "nao funcion"]),
    ("O que os clientes elogiam?", ["otim", "recomend", "excelent", "gostei", "perfeit"]),
    ("Há reclamações sobre o atendimento do vendedor?", ["vendedor", "atendimento", "resposta", "contato"]),
    ("Clientes que receberam produto diferente do pedido", ["diferent", "errad", "troc", "nao era"]),
    ("Problemas com embalagem", ["embalag", "caixa", "amassad", "danific"]),
    ("Pedidos que nunca chegaram", ["nunca", "nao recebi", "nao chegou", "cancel"]),
    ("Qual é a capital da França?", None),
    ("Como fazer bolo de chocolate?", None),
    ("Quem ganhou a Copa do Mundo de 2002?", None),
]
K = config.TOP_K_CONTEXT


class Veredito(BaseModel):
    total: int         # quantas afirmações a resposta tem
    sustentadas: int   # quantas delas o contexto realmente sustenta


# chain do "juiz": prompt | LLM com saída estruturada
JUIZ = ChatPromptTemplate.from_messages([
    ("system", "Você é um verificador. Conte as afirmações da RESPOSTA e quantas delas são "
               "sustentadas pelo CONTEXTO."),
    ("human", "CONTEXTO:\n{contexto}\n\nRESPOSTA:\n{resposta}"),
])


def acerta_termo(texto: str, termos: list[str]) -> bool:
    # tokeniza igual ao BM25 (sem acento) pra comparar com os termos-chave
    return any(t in " ".join(tokenize(texto)) for t in termos)


def precisao_recuperacao(busca, nome: str) -> None:
    precs = []
    for pergunta, termos in GOLDEN:
        if termos is None:
            continue  # as de fora do escopo não entram aqui
        # limiar 0: quero medir só a qualidade do ranking, sem o filtro de similaridade
        ev = select_evidence(busca(pergunta, Filters()), K, 0.0)
        precs.append(sum(acerta_termo(e.texto, termos) for e in ev) / max(len(ev), 1))
    print(f"Precision@{K} ({nome}): {statistics.mean(precs):.2f}")


def fidelidade(resposta: str, contexto: str) -> float:
    try:
        v = (JUIZ | get_llm(0, schema=Veredito)).invoke({"contexto": contexto, "resposta": resposta})
        return v.sustentadas / v.total if v.total else 1.0
    except Exception:
        return float("nan")  # se o juiz falhar, não entra na média


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--no-llm", action="store_true")
    args = ap.parse_args()

    print("== Recuperação ==")
    precisao_recuperacao(lambda q, f: retrieve(q, f, 40), "somente vetor")
    precisao_recuperacao(lambda q, f: retrieve_hybrid(q, f, 40), "híbrido BM25+vetor (RRF)")
    if args.no_llm:
        raise SystemExit

    print("\n== Fim a fim ==")
    recusa_certa, latencias, fids, tem_citacao = [], [], [], []
    for pergunta, termos in GOLDEN:
        r = answer(pergunta)
        deveria_recusar = termos is None
        recusa_certa.append((r.status == "sem_evidencia") == deveria_recusar)
        latencias.append(r.latencia_s)
        if r.status == "ok":
            tem_citacao.append(len(r.citadas) > 0)
            if args.judge:
                fids.append(fidelidade(r.resposta, r.contexto))
        print(f"[{r.status:13}] {r.latencia_s:4.1f}s  {pergunta}")

    print(f"\nAcerto de recusa (responde o que deve, recusa o que deve): {statistics.mean(recusa_certa):.0%}")
    if tem_citacao:
        print(f"Respostas com citação: {statistics.mean(tem_citacao):.0%}")
    print(f"Latência média: {statistics.mean(latencias):.1f}s")
    fids = [x for x in fids if x == x]  # x == x é falso para NaN
    if fids:
        print(f"Fidelidade média (LLM-as-judge): {statistics.mean(fids):.0%}")
