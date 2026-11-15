# Ajuda a escolher o MIN_SIMILARITY. Mostra a similaridade do melhor resultado para perguntas
# DENTRO do escopo e FORA do escopo; o limiar bom fica entre os dois grupos.
# Rodar:  python -m scripts.calibrate   (não chama o LLM, só embeddings + índice)
from src.rag import Filters, retrieve

# frases no estilo de comentário de cliente (é assim que a reescrita vai montar a consulta)
DENTRO = [
    "produto chegou atrasado, demorou muito para entregar",
    "o produto veio com defeito e não funciona",
    "recebi um produto diferente do que comprei",
    "ótima qualidade, recomendo, chegou antes do prazo",
    "vendedor não respondeu, atendimento péssimo",
    "embalagem chegou danificada",
    "nunca recebi o pedido",
]
# assuntos que nada têm a ver com e-commerce
FORA = [
    "qual é a capital da França?",
    "como fazer bolo de chocolate",
    "quem ganhou a Copa do Mundo de 2002?",
    "explique a teoria da relatividade",
    "qual a previsão do tempo para amanhã?",
    "escreva um poema sobre o mar",
]


def melhor_similaridade(q: str) -> float:
    # a lista vem ordenada, então o primeiro item é o mais parecido
    return retrieve(q, Filters(), 40)[0]["similarity"]


if __name__ == "__main__":
    print("== DENTRO do escopo ==")
    dentro = []
    for q in DENTRO:
        s = melhor_similaridade(q)
        dentro.append(s)
        print(f"{s:.3f}  {q}")

    print("\n== FORA do escopo ==")
    fora = []
    for q in FORA:
        s = melhor_similaridade(q)
        fora.append(s)
        print(f"{s:.3f}  {q}")

    print(f"\nmenor DENTRO = {min(dentro):.3f} | maior FORA = {max(fora):.3f}")
    print(f"limiar sugerido (ponto médio) = {(min(dentro) + max(fora)) / 2:.3f}")
    print("Se os grupos se misturam, só o limiar não resolve: aí a 2ª barreira (SEM_EVIDENCIA no LLM) segura.")
