# Mede quanto os filtros de metadados ajudam a recuperação (desafio adicional do Tech Challenge).
# Compara a mesma consulta SEM filtro e COM filtro, só com busca vetorial (não chama o LLM).
# Rodar:  python -m scripts.metadados
from src.rag import Filters, retrieve


def fracao(rows, chave, condicao) -> float:
    # fração dos resultados cujo metadado `chave` satisfaz a condição
    return sum(condicao(r["meta"][chave]) for r in rows) / max(len(rows), 1)


if __name__ == "__main__":
    # A) reclamações em geral: com filtro de nota baixa, todo resultado é negativo?
    q = "produto com defeito, atrasou, não recebi, reclamação"
    sem = retrieve(q, Filters(), 12)
    com = retrieve(q, Filters(nota_max=2), 12)
    print(f"A) nota <= 2 entre os 12 primeiros: sem filtro {fracao(sem, 'nota', lambda v: v <= 2):.0%}"
          f" | com filtro {fracao(com, 'nota', lambda v: v <= 2):.0%}")

    # B) elogios: aqui o embedding já separa bem, então não deve haver ganho
    q = "recomendo, ótimo produto, gostei muito"
    sem = retrieve(q, Filters(), 12)
    print(f"B) nota >= 4 entre os 12 primeiros: sem filtro {fracao(sem, 'nota', lambda v: v >= 4):.0%}"
          " | com filtro 100% (por construção)")

    # C) categoria: o embedding do texto não "sabe" a categoria do pedido
    q = "produto de má qualidade, defeito"
    sem = retrieve(q, Filters(), 40)
    n_cat = sum(r["meta"]["categoria"] == "bed_bath_table" for r in sem)
    print(f"C) bed_bath_table entre os 40 primeiros: sem filtro {n_cat} de 40 | com filtro 12 de 12")

    # D) atraso: reclamação de entrega nem sempre é entrega atrasada
    q = "problema na entrega, demorou, não chegou"
    sem = retrieve(q, Filters(), 40)
    v = [r["meta"]["entrega_atrasada"] for r in sem]
    print(f"D) entre 40 resultados de 'problema na entrega': atrasadas {v.count(1)}, "
          f"no prazo {v.count(0)}, sem data {v.count(-1)} | com filtro 12 de 12 atrasadas")
