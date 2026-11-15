# Faz uma pergunta pelo terminal.
# Exemplo:  python cli.py "Quais os principais problemas de entrega?"
import sys

from src.rag import answer

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit('Uso: python cli.py "sua pergunta"')

    r = answer(" ".join(sys.argv[1:]))  # junta as palavras caso a pergunta venha sem aspas

    print(f"\nConsulta: {r.consulta}\nFiltros: {r.filtros.describe()}\nStatus: {r.status} ({r.motivo})\n")
    print(r.resposta, "\n")
    for e in r.evidencias:
        mark = "*" if e.ref in r.citadas else " "  # * = evidência que o LLM citou na resposta
        print(f"{mark}[{e.ref}] sim={e.similarity:.3f} nota={e.meta['nota']} :: {e.texto[:160]}")
