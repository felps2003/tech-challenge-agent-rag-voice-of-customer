# Interface web em Streamlit.   Rodar:  streamlit run app.py
import streamlit as st

from src import config
from src.index import get_vectorstore
from src.rag import Filters, answer

# botões de exemplo; a última pergunta é de propósito fora do escopo, pra mostrar a recusa
EXEMPLOS = [
    "Quais são os principais problemas relatados pelos clientes?",
    "O que os clientes relatam sobre problemas relacionados à entrega?",
    "Quais aspectos da experiência de compra são mais elogiados?",
    "Quais padrões aparecem nas avaliações de clientes insatisfeitos?",
    "Quais avaliações sustentam que há problemas recorrentes de prazo de entrega?",
    "Qual é a capital da França?",
]

st.set_page_config(page_title="Voice of Customer RAG", layout="wide")
st.title("Voice of Customer Intelligence (Olist)")
st.caption("Pergunte em linguagem natural. A resposta só usa avaliações recuperadas, e você vê quais.")


@st.cache_data(show_spinner=False)  # lê as categorias uma vez e guarda, senão relê a cada clique
def listar_categorias() -> list[str]:
    # lê em páginas de 5000: pedir tudo de uma vez estoura o limite de variáveis do SQLite do Chroma
    vs = get_vectorstore()
    total = vs._collection.count()
    cats: set[str] = set()
    for off in range(0, total, 5000):
        pagina = vs.get(include=["metadatas"], limit=5000, offset=off)["metadatas"]
        cats.update(m["categoria"] for m in pagina)
    return sorted(cats)


@st.cache_resource(show_spinner="Carregando modelo de embeddings e índice BM25 (só na 1ª vez)...")
def aquecer():
    # carrega tudo que é pesado já na abertura, senão a 1ª pergunta demora ~30s (ruim pra demo)
    from src import lexical
    from src.embeddings import E5Embeddings

    E5Embeddings().embed_query("aquecimento")  # carrega o modelo de embeddings
    lexical._load()                            # monta o índice BM25
    return True


try:
    aquecer()
    n_docs = get_vectorstore()._collection.count()
    categorias_disponiveis = listar_categorias()
except Exception as e:  # acontece se o índice ainda não foi criado
    st.error(f"Índice não encontrado. Rode `python -m src.prepare` e `python -m src.index`. ({e})")
    st.stop()

# ---- barra lateral: filtros por metadados (desafio adicional) e opções do pipeline ----
with st.sidebar:
    st.header("Filtros por metadados")
    nota = st.slider("Nota da avaliação", 1, 5, (1, 5))
    categorias = st.multiselect("Categoria do produto", categorias_disponiveis)
    anos = st.multiselect("Ano da compra", [2016, 2017, 2018])
    atrasada = st.checkbox("Somente entregas atrasadas")
    st.divider()
    st.header("Pipeline")
    auto = st.checkbox("Filtros automáticos (o LLM infere a nota da pergunta)", True)
    rewrite = st.checkbox("Reescrever a consulta com o LLM", True)
    hybrid = st.checkbox("Recuperação híbrida (vetor + BM25 + RRF)", True)
    # passo de 0.005 para o valor calibrado (0.905) cair exatamente num ponto do slider
    min_sim = st.slider("Similaridade mínima", 0.5, 0.95, config.MIN_SIMILARITY, 0.005,
                        format="%.3f")
    st.caption(f"{n_docs:,} avaliações indexadas · LLM: {config.LLM_PROVIDER} · "
               f"embeddings: {config.EMBEDDING_MODEL}")

# guarda a pergunta no session_state pra os botões de exemplo conseguirem preencher a caixa
if "q" not in st.session_state:
    st.session_state.q = ""

colunas = st.columns(3)
for i, ex in enumerate(EXEMPLOS):
    if colunas[i % 3].button(ex, use_container_width=True, key=f"ex{i}"):
        st.session_state.q = ex

pergunta = st.text_input("Sua pergunta", key="q")

if st.button("Perguntar", type="primary") and pergunta.strip():
    # filtros marcados à mão; slider em 1 ou 5 significa "sem limite" naquele lado
    manual = Filters(
        nota_min=nota[0] if nota[0] > 1 else None,
        nota_max=nota[1] if nota[1] < 5 else None,
        categorias=categorias,
        anos=anos,
        entrega_atrasada=True if atrasada else None,
    )
    with st.spinner("Recuperando evidências e gerando a resposta..."):
        try:
            r = answer(pergunta, manual, auto_filters=auto, rewrite=rewrite,
                       min_sim=min_sim, hybrid=hybrid)
        except Exception as e:
            st.error(f"Erro: {e}")
            st.stop()

    if r.status == "ok":
        st.subheader("Resposta")
        st.markdown(r.resposta)
    else:
        st.warning(r.resposta)  # recusa: mostra o motivo técnico logo abaixo
        st.caption(f"Motivo técnico: {r.motivo}")

    # painel que mostra cada passo do pipeline (bom para explicar no vídeo)
    with st.expander("Como o pipeline processou a pergunta", expanded=True):
        st.markdown(f"**1. Pergunta original:** {r.pergunta}")
        st.markdown(f"**2. Consulta de busca (após a reescrita):** {r.consulta}")
        st.markdown(f"**3. Filtros aplicados:** {r.filtros.describe()}")
        st.markdown(f"**4. Evidências acima do limiar ({min_sim:.3f}):** {len(r.evidencias)}")
        st.markdown(f"**5. Decisão:** {r.status} - {r.motivo}")
        st.caption(f"Latência: {r.latencia_s:.1f}s · etapas (s): {r.etapas_s}")

    if r.evidencias:
        st.subheader(f"Evidências ({len(r.evidencias)})")
        for e in r.evidencias:
            m = e.meta
            marca = "citada" if e.ref in r.citadas else "recuperada"
            titulo = (f"[{e.ref}] {marca} · nota {m['nota']} · {m['categoria']} · "
                      f"sim {e.similarity:.3f} · via {e.via or 'vetor'}")
            with st.expander(titulo):
                st.write(e.texto)
                atraso = m["dias_atraso"] if m["dias_atraso"] != -999 else "n/d"  # -999 = sem data
                st.caption(f"review_id: {m['review_id']} · order_id: {m['order_id']} · "
                           f"compra: {m['mes_compra']} · atraso (dias): {atraso}")

    if r.contexto:
        with st.expander("Contexto enviado ao LLM"):
            st.code(r.contexto, language="text")
