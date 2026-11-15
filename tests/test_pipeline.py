# Testes das funções que não dependem de rede, de modelo nem do dataset.
# Rodar:  python -m pytest
import pytest
from pydantic import ValidationError

from src.lexical import tokenize
from src.prepare import build_document, clean_text
from src.rag import (ConsultaReescrita, Evidence, Filters, build_context, build_where,
                     extract_citations, filtros_da_reescrita, is_sufficient, merge_filters,
                     rrf_fuse, select_evidence)


def test_clean_text_trata_ausente_e_ruido():
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""
    # link some, letras repetidas encurtam, espaços e quebras viram um espaço só
    assert clean_text("  ótimooooo  veja http://x.com/a  tudo\n\nbom ") == "ótimooo veja tudo bom"


def test_mask_pii():
    out = clean_text("fale comigo joao@x.com ou (11) 98765-4321, cpf 123.456.789-09")
    assert "joao@x.com" not in out and "98765" not in out and "123.456" not in out
    assert "[EMAIL]" in out and "[CPF]" in out and "[TELEFONE]" in out


def test_build_document():
    assert build_document("Ruim", "Chegou quebrado") == "Ruim. Chegou quebrado"
    assert build_document("Ruim!", "Quebrado") == "Ruim! Quebrado"  # não duplica pontuação
    assert build_document(None, "Só mensagem") == "Só mensagem"
    assert build_document("Só título", None) == "Só título"


def test_build_where():
    assert build_where(Filters()) is None
    assert build_where(Filters(nota_max=2)) == {"nota": {"$lte": 2}}
    w = build_where(Filters(nota_max=2, categorias=["toys"], entrega_atrasada=True))
    assert w == {"$and": [{"nota": {"$lte": 2}}, {"categoria": {"$in": ["toys"]}},
                          {"entrega_atrasada": 1}]}


def test_reescrita_estruturada_valida_a_nota():
    r = ConsultaReescrita(consulta="atraso na entrega", nota_max=2, entrega_atrasada=True)
    f = filtros_da_reescrita(r)
    assert f.nota_max == 2 and f.entrega_atrasada is True
    with pytest.raises(ValidationError):  # nota 9 não existe; o Pydantic barra
        ConsultaReescrita(consulta="x", nota_max=9)


def test_merge_filters_manual_vence():
    m = merge_filters(Filters(nota_max=3), Filters(nota_max=1, nota_min=1))
    assert m.nota_max == 3 and m.nota_min == 1


def _cand(i, sim, texto):
    return {"doc_id": f"d{i}", "texto": texto, "similarity": sim, "meta": {"nota": 1}}


def test_select_evidence_limiar_e_duplicata():
    cands = [_cand(1, 0.90, "Chegou atrasado demais!"), _cand(2, 0.88, "chegou atrasado demais"),
             _cand(3, 0.85, "Produto quebrado"), _cand(4, 0.70, "Nada a ver")]
    ev = select_evidence(cands, top_k=10, min_sim=0.80)
    assert [e.doc_id for e in ev] == ["d1", "d3"]  # d2 é duplicata, d4 está abaixo do limiar
    assert [e.ref for e in ev] == ["R1", "R2"]


def test_select_evidence_pula_baixo_mas_mantem_os_proximos():
    # depois do RRF a ordem pode misturar similaridades, então não pode parar no primeiro baixo
    cands = [_cand(1, 0.70, "a b c"), _cand(2, 0.90, "produto bom")]
    assert [e.doc_id for e in select_evidence(cands, 5, 0.80)] == ["d2"]


def test_is_sufficient():
    assert is_sufficient([], 3)[0] is False
    ev = [Evidence("R1", "d", "t", 0.9, {})] * 2
    assert is_sufficient(ev, 3)[0] is False
    assert is_sufficient(ev, 2)[0] is True


def test_extract_citations_ignora_referencia_invalida():
    ev = [Evidence("R1", "a", "t", 0.9, {}), Evidence("R2", "b", "t", 0.9, {})]
    assert extract_citations("Atraso [R2][R1] e [R9]. Custa R$ 50", ev) == ["R2", "R1"]


def test_rrf_prefere_quem_aparece_nos_dois_rankings():
    s = rrf_fuse([["a", "b", "c"], ["c", "d"]])
    assert s["c"] > s["b"] and s["c"] > s["d"]
    assert s["a"] > s["b"]


def test_tokenize_tira_acento_e_stopword():
    assert tokenize("O produto não chegou no PRAZO!") == ["produto", "chegou", "prazo"]


def test_build_context():
    ev = [Evidence("R1", "a", "chegou quebrado", 0.9,
                   {"nota": 1, "categoria": "toys", "mes_compra": "2017-05"})]
    ctx = build_context(ev)
    assert "[R1]" in ctx and "nota 1" in ctx and "toys" in ctx and "chegou quebrado" in ctx
