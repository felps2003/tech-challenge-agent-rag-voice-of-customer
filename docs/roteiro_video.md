# Roteiro do vídeo (máx. 7 min)

Grave a tela do Streamlit com o `.env` fora de vista (não mostre a chave).

| Tempo | Conteúdo | O que mostrar |
|---|---|---|
| 0:00–0:45 | **Problema de negócio.** Milhares de avaliações, gestores não conseguem ler tudo. Perguntas que queremos responder. | Slide ou README, seção de objetivo. |
| 0:45–1:45 | **Arquitetura.** Fluxo dos 6 passos; por que RAG e não só LLM (evidência e rastreabilidade). | Diagrama do README. |
| 1:45–2:45 | **Base de conhecimento.** Só avaliações com texto, título+mensagem, limpeza leve (e por que sem stopwords), duplicatas, ausentes, metadados. Mostrar os números impressos por `prepare`. | Terminal com a saída de `src.prepare`. |
| 2:45–3:30 | **Embeddings e índice.** E5 multilíngue local, prefixos query/passage, ChromaDB cosseno. | `src/embeddings.py` e `src/index.py` rapidamente. |
| 3:30–5:30 | **Demo.** (1) "Principais problemas?" (2) "Problemas de entrega" mostrando a consulta reescrita e os filtros automáticos (3) elogios (4) filtro manual de categoria + nota baixa. Em cada uma: abrir 2 evidências e apontar as citadas `[R#]`. | Streamlit, painel "Como o pipeline processou". |
| 5:30–6:15 | **Pergunta sem evidência.** "Qual a capital da França?" → recusa. Explicar as duas barreiras (limiar antes do LLM + `SEM_EVIDENCIA`) e a calibração. | Streamlit + saída de `calibrate`. |
| 6:15–7:00 | **Desafios e decisões.** Gestor fala diferente do cliente (reescrita); filtro dentro da busca; resposta é sobre amostra recuperada, não sobre a base toda; limitações e próximos passos (agregação por tópicos, reranker). | Slide final. |

Dicas: ensaie com cronômetro, deixe as perguntas já copiadas, e mostre o **raciocínio** (decisões e trade-offs), não só a aplicação funcionando.
