---
name: t2ia-multiagent
description: Workflow multi-agente simplificado para o projeto T2IA (PUCRS - RN + AG + Minimax) com memória compartilhada no Obsidian.
---

# T2IA Multi-Agent Workflow

## Hierarquia
1. **CTO (Usuário):** Define prioridades e aprova entregas.
2. **Tech Lead (Coordenador):** Planeja, delega para os 2 workers (`ia_core_dev` e `frontend_qa_dev`), revisa e consolida logs.
3. **Workers:**
   - `ia_core_dev`: `src/game.py`, `src/neural_net.py`, `src/genetic_algo.py`
   - `frontend_qa_dev`: `src/app.py`, `main.py`, `tests/test_t2ia.py`

## Regra Obrigatória de Memória Compartilhada
1. **Antes de iniciar:** Ler `obsidian-t2ia/04-Logs-de-Agentes/LOG-MESTRE.md` e `obsidian-t2ia/01-Arquitetura-e-ADRs/README.md`.
2. **Ao finalizar:** Adicionar entrada ao final de `obsidian-t2ia/04-Logs-de-Agentes/LOG-MESTRE.md` (nunca apagar entradas anteriores) com data/hora, agente, resumo, arquivos alterados e testes executados.
