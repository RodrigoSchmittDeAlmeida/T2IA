# Contexto Completo do Projeto T2IA (Para Enviar ao Colega)

> **Disciplina:** PUCRS – Inteligência Artificial (Profa. Silvia Moraes)  
> **Tema:** T2 – Aprendizagem por Reforço: Rede Neural + Algoritmo Genético + Minimax (Jogo da Velha)  
> **Stack:** Python 3.12+ puro (apenas biblioteca padrão + `tkinter`, **zero** `pip install`).

---

## 1. O que JÁ ESTÁ PRONTO (Parte 1 — Motor de IA)
A **Parte 1** já está implementada em `src/` e testada em `tests/test_part1.py`:
- `src/__init__.py` — Exporta `TicTacToe`, `MinimaxPlayer`, `MLP`, `GeneticAlgorithm`, `evaluate_accuracy`.
- `src/game.py`:
  - `TicTacToe`: tabuleiro de 9 posições (`0` vazio, `1` X, `-1` O), `reset()`, `available_moves()`, `make_move(pos, player)`, `check_winner()` (`1`, `-1`, `0` empate, `None` em andamento), `is_game_over()`, `encode_for_player(player)`, `copy()`.
  - `MinimaxPlayer`: `choose_move(board, player, mode="dificil")` nos modos `"medio"` (50% Minimax / 50% aleatório) e `"dificil"` (100% Minimax ótimo).
- `src/neural_net.py`:
  - `MLP(input_size=9, hidden_size=18, output_size=9, weights=None)`: apenas *forward pass* (`math.tanh`), sem *backpropagation*, `num_weights()` (`351` no padrão), `set_weights(weights)`, `get_weights()`, `forward(inputs)`, `choose_move(board, player, allow_invalid=True)`.
- `src/genetic_algo.py`:
  - `GeneticAlgorithm(...)`: população configurável `N`, aptidão alternando quem começa, elitismo + torneio, cruzamento `BLX-alpha` (valores reais), mutação gaussiana, parada por gerações/convergência e `step_generation(mode)`.
  - `evaluate_accuracy(weights, games_per_mode=20, hidden_size=18, seed=42)`.

Para rodar os testes da Parte 1:
```bash
python -m unittest discover -s tests -v
```

---

## 2. O que FALTA FAZER (Parte 2 — Sua Parte: Front-End + Integração)
A **Parte 2** **NÃO foi implementada ainda** — cabe a você criar `src/app.py` e `main.py` usando `tkinter`:
1. **Modo 1 — Usuário vs Minimax (0,3 pt):** Tabuleiro 3x3 clicável, escolha de `X` ou `O` e dificuldade (`"medio"` ou `"dificil"`).
2. **Modo 2 — Treinamento: Rede Neural vs Minimax (0,6 pt):** Configuração de `N`, gerações, mutação e ordem dos modos (`"curriculo"`, `"medio"`, `"dificil"`, `"misto"`), botões `Iniciar`, `Passo (1 Geração)`, `Pausar/Parar`, `Resetar`, chamando `ga.step_generation(mode)` com `root.after(...)` e desenhando o gráfico de evolução em `tk.Canvas`.
3. **Modo 3 — Usuário vs Rede Neural Treinada + Acurácia (0,6 pt):** Liberado após haver `best_weights`, permitindo jogar contra a `MLP` treinada (alternando quem começa), exibindo acurácia acumulada e botão para rodar `evaluate_accuracy(best_weights)`.
4. **`main.py`:** Ponto de entrada na raiz para abrir a janela Tkinter.

---

## 3. Histórico de Logs
- Consulte também `obsidian-t2ia/04-Logs-de-Agentes/LOG-MESTRE.md` para o histórico completo append-only dos agentes.
