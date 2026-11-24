# T2IA — Jogo da Velha com Aprendizagem por Reforço (MLP + AG + Minimax)

Trabalho 2 da disciplina de **Inteligência Artificial (PUCRS — Profa. Silvia Moraes)**.  
O objetivo é treinar uma Rede Neural MLP (sem *backpropagation*) para aprender a jogar Jogo da Velha utilizando um Algoritmo Genético (AG) contra o algoritmo Minimax.

---

## Como Executar

O projeto utiliza **apenas Python 3.12+ e biblioteca padrão** (`tkinter`, `math`, `random`, `unittest`), sem necessidade de `pip install`.

```bash
# Abrir a interface gráfica
python main.py

# Rodar todos os testes automatizados
python -m unittest discover -s tests -v
```

---

## Estrutura do Projeto

```text
T2IA/
├── main.py                               # Ponto de entrada da aplicação Tkinter
├── src/
│   ├── __init__.py
│   ├── game.py                           # Regras do Jogo da Velha (TicTacToe) e MinimaxPlayer (Médio/Difícil)
│   ├── neural_net.py                     # Rede Neural MLP 2 camadas (9 → 18 → 9, apenas forward pass)
│   ├── genetic_algo.py                   # Algoritmo Genético (aptidão, elitismo, torneio, BLX-α, mutação)
│   └── app.py                            # Interface gráfica Tkinter com os 3 modos e gráficos de evolução
├── tests/
│   ├── test_part1.py                     # Testes unitários do motor de IA
│   └── test_app.py                       # Testes de fumaça da interface Tkinter
└── docs/
    └── RELATORIO-PPT-E-ROTEIRO-VIDEO.md  # Estrutura do relatório (PPT) e roteiro do vídeo
```

---

## Arquitetura da Solução

1. **Jogo da Velha e Minimax (`src/game.py`):**
   - Controle algorítmico completo do estado do jogo (vitória, empate e jogada válida a cada turno).
   - **Minimax Médio:** 50% de jogadas ótimas via Minimax e 50% aleatórias válidas.
   - **Minimax Difícil:** 100% Minimax (Negamax com cache/memoização).

2. **Rede Neural MLP (`src/neural_net.py`):**
   - Topologia padrão `9 → 18 → 9` (`351` pesos/biases), função de ativação `tanh`.
   - Implementa apenas a propagação (*forward pass*), sem *backpropagation*.
   - Entrada relativa ao jogador atual (`+1.0` para a rede, `-1.0` para o oponente, `0.0` vazio), permitindo jogar tanto como `X` quanto como `O`.

3. **Algoritmo Genético (`src/genetic_algo.py`):**
   - Cromossomos de valores reais (`float`/`double`) representando os pesos da MLP.
   - Avaliação alternando quem inicia (`X` e `O`), premiando jogadas válidas, empates e vitórias, e penalizando derrotas e jogadas inválidas.
   - Seleção por **Elitismo + Torneio**, cruzamento **BLX-α** para reais, **mutação gaussiana**, currículo **Médio → Difícil** e parada por gerações máximas ou convergência.

4. **Front-End Tkinter (`src/app.py`):**
   - **Aba 1 — Você vs Minimax:** partida contra Minimax Médio ou Difícil com alternância de quem começa.
   - **Aba 2 — Treinamento (RN vs Minimax):** parâmetros configuráveis, execução contínua ou passo a passo (*Passo — 1 geração*), gráfico de evolução do fitness, gráfico de barras da população e partida de demonstração ao vivo.
   - **Aba 3 — Você vs Rede Treinada + Acurácia:** partida contra a rede treinada, mapa de calor das 9 saídas, acurácia em tempo real, avaliação formal contra Minimax (`evaluate_accuracy`) e exportação/importação de pesos em `.json`.
