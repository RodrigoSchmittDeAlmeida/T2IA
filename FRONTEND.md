# Front-End do T2IA (Parte 2)

Este documento descreve o front-end do T2: o que foi feito, como funciona e o que mudou no código da Parte 1 (motor de IA).

## Como rodar

```bash
python3 main.py                               # abre a interface
python3 -m unittest discover -s tests -v      # roda todos os testes (Parte 1 + front-end)
```

Usa só a biblioteca padrão e o Tkinter. Não precisa de `pip install`.

## Arquivos novos

| Arquivo | Conteúdo |
|---|---|
| `main.py` | Ponto de entrada: chama `src.app.run()` |
| `src/app.py` | Toda a interface Tkinter (widgets + 3 abas) |
| `tests/test_app.py` | Testes de fumaça da interface (pulados se não houver display) |

## Estrutura de `src/app.py`

```
App (janela principal, ttk.Notebook)
├── MinimaxTab   ─┐
├── NetworkTab   ─┴─ herdam de HumanVsAITab (lógica comum de partida humano vs IA)
└── TrainingTab      (treinamento da rede pelo AG)

Widgets reutilizáveis:
  BoardCanvas      tabuleiro 3x3 clicável, destaca última jogada, linha vencedora e jogada inválida
  OutputsCanvas    mapa de calor das 9 saídas da rede
  LineChart        gráfico de evolução do fitness
  PopulationChart  barras com o fitness de cada indivíduo de uma geração
```

O estado do jogo é sempre controlado pelas classes da Parte 1. A interface não reimplementa nenhuma regra:

- `TicTacToe.make_move` diz se a jogada é válida;
- `TicTacToe.check_winner` é chamado após **cada** jogada e diz se alguém venceu, se deu empate ou se o jogo continua.

A rede treinada fica guardada em `App.best_weights` / `App.hidden_size`. A aba de treino atualiza esses valores a cada geração (método `App.set_best`), e a aba 3 lê deles.

## Aba 1: Você vs Minimax (0,3 pt)

- Escolha da dificuldade: **Médio** (50% Minimax, 50% aleatório) ou **Difícil** (100% Minimax). A jogada vem de `MinimaxPlayer.choose_move(board, player, mode)`.
- **Quem começa:** Alternar (padrão), Você ou Computador. Quem começa joga com X.
- Placar acumulado e botão para zerar.
- A jogada do computador espera ~350 ms (`after`), para a partida dar para ser acompanhada visualmente.

## Aba 2: Treinamento, rede vs Minimax (0,6 pt)

### Parâmetros configuráveis
Todos são repassados direto para o construtor de `GeneticAlgorithm`:

| Campo na tela | Parâmetro | Padrão |
|---|---|---|
| População (N) | `population_size` | 60 |
| Máx. gerações | `max_generations` | 100 |
| Parada por estagnação | `stagnation_limit` | 25 |
| Elitismo | `elitism_count` | 4 |
| Tamanho do torneio | `tournament_size` | 3 |
| Taxa de mutação | `mutation_rate` | 0.08 |
| Desvio da mutação | `mutation_std` | 0.25 |
| Alpha do BLX-α | `blx_alpha` | 0.5 |
| Neurônios ocultos | `hidden_size` | 18 |
| Semente | `seed` (vazio = aleatória) | — |

**Ordem dos modos do Minimax** (passada para `step_generation(mode)`):

| Opção na tela | Valor passado |
|---|---|
| Currículo (Médio → Difícil) | `"curriculo"` |
| Só Médio | `"medio"` |
| Só Difícil | `"dificil"` |
| Misto (metade de cada) | `"misto"` |

A alternância entre X e O (o Minimax nem sempre começa) é feita pelo próprio AG em `evaluate_chromosome`.

A tela valida os parâmetros antes de criar o AG: os valores precisam estar dentro dos limites, e o elitismo precisa ser menor que a população. Depois que o treino começa, os campos ficam bloqueados até clicar em **Resetar**.

### Controles
- **Iniciar / Continuar:** roda uma geração por vez via `root.after(pausa, ...)`. O treino roda no laço de eventos do Tkinter, sem threads, e a janela não trava, porque cada geração leva ~10–50 ms.
- **Pausar:** cancela o próximo `after`.
- **Passo (1 geração):** executa exatamente uma `step_generation` (é o modo "passo a passo").
- **Resetar:** descarta o AG e limpa gráficos e tabela.
- **Pausa entre gerações (ms):** controla a velocidade do treino.
- O treino para sozinho quando `step_generation` retorna `converged = True`. A tela diz o motivo: máximo de gerações **ou** convergência (N gerações sem melhorar o melhor global).

### Visualização da evolução
1. **Gráfico de fitness:** linhas do melhor da geração, da média da população e do melhor global. O fundo de cada geração tem a cor do modo do Minimax usado (amarelo = Médio, lilás = Difícil, verde = Misto). Assim dá para ver a troca de modo no currículo.
2. **Gráfico da população:** uma barra por indivíduo, ordenada por fitness. Os indivíduos da elite aparecem em laranja.
3. **Tabela por geração:** Geração, Modo, Melhor, Média, Melhor global, Vitórias, Empates, Derrotas, Inválidas e Estagnação. **Clicar numa linha** mostra a população daquela geração no gráfico 2.
4. **Resumo** no painel esquerdo, com os números da última geração.
5. **Partida de demonstração:** um tabuleiro onde a melhor rede (melhor global) joga contra o Minimax no modo da geração atual, animado jogada a jogada. A rede alterna entre X e O a cada partida. Ela escolhe a casa do mesmo jeito que no treino (`allow_invalid=True`): se escolher casa ocupada, a casa fica vermelha e a partida termina como derrota. Pode ser desligada e tem controle de velocidade. O botão "Jogar uma partida agora" força uma nova partida.

## Aba 3: Você vs Rede treinada + acurácia (0,6 pt)

- Fica disponível quando já existe uma rede: depois de pelo menos 1 geração, ou carregando um arquivo.
- **Quem começa:** Alternar (padrão), Você ou Computador.
- **Escolha da jogada da rede:** argmax das 9 saídas de `MLP.forward`.
  - Se o argmax cai numa casa livre, é uma jogada válida.
  - Se cai numa casa ocupada, conta como **jogada inválida**, a casa aparece em vermelho e, para a partida continuar, a rede joga na casa livre com maior saída.
- **Mapa de calor das saídas:** mostra os 9 valores (tanh) da última jogada. A casa escolhida pelo argmax tem borda grossa.
- **Acurácia contra o usuário** (atualizada a cada jogada):
  - *Acurácia de jogadas válidas* = jogadas válidas / jogadas da rede
  - *Taxa de não-derrota* = (vitórias + empates da rede) / partidas
  - *Taxa de vitória* = vitórias da rede / partidas
- **Avaliação automática vs Minimax:** chama `evaluate_accuracy(best_weights, games_per_mode=N)` e mostra a tabela Médio / Difícil / Geral (V, E, D, inválidas, % não-derrota, % jogadas válidas).
- **Salvar / Carregar rede (.json):** grava `{"hidden_size", "weights", "info"}`. Serve para levar uma rede já treinada para a apresentação sem precisar treinar na hora. Ao carregar, o tamanho do vetor de pesos é validado criando uma `MLP`.

## Alterações no código existente (Parte 1)

Houve **uma única alteração**, em `src/genetic_algo.py`, no método `GeneticAlgorithm.step_generation`. Foram acrescentados dois campos ao dicionário `metrics` que o método retorna:

```python
            "converged": converged,
            "mode": effective_mode,
            # Fitness de todos os indivíduos da geração (ordem decrescente),
            # usado pelo front-end para visualizar a população.
            "population_fitness": [round(item[0], 2) for item in evaluated],
            "stagnation": self.stagnation_counter,
        }
```

- `population_fitness`: lista com o fitness de cada indivíduo da geração, já ordenada do maior para o menor (vem de `evaluated`, que já era calculado e ordenado no método). Alimenta o gráfico da população.
- `stagnation`: valor do `stagnation_counter` naquela geração. Aparece na tabela e no resumo.

**Por que foi necessário:** antes, `step_generation` só retornava o melhor e a média. Para "mostrar a evolução da população" (como pede o enunciado), o front-end precisava do fitness de todos os indivíduos. Calcular isso fora do AG exigiria jogar todas as partidas de novo.

**Impacto:** a mudança é só aditiva. Nenhuma lógica do AG mudou (seleção, cruzamento, mutação, aptidão e parada estão iguais), nenhum campo antigo foi alterado ou removido, e os 8 testes da Parte 1 continuam passando sem modificação. Nenhum outro arquivo da Parte 1 (`game.py`, `neural_net.py`, `__init__.py`, `tests/test_part1.py`) foi alterado.

## Testes do front-end (`tests/test_app.py`)

- `test_user_vs_minimax`: joga 2 partidas contra o Minimax Difícil (alternando quem começa) e verifica que o humano nunca vence.
- `test_training_and_user_vs_network`: treina 4 gerações com N=12 usando o botão Passo, confere que o treino para no máximo de gerações e que `population_fitness` tem 12 valores, joga uma partida contra a rede, roda a avaliação automática e testa o Resetar.

Os testes são pulados automaticamente se não houver display gráfico.

## Ponto de atenção no treino (para a Parte 1)

Com os parâmetros padrão e a ordem **Currículo**, o treino parou na geração 50 por estagnação e **não chegou a treinar no modo Difícil**. O motivo é que o melhor global obtido no modo Médio (com vitórias, ~48) é mais alto do que dá para alcançar contra o Difícil, onde o máximo é empatar, então a estagnação nunca zera. A rede resultante teve 12,5% de não-derrota e 21 partidas inválidas em 40 na `evaluate_accuracy`.

Possíveis correções no AG:
- zerar `best_fitness` e `stagnation_counter` quando o modo efetivo muda; ou
- comparar o fitness só dentro do mesmo modo.
