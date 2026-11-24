# Relatório (Estrutura para PPT) e Roteiro de Vídeo — T2IA

> **Disciplina:** PUCRS – Inteligência Artificial | **Profa.:** Silvia Moraes  
> **Trabalho:** T2 – Aprendizagem por Reforço: Rede Neural + Algoritmo Genético + Minimax

---

## PARTE A — CONTEÚDO DO RELATÓRIO (SLIDE POR SLIDE PARA O PPT — 2,0 pts)

### Slide 1 — Capa
- **Título:** Aprendizagem por Reforço no Jogo da Velha: MLP + Algoritmo Genético + Minimax
- **Disciplina:** Inteligência Artificial — PUCRS (Profa. Silvia Moraes)
- **Integrantes:** [Nome do Integrante 1] e [Nome do Integrante 2]
- **Stack:** Python 3.12+ (biblioteca padrão + Tkinter, sem bibliotecas externas de ML)

---

### Slide 2 — Objetivo e Arquitetura Geral
- **Objetivo:** Fazer uma Rede Neural MLP aprender a jogar Jogo da Velha por reforço, **sem usar backpropagation**, evoluindo seus pesos com um Algoritmo Genético (AG) contra o Minimax.
- **Fluxo da Solução:**
  1. O tabuleiro (`TicTacToe`) envia o estado codificado para a MLP.
  2. A MLP propaga (*forward pass*) e escolhe uma célula (`0..8`).
  3. O Minimax joga em modo **Médio** ou **Difícil**, alternando quem inicia (`X` ou `O`).
  4. Ao fim da partida, a função de aptidão pontua o desempenho do cromossomo.
  5. O AG aplica **Elitismo + Torneio + Cruzamento BLX-α + Mutação Gaussiana** para gerar a próxima população.

---

### Slide 3 — Adaptação do Minimax e Controle do Jogo
- **Controle Algorítmico (`src/game.py`):**
  - Tabuleiro 3×3 (`0` = vazio, `1` = X, `-1` = O).
  - Verificação de vitória (linhas, colunas, diagonais), empate e jogada válida a cada turno.
- **Minimax com 2 Modos (`MinimaxPlayer`):**
  - **Modo Médio:** 50% de chance de executar o Minimax ótimo e 50% de escolher uma jogada aleatória válida (permite que a rede vença e aprenda no início).
  - **Modo Difícil:** 100% Minimax (Negamax com memoização em cache para acelerar o treino).
- **Alternância de Turnos:** Em todas as avaliações, metade das partidas a rede começa como `X` e metade o Minimax começa como `X`.

---

### Slide 4 — Topologia da Rede Neural MLP (`src/neural_net.py`)
- **Arquitetura:** MLP de 2 camadas (sem *backpropagation*, apenas propagação):
  - **Entrada:** 9 neurônios com codificação relativa ao jogador da rede (`+1.0` = peça da rede, `-1.0` = oponente, `0.0` = vazio). *Escolha de projeto:* permite que a mesma rede jogue tanto como `X` quanto como `O`.
  - **Camada Oculta:** 18 neurônios, função de ativação `tanh`.
  - **Camada de Saída:** 9 neurônios (`tanh`), um escore por casa do tabuleiro. A jogada é `argmax(saída)`.
- **Total de Pesos (Tamanho do Cromossomo):**
  - `(9 × 18) + 18 + (18 × 9) + 9 = 351` valores reais (`float`/`double`).

---

### Slide 5 — Estrutura e Operadores do Algoritmo Genético (`src/genetic_algo.py`)
- **População Inicial:** `N = 80` cromossomos com pesos aleatórios em `[-1.0, 1.0]`.
- **Função de Aptidão (Fitness):**
  - `+5` por jogada válida realizada.
  - `+100` por vitória | `+35` por empate | `-60` por derrota.
  - `-120` e encerramento imediato da partida em caso de **jogada inválida** (casa já ocupada).
  - No modo Médio (aleatório), cada cromossomo joga **6 partidas** (3 como X, 3 como O) para reduzir ruído; no Difícil, **2 partidas** (1 como X, 1 como O).
- **Operadores Genéticos:**
  - **Seleção:** Elitismo (`4` melhores preservados) + Torneio (`k = 3`).
  - **Cruzamento (Valores Reais):** `BLX-α` (`α = 0.25`) entre dois pais.
  - **Mutação:** Perturbação gaussiana (`taxa = 0.04`, `σ = 0.18`), limitada em `[-10, 10]`.
- **Ordem dos Modos e Parada:**
  - **Currículo (`Médio → Difícil`):** Primeira metade das gerações no modo Médio; segunda metade no modo Difícil (reiniciando a referência de estagnação na troca de modo).
  - Parada por **máximo de gerações** (`120`) ou **convergência** (`30` gerações sem melhora na fase final).

---

### Slide 6 — Etapas de Desenvolvimento e Front-End Mínimo (`src/app.py`)
- **Divisão em 2 Partes:** Motor de IA (`src/game.py`, `neural_net.py`, `genetic_algo.py`) + Front-end Tkinter e testes (`src/app.py`, `main.py`, `tests/`).
- **3 Módulos no Front-End:**
  1. **Você vs Minimax:** Escolha de dificuldade (`Médio`/`Difícil`), alternância de quem começa e placar.
  2. **Treinamento (RN vs Minimax):** Parâmetros configuráveis, execução contínua ou **Passo (1 geração)**, gráfico de evolução do fitness, gráfico de barras da população (destacando a elite), tabela por geração e partida de demonstração ao vivo.
  3. **Você vs Rede Treinada + Acurácia:** Mapa de calor das 9 saídas da rede, cálculo em tempo real de acurácia de jogadas válidas e taxa de não-derrota, avaliação automática contra Minimax e salvar/carregar pesos em `.json`.

---

### Slide 7 — Análise de Resultados (Experimento Padrão: `seed=42`, `N=80`, Currículo)
- **Evolução das Gerações:**

| Geração | Modo | Melhor Fitness | Fitness Médio | Vitórias | Empates | Derrotas | Inválidas |
|---:|---|---:|---:|---:|---:|---:|---:|
| 1 | Médio | -73,33 | -107,16 | 2 | 0 | 478 | 441 |
| 30 | Médio | 50,00 | -71,00 | 61 | 1 | 418 | 332 |
| 60 | Médio | 50,83 | -48,33 | 120 | 0 | 360 | 308 |
| 61 | Difícil | -47,50 | -84,47 | 0 | 0 | 160 | 92 |
| 90 | Difícil | 57,50 | -3,69 | 0 | 87 | 73 | 38 |
| 108 (conv.) | Difícil | **57,50** | **9,09** | 0 | **105** | 55 | 36 |

- **Acurácia Final (`evaluate_accuracy` — 40 partidas):**
  - **Contra Minimax Difícil (20 jogos):** **100,0% de não-derrota** (20 empates, 0 derrotas, 0 inválidas) e **100,0% de jogadas válidas**.
  - **Contra Minimax Médio (20 jogos):** 30,0% de não-derrota (6 empates) e 82,43% de jogadas válidas.
  - **Geral (40 jogos):** **65,0% de não-derrota** e **92,07% de jogadas válidas**.

---

### Slide 8 — Considerações Finais
- **Pontos Positivos:**
  - A rede aprendeu do zero (sem *backpropagation*) a respeitar casas ocupadas e atingir **100% de empates contra o Minimax Difícil** (resultado ótimo no Jogo da Velha contra jogador perfeito).
  - Currículo `Médio → Difícil` + reinício de estagnação na troca de fase evitou parada prematura.
  - Interface didática permite acompanhar cada geração, a população e testar a rede ao vivo.
- **Pontos Negativos / Limitações:**
  - Ao especializar na segunda metade apenas contra o Minimax Difícil (determinístico), a rede memoriza bem as linhas ótimas, mas perde parte da generalização contra jogadas aleatórias do modo Médio (*esquecimento catastrófico* parcial).
  - Como melhoria futura, poderia ser usada uma fase final mista determinística (avaliando contra um conjunto fixo de partidas médias + difíceis) para manter 100% de jogadas válidas em qualquer tabuleiro.

---
---

## PARTE B — ROTEIRO PARA O VÍDEO DE APRESENTAÇÃO (1,0 pt)

> **Regras do enunciado:**
> - **Ambos os integrantes devem aparecer no vídeo falando sobre o trabalho.**
> - **Não usar IA para narrar.**
> - Duração sugerida: **4 a 6 minutos** (objetivo, claro e mostrando o sistema funcionando).

### Bloco 1 — Abertura (Integrante 1 + Integrante 2 juntos — ~30s)
- **Integrante 1:** *"Olá professora Silvia, eu sou o [Nome 1]..."*
- **Integrante 2:** *"...e eu sou o [Nome 2]. Vamos apresentar o T2 de Inteligência Artificial: uma solução de aprendizagem por reforço para o Jogo da Velha usando Rede Neural MLP, Algoritmo Genético e Minimax, implementada em Python puro com Tkinter."*

### Bloco 2 — Motor de IA: Minimax, MLP e AG (Integrante 1 — ~2min)
- *(Mostrar rapidamente os arquivos `src/game.py`, `src/neural_net.py` e `src/genetic_algo.py` na tela)*
- **Fala do Integrante 1:**
  1. **Jogo e Minimax (`game.py`):** *"O controle do tabuleiro é algorítmico: a cada jogada verificamos vitória, empate ou casa ocupada. Adaptamos o Minimax para dois modos: no Médio ele usa Minimax em 50% das jogadas e aleatório nos outros 50%; no Difícil ele usa 100% Minimax ótimo com cache."*
  2. **Rede Neural (`neural_net.py`):** *"Construímos uma MLP de 2 camadas: 9 entradas, 18 neurônios ocultos com ativação tanh e 9 saídas, totalizando 351 pesos. Não há backpropagation — implementamos só a propagação (forward). A entrada usa codificação relativa (+1 para a rede, -1 para o oponente, 0 vazio), permitindo que a rede jogue tanto como X quanto como O."*
  3. **Algoritmo Genético (`genetic_algo.py`):** *"Cada cromossomo é o vetor de 351 pesos em ponto flutuante. Na função de aptidão, alternamos quem começa jogando, premiamos jogadas válidas, empates (+35) e vitórias (+100), e penalizamos derrotas (-60) e jogadas inválidas (-120 encerrando o jogo). Usamos elitismo, torneio, cruzamento BLX-alpha para valores reais e mutação gaussiana, no currículo Médio seguido de Difícil."*

### Bloco 3 — Demonstração ao Vivo no Front-End e Resultados (Integrante 2 — ~2min30s)
- *(Rodar `python main.py` e compartilhar a tela da janela Tkinter)*
- **Fala do Integrante 2:**
  1. **Aba 1 (Você vs Minimax):** *"Aqui na primeira aba podemos jogar contra o Minimax no modo Médio ou Difícil, alternando quem começa."* *(Jogar 1 partida rápida mostrando o bloqueio/empate).*
  2. **Aba 2 (Treinamento RN vs Minimax):** *"Na segunda aba configuramos a população (N=80), gerações (120), mutação e o currículo Médio para Difícil. Podemos clicar em 'Passo' para ver geração por geração, ou 'Iniciar' para acompanhar o gráfico de fitness, o gráfico da população com a elite em laranja e a partida de demonstração ao vivo."* *(Clicar em 'Passo' 2 vezes e depois 'Iniciar' por alguns segundos ou carregar a rede treinada).*
  3. **Aba 3 (Você vs Rede Treinada + Acurácia):** *"Na terceira aba jogamos contra a rede treinada, vendo o mapa de calor das 9 saídas e a acurácia em tempo real. Clicando em 'Avaliar Acurácia vs Minimax', vemos que no modo Difícil a rede atinge 100% de empates sem jogadas inválidas, e no geral chega a 65% de não-derrota e 92% de jogadas válidas."* *(Mostrar a tabela e jogar 1 partida contra a rede).*

### Bloco 4 — Conclusão (Ambos — ~30s)
- **Integrante 1:** *"Como ponto positivo, provamos que o AG consegue treinar uma MLP do zero para empatar 100% contra o Minimax ótimo sem backpropagation."*
- **Integrante 2:** *"Como limitação, notamos que o treino focado no Difícil determinístico reduz parte da generalização contra jogadas aleatórias do Médio, o que documentamos nas considerações finais. Obrigado!"*
