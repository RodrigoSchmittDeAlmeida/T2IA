"""
Algoritmo Genético (AG) para treinar os pesos da MLP no Jogo da Velha.

Características principais:
- Cromossomos de valores reais (float/double) representando pesos e biases da MLP.
- Avaliação contra o jogador Minimax (Médio, Difícil ou Misto/Currículo), sempre
  alternando quem inicia a partida (X vs O).
- Elitismo + Seleção por Torneio.
- Cruzamento BLX-alpha para valores reais.
- Mutação gaussiana.
"""

import random

from .game import PLAYER_O, PLAYER_X, TicTacToe, MinimaxPlayer
from .neural_net import MLP


class GeneticAlgorithm:
    """Motor evolutivo para otimização dos pesos da Rede Neural MLP."""

    def __init__(
        self,
        population_size=80,
        hidden_size=18,
        mutation_rate=0.04,
        mutation_std=0.18,
        elitism_count=4,
        tournament_size=3,
        max_generations=120,
        stagnation_limit=30,
        blx_alpha=0.25,
        seed=None,
    ):
        self.population_size = population_size
        self.hidden_size = hidden_size
        self.mutation_rate = mutation_rate
        self.mutation_std = mutation_std
        self.elitism_count = min(elitism_count, population_size)
        self.tournament_size = min(tournament_size, population_size)
        self.max_generations = max_generations
        self.stagnation_limit = stagnation_limit
        self.blx_alpha = blx_alpha

        self._rng = random.Random(seed)
        self._mlp = MLP(input_size=9, hidden_size=hidden_size, output_size=9)
        self._minimax = MinimaxPlayer(rng=self._rng)
        self.chromosome_length = self._mlp.num_weights()

        self.population = [
            [self._rng.uniform(-1.0, 1.0) for _ in range(self.chromosome_length)]
            for _ in range(self.population_size)
        ]

        self.generation = 0
        self.best_fitness = float("-inf")
        self.best_weights = list(self.population[0])
        self.stagnation_counter = 0
        self._last_mode = None
        self.history = []

    def _play_single_game(self, net_player, minimax_mode):
        """Executa uma partida entre a MLP configurada e o Minimax."""
        game = TicTacToe()
        turn = PLAYER_X  # X sempre começa no Jogo da Velha
        score = 0.0

        while True:
            if turn == net_player:
                move = self._mlp.choose_move(game, net_player, allow_invalid=True)
                if not game.make_move(move, net_player):
                    # Jogada inválida: penalidade severa e fim imediato da partida
                    return score - 120.0, "invalid"
                # Recompensa por realizar uma jogada válida
                score += 5.0
            else:
                move = self._minimax.choose_move(game, turn, mode=minimax_mode)
                game.make_move(move, turn)

            winner = game.check_winner()
            if winner is not None:
                if winner == net_player:
                    return score + 100.0, "win"
                if winner == 0:
                    return score + 35.0, "draw"
                return score - 60.0, "loss"

            turn = -turn

    def evaluate_chromosome(self, weights, mode="dificil", games_count=None):
        """Avalia um cromossomo contra o Minimax e retorna (fitness_medio, stats).

        Modos suportados:
        - "medio": joga `games_count` (padrão 6) partidas alternando X e O.
        - "dificil": joga `games_count` (padrão 2) partidas alternando X e O.
        - "misto": metade das partidas no "medio" e metade no "dificil".
        """
        self._mlp.set_weights(weights)

        if games_count is None:
            if mode == "medio":
                games_count = 6
            elif mode == "dificil":
                games_count = 2
            else:
                games_count = 6

        wins = 0
        draws = 0
        losses = 0
        invalid_moves = 0
        total_score = 0.0

        for g in range(games_count):
            net_player = PLAYER_X if (g % 2 == 0) else PLAYER_O
            if mode == "misto":
                game_mode = "medio" if g < (games_count // 2) else "dificil"
            else:
                game_mode = mode

            game_score, outcome = self._play_single_game(net_player, game_mode)
            total_score += game_score

            if outcome == "win":
                wins += 1
            elif outcome == "draw":
                draws += 1
            elif outcome == "loss":
                losses += 1
            else:
                losses += 1
                invalid_moves += 1

        avg_fitness = total_score / games_count
        stats = {
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "invalid_moves": invalid_moves,
        }
        return avg_fitness, stats

    def _tournament_select(self, evaluated):
        """Seleciona um indivíduo por torneio entre `tournament_size` candidatos."""
        candidates = self._rng.sample(evaluated, self.tournament_size)
        winner = max(candidates, key=lambda item: item[0])
        return winner[1]

    def _blx_alpha_crossover(self, parent1, parent2):
        """Cruzamento BLX-alpha para cromossomos de valores reais."""
        alpha = self.blx_alpha
        child = [0.0] * self.chromosome_length
        for i in range(self.chromosome_length):
            p1 = parent1[i]
            p2 = parent2[i]
            c_min = p1 if p1 < p2 else p2
            c_max = p2 if p1 < p2 else p1
            span = (c_max - c_min) * alpha
            child[i] = self._rng.uniform(c_min - span, c_max + span)
        return child

    def _mutate(self, chromosome):
        """Aplica mutação gaussiana gene a gene com probabilidade `mutation_rate`."""
        rate = self.mutation_rate
        std = self.mutation_std
        for i in range(self.chromosome_length):
            if self._rng.random() < rate:
                val = chromosome[i] + self._rng.gauss(0.0, std)
                # Limite suave para evitar saturação extrema na função tanh
                if val > 10.0:
                    val = 10.0
                elif val < -10.0:
                    val = -10.0
                chromosome[i] = val
        return chromosome

    def step_generation(self, mode="dificil"):
        """Executa 1 geração completa do AG e retorna métricas detalhadas.

        Se `mode == "curriculo"`, treina a primeira metade de `max_generations`
        no modo "medio" e a segunda metade no modo "dificil".
        """
        self.generation += 1

        if mode == "curriculo":
            half = max(1, self.max_generations // 2)
            effective_mode = "medio" if self.generation <= half else "dificil"
        else:
            effective_mode = mode

        # Ao trocar de modo no currículo (ex.: Médio -> Difícil), reinicia a
        # referência de melhor fitness e estagnação para avaliar o novo modo de forma justa.
        if self._last_mode is not None and effective_mode != self._last_mode:
            self.best_fitness = float("-inf")
            self.stagnation_counter = 0
        self._last_mode = effective_mode

        evaluated = []
        total_wins = 0
        total_draws = 0
        total_losses = 0
        total_invalids = 0
        fitness_sum = 0.0

        for chrom in self.population:
            fit, stats = self.evaluate_chromosome(chrom, mode=effective_mode)
            evaluated.append((fit, chrom))
            fitness_sum += fit
            total_wins += stats["wins"]
            total_draws += stats["draws"]
            total_losses += stats["losses"]
            total_invalids += stats["invalid_moves"]

        # Ordena do maior fitness para o menor
        evaluated.sort(key=lambda item: item[0], reverse=True)

        gen_best_fitness, gen_best_chrom = evaluated[0]
        avg_fitness = fitness_sum / self.population_size

        if gen_best_fitness > self.best_fitness:
            self.best_fitness = gen_best_fitness
            self.best_weights = list(gen_best_chrom)
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1

        # No modo currículo, não interrompe por estagnação durante a fase inicial (Médio)
        in_warmup = mode == "curriculo" and self.generation <= max(1, self.max_generations // 2)
        converged = (
            self.generation >= self.max_generations
            or (not in_warmup and self.stagnation_counter >= self.stagnation_limit)
        )

        # Elitismo: preserva os melhores indivíduos intactos
        next_population = [
            list(evaluated[i][1]) for i in range(self.elitism_count)
        ]

        # Preenche o restante da população com torneio + cruzamento + mutação
        while len(next_population) < self.population_size:
            p1 = self._tournament_select(evaluated)
            p2 = self._tournament_select(evaluated)
            child = self._blx_alpha_crossover(p1, p2)
            self._mutate(child)
            next_population.append(child)

        self.population = next_population

        metrics = {
            "generation": self.generation,
            "best_fitness": round(gen_best_fitness, 2),
            "global_best_fitness": round(self.best_fitness, 2),
            "avg_fitness": round(avg_fitness, 2),
            "wins": total_wins,
            "draws": total_draws,
            "losses": total_losses,
            "invalid_moves": total_invalids,
            "best_weights": list(self.best_weights),
            "converged": converged,
            "mode": effective_mode,
            # Fitness de todos os indivíduos da geração (ordem decrescente),
            # usado pelo front-end para visualizar a população.
            "population_fitness": [round(item[0], 2) for item in evaluated],
            "stagnation": self.stagnation_counter,
        }
        self.history.append(metrics)
        return metrics


def evaluate_accuracy(weights, games_per_mode=20, hidden_size=18, seed=42):
    """Avalia uma rede neural treinada contra o Minimax nos modos Médio e Difícil.

    Retorna um dicionário com métricas de acurácia (% sem derrota, % de vitórias,
    % de jogadas válidas e contagens) para cada modo e para o resultado geral.
    """
    mlp = MLP(input_size=9, hidden_size=hidden_size, output_size=9, weights=weights)
    minimax = MinimaxPlayer(rng=random.Random(seed))

    results = {}
    overall_wins = 0
    overall_draws = 0
    overall_losses = 0
    overall_invalids = 0
    overall_net_moves = 0
    overall_valid_moves = 0

    for mode in ("medio", "dificil"):
        wins = 0
        draws = 0
        losses = 0
        invalids = 0
        net_moves = 0
        valid_moves = 0

        for g in range(games_per_mode):
            net_player = PLAYER_X if (g % 2 == 0) else PLAYER_O
            game = TicTacToe()
            turn = PLAYER_X

            while True:
                if turn == net_player:
                    net_moves += 1
                    move = mlp.choose_move(game, net_player, allow_invalid=True)
                    if not game.make_move(move, net_player):
                        invalids += 1
                        losses += 1
                        break
                    valid_moves += 1
                else:
                    move = minimax.choose_move(game, turn, mode=mode)
                    game.make_move(move, turn)

                winner = game.check_winner()
                if winner is not None:
                    if winner == net_player:
                        wins += 1
                    elif winner == 0:
                        draws += 1
                    else:
                        losses += 1
                    break

                turn = -turn

        non_loss_rate = ((wins + draws) / games_per_mode) * 100.0 if games_per_mode else 0.0
        win_rate = (wins / games_per_mode) * 100.0 if games_per_mode else 0.0
        valid_move_rate = (valid_moves / net_moves) * 100.0 if net_moves else 0.0

        results[mode] = {
            "games": games_per_mode,
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "invalid_games": invalids,
            "win_rate": round(win_rate, 2),
            "non_loss_rate": round(non_loss_rate, 2),
            "valid_move_rate": round(valid_move_rate, 2),
        }

        overall_wins += wins
        overall_draws += draws
        overall_losses += losses
        overall_invalids += invalids
        overall_net_moves += net_moves
        overall_valid_moves += valid_moves

    total_games = games_per_mode * 2
    results["geral"] = {
        "games": total_games,
        "wins": overall_wins,
        "draws": overall_draws,
        "losses": overall_losses,
        "invalid_games": overall_invalids,
        "win_rate": round((overall_wins / total_games) * 100.0, 2) if total_games else 0.0,
        "non_loss_rate": round(((overall_wins + overall_draws) / total_games) * 100.0, 2) if total_games else 0.0,
        "valid_move_rate": round((overall_valid_moves / overall_net_moves) * 100.0, 2) if overall_net_moves else 0.0,
    }
    return results
