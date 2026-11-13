"""
Módulo do jogo da velha (Tic-Tac-Toe) e do adversário Minimax.

Representação do tabuleiro:
- Lista de 9 inteiros (índices 0 a 8, linha por linha).
- 0  = célula vazia
- 1  = jogador X
- -1 = jogador O
"""

import random

EMPTY = 0
PLAYER_X = 1
PLAYER_O = -1

# Todas as 8 combinações vencedoras (3 linhas, 3 colunas, 2 diagonais)
WIN_LINES = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 7, 8),
    (0, 3, 6),
    (1, 4, 7),
    (2, 5, 8),
    (0, 4, 8),
    (2, 4, 6),
)


def _check_board_winner(board):
    """Verifica o vencedor em uma sequência de 9 posições.

    Retorna:
        1  se X venceu
       -1  se O venceu
        0  se houve empate
      None se o jogo ainda está em andamento
    """
    for a, b, c in WIN_LINES:
        val = board[a]
        if val != EMPTY and val == board[b] == board[c]:
            return val
    if EMPTY not in board:
        return 0
    return None


class TicTacToe:
    """Representa uma partida de Jogo da Velha 3x3."""

    def __init__(self):
        self.board = [EMPTY] * 9

    def reset(self):
        """Reinicia o tabuleiro para o estado vazio e retorna a lista de células."""
        self.board = [EMPTY] * 9
        return self.board

    def available_moves(self):
        """Retorna os índices (0..8) das células atualmente vazias."""
        return [i for i, cell in enumerate(self.board) if cell == EMPTY]

    def make_move(self, pos, player):
        """Tenta realizar uma jogada na posição `pos` para `player` (1 ou -1).

        Retorna True se a jogada for válida e aplicada, ou False caso contrário.
        """
        if pos is None or not (0 <= pos < 9):
            return False
        if player not in (PLAYER_X, PLAYER_O):
            return False
        if self.board[pos] != EMPTY:
            return False
        self.board[pos] = player
        return True

    def check_winner(self):
        """Retorna 1 (X), -1 (O), 0 (empate) ou None (jogo continua)."""
        return _check_board_winner(self.board)

    def is_game_over(self):
        """Retorna True se a partida terminou (vitória ou empate)."""
        return self.check_winner() is not None

    def encode_for_player(self, player):
        """Codifica o tabuleiro sob a perspectiva do jogador informado.

        Retorna uma lista de 9 floats:
        - +1.0 para peças do próprio `player`
        - -1.0 para peças do oponente
        -  0.0 para células vazias
        """
        return [float(cell * player) for cell in self.board]

    def copy(self):
        """Retorna uma cópia independente do estado atual do jogo."""
        clone = TicTacToe()
        clone.board = self.board.copy()
        return clone

    def __str__(self):
        symbols = {PLAYER_X: "X", PLAYER_O: "O", EMPTY: "."}
        rows = [
            " ".join(symbols[self.board[r * 3 + c]] for c in range(3))
            for r in range(3)
        ]
        return "\n".join(rows)


class MinimaxPlayer:
    """Jogador Minimax com memoização para os modos Médio e Difícil."""

    def __init__(self, rng=None):
        self._rng = rng or random.Random()
        # Cache compartilhado entre partidas: (tuple(board), player) -> (score, best_move)
        self._cache = {}

    def _negamax(self, board_list, player):
        """Calcula o valor ótimo (Negamax) e a melhor jogada para `player`.

        Usa pontuação com desconto por profundidade (10 - d) de forma relativa,
        garantindo que vitórias mais rápidas tenham escore maior e derrotas mais
        lentas tenham penalidade menor, mantendo a chave de cache independente
        da profundidade absoluta da raiz.
        """
        key = (tuple(board_list), player)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        winner = _check_board_winner(board_list)
        if winner is not None:
            if winner == 0:
                res = (0, None)
            elif winner == player:
                res = (10, None)
            else:
                res = (-10, None)
            self._cache[key] = res
            return res

        best_score = -999
        best_move = None

        for pos in range(9):
            if board_list[pos] != EMPTY:
                continue

            board_list[pos] = player
            opp_score, _ = self._negamax(board_list, -player)
            board_list[pos] = EMPTY

            # Inverte perspectiva e aproxima 1 unidade de 0 para preferir vitórias rápidas
            score = -opp_score
            if score > 0:
                score -= 1
            elif score < 0:
                score += 1

            if score > best_score:
                best_score = score
                best_move = pos

        res = (best_score, best_move)
        self._cache[key] = res
        return res

    def choose_move(self, board, player, mode="dificil"):
        """Escolhe uma jogada para `player` no tabuleiro `board`.

        Args:
            board: instância de `TicTacToe` ou lista de 9 inteiros.
            player: 1 (X) ou -1 (O).
            mode: "medio" (50% Minimax ótimo, 50% aleatório) ou "dificil" (100% ótimo).

        Retorna o índice da célula (0..8) ou None se não houver jogadas disponíveis.
        """
        cells = board.board if hasattr(board, "board") else board
        available = [i for i, v in enumerate(cells) if v == EMPTY]
        if not available:
            return None

        if mode == "medio" and self._rng.random() < 0.5:
            return self._rng.choice(available)

        _, best_move = self._negamax(list(cells), player)
        return best_move
