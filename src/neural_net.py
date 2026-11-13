"""
Rede Neural MLP (Multi-Layer Perceptron) de 2 camadas em Python puro.

Importante:
- Não utiliza backpropagation nem gradiente descendente.
- Os pesos e biases são evoluídos exclusivamente pelo Algoritmo Genético.
"""

import math
import random


class MLP:
    """Perceptron Multicamadas com 1 camada oculta (entrada -> oculta -> saída)."""

    def __init__(self, input_size=9, hidden_size=18, output_size=9, weights=None):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self._n_w1 = input_size * hidden_size
        self._n_b1 = hidden_size
        self._n_w2 = hidden_size * output_size
        self._n_b2 = output_size
        self._total_weights = self._n_w1 + self._n_b1 + self._n_w2 + self._n_b2

        if weights is not None:
            self.set_weights(weights)
        else:
            self._weights = [
                random.uniform(-1.0, 1.0) for _ in range(self._total_weights)
            ]

    def num_weights(self):
        """Retorna o número total de pesos e biases do cromossomo (351 no padrão)."""
        return self._total_weights

    def set_weights(self, weights):
        """Define o vetor completo de pesos e biases da rede."""
        if len(weights) != self._total_weights:
            raise ValueError(
                f"Esperados {self._total_weights} pesos, recebidos {len(weights)}."
            )
        self._weights = [float(w) for w in weights]

    def get_weights(self):
        """Retorna uma cópia da lista de pesos e biases atuais."""
        return list(self._weights)

    def forward(self, inputs):
        """Executa a propagação direta (forward pass).

        Usa `math.tanh` na camada oculta e na camada de saída.
        Retorna uma lista de `output_size` floats representando os escores.
        """
        if len(inputs) != self.input_size:
            raise ValueError(
                f"Esperadas {self.input_size} entradas, recebidas {len(inputs)}."
            )

        w = self._weights
        in_sz = self.input_size
        hid_sz = self.hidden_size
        out_sz = self.output_size

        off_b1 = self._n_w1
        off_w2 = off_b1 + self._n_b1
        off_b2 = off_w2 + self._n_w2

        # Camada oculta com ativação tangente hiperbólica
        hidden = [0.0] * hid_sz
        for j in range(hid_sz):
            acc = w[off_b1 + j]
            for i in range(in_sz):
                acc += inputs[i] * w[i * hid_sz + j]
            hidden[j] = math.tanh(acc)

        # Camada de saída com ativação tangente hiperbólica
        outputs = [0.0] * out_sz
        for k in range(out_sz):
            acc = w[off_b2 + k]
            for j in range(hid_sz):
                acc += hidden[j] * w[off_w2 + j * out_sz + k]
            outputs[k] = math.tanh(acc)

        return outputs

    def choose_move(self, board, player, allow_invalid=True):
        """Escolhe uma posição (0..8) para `player` com base na saída da rede.

        Args:
            board: instância de `TicTacToe` ou lista de 9 células (0, 1, -1).
            player: 1 (X) ou -1 (O).
            allow_invalid: se True, retorna argmax global (usado no treino para
                           penalizar jogadas em casas ocupadas); se False, restringe
                           a escolha apenas às células vazias.
        """
        if hasattr(board, "encode_for_player"):
            cells = board.board
            encoded = board.encode_for_player(player)
        else:
            cells = board
            encoded = [float(c * player) for c in cells]

        scores = self.forward(encoded)

        if allow_invalid:
            return max(range(self.output_size), key=lambda idx: scores[idx])

        valid_moves = [i for i, c in enumerate(cells) if c == 0]
        if not valid_moves:
            return None
        return max(valid_moves, key=lambda idx: scores[idx])
