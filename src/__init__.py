"""
Pacote principal do motor de IA do projeto T2IA (PUCRS).

Contém:
- game: lógica do Jogo da Velha e jogador Minimax (Médio e Difícil).
- neural_net: Rede Neural MLP de 2 camadas (apenas propagação / forward pass).
- genetic_algo: Algoritmo Genético para evolução dos pesos da MLP.
"""

from .game import TicTacToe, MinimaxPlayer
from .neural_net import MLP
from .genetic_algo import GeneticAlgorithm, evaluate_accuracy

__all__ = [
    "TicTacToe",
    "MinimaxPlayer",
    "MLP",
    "GeneticAlgorithm",
    "evaluate_accuracy",
]
