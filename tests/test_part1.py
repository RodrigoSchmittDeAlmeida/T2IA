import unittest
from src import GeneticAlgorithm, MLP, MinimaxPlayer, TicTacToe, evaluate_accuracy


class TestTicTacToe(unittest.TestCase):
    def test_win_row_col_diagonal(self):
        game = TicTacToe()
        for pos in (0, 1, 2):
            self.assertTrue(game.make_move(pos, 1))
        self.assertEqual(game.check_winner(), 1)
        self.assertTrue(game.is_game_over())

        game.reset()
        for pos in (1, 4, 7):
            self.assertTrue(game.make_move(pos, -1))
        self.assertEqual(game.check_winner(), -1)

        game.reset()
        for pos in (0, 4, 8):
            self.assertTrue(game.make_move(pos, 1))
        self.assertEqual(game.check_winner(), 1)

    def test_draw_and_invalid_move(self):
        game = TicTacToe()
        moves = [(0, 1), (1, -1), (2, 1), (4, -1), (3, 1), (5, -1), (7, 1), (6, -1), (8, 1)]
        for pos, player in moves:
            self.assertTrue(game.make_move(pos, player))
        self.assertEqual(game.check_winner(), 0)
        self.assertFalse(game.make_move(0, 1))

    def test_encode_for_player(self):
        game = TicTacToe()
        game.make_move(0, 1)
        game.make_move(4, -1)
        self.assertEqual(game.encode_for_player(1), [1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
        self.assertEqual(game.encode_for_player(-1), [-1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])


class TestMinimaxPlayer(unittest.TestCase):
    def test_dificil_blocks_and_wins(self):
        mm = MinimaxPlayer()
        # Bloqueia vitória imediata do oponente
        board = [1, 1, 0, 0, -1, 0, 0, 0, 0]
        self.assertEqual(mm.choose_move(board, -1, mode="dificil"), 2)

    def test_medio_returns_valid_move(self):
        mm = MinimaxPlayer()
        board = [1, 0, 0, 0, -1, 0, 0, 0, 0]
        move = mm.choose_move(board, 1, mode="medio")
        self.assertIn(move, [1, 2, 3, 5, 6, 7, 8])


class TestMLP(unittest.TestCase):
    def test_num_weights_and_forward(self):
        mlp = MLP(input_size=9, hidden_size=18, output_size=9)
        self.assertEqual(mlp.num_weights(), 351)
        weights = [0.05] * 351
        mlp.set_weights(weights)
        self.assertEqual(len(mlp.get_weights()), 351)
        out = mlp.forward([0.0] * 9)
        self.assertEqual(len(out), 9)
        self.assertTrue(all(-1.0 <= v <= 1.0 for v in out))

    def test_choose_move_valid_when_masked(self):
        mlp = MLP()
        board = [1, 1, 1, 1, 1, 1, 1, 1, 0]
        self.assertEqual(mlp.choose_move(board, 1, allow_invalid=False), 8)


class TestGeneticAlgorithm(unittest.TestCase):
    def test_step_generation_and_accuracy(self):
        ga = GeneticAlgorithm(population_size=10, max_generations=5, seed=42)
        stats = ga.step_generation(mode="curriculo")
        self.assertEqual(stats["generation"], 1)
        self.assertIn("best_fitness", stats)
        self.assertIn("best_weights", stats)
        acc = evaluate_accuracy(stats["best_weights"], games_per_mode=4, seed=42)
        self.assertIn("medio", acc)
        self.assertIn("dificil", acc)
        self.assertIn("geral", acc)


if __name__ == "__main__":
    unittest.main()
