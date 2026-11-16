"""Testes de fumaça do front-end Tkinter (pulados se não houver display)."""

import tkinter as tk
import unittest

from src.app import App


class TestApp(unittest.TestCase):
    def setUp(self):
        try:
            self.root = tk.Tk()
        except tk.TclError:
            self.skipTest("Sem display disponível para o Tkinter")
        self.root.withdraw()
        self.app = App(self.root)

    def tearDown(self):
        self.root.destroy()

    def _pump(self, ms):
        """Processa eventos do Tk por `ms` milissegundos."""
        done = []
        self.root.after(ms, lambda: done.append(True))
        while not done:
            self.root.update()

    def _play_until_end(self, tab):
        tab.new_game()
        for _ in range(60):
            if not tab.playing:
                break
            if tab.turn == tab.human:
                tab.on_cell(tab.game.available_moves()[0])
            self._pump(tab.ai_delay_ms + 50)
        self.assertFalse(tab.playing)
        self.assertIsNotNone(tab.game.check_winner())

    def test_user_vs_minimax(self):
        tab = self.app.minimax_tab
        tab.ai_delay_ms = 1
        tab.mode_var.set("dificil")
        for _ in range(2):  # alterna quem começa
            self._play_until_end(tab)
        # Minimax difícil nunca perde
        self.assertEqual(tab.score["human"], 0)
        self.assertEqual(tab.games_played, 2)

    def test_training_and_user_vs_network(self):
        net_tab = self.app.network_tab
        self.assertIsNone(net_tab.mlp)

        tr = self.app.training_tab
        tr.vars["population_size"][0].set("12")
        tr.vars["max_generations"][0].set("4")
        tr.seed_var.set("7")
        tr.demo_var.set(False)
        for _ in range(4):
            tr.step()
        self.assertEqual(len(tr.records), 4)
        self.assertTrue(tr._finished())
        self.assertEqual(len(tr.records[-1]["population_fitness"]), 12)
        tr.step()  # não deve passar do máximo de gerações
        self.assertEqual(len(tr.records), 4)

        self.assertIsNotNone(net_tab.mlp)
        net_tab.ai_delay_ms = 1
        self._play_until_end(net_tab)
        self.assertGreater(net_tab.net_moves, 0)
        net_tab.eval_games.set(4)
        net_tab.run_evaluation()
        self.assertEqual(len(net_tab.eval_tree.get_children()), 3)

        tr.reset()
        self.assertIsNone(tr.ga)
        self.assertEqual(tr.records, [])


if __name__ == "__main__":
    unittest.main()
