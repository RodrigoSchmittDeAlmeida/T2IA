"""
Front-end (Tkinter) do projeto T2IA — Jogo da Velha com RN + AG + Minimax.

Três modos, um por aba:
1. Você vs Minimax (modos Médio e Difícil).
2. Treinamento: a rede neural aprende jogando contra o Minimax, com a
   evolução do Algoritmo Genético acompanhada geração a geração.
3. Você vs Rede treinada, com cálculo da acurácia da rede.

Todo o controle de estado do jogo (vitória, empate, jogada válida) é feito
pelas classes de `src/game.py`. O treino roda no laço de eventos do Tkinter
via `after(...)`, uma geração por vez.
"""

import json
import random
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from .game import PLAYER_O, PLAYER_X, WIN_LINES, MinimaxPlayer, TicTacToe
from .genetic_algo import GeneticAlgorithm, evaluate_accuracy
from .neural_net import MLP

# ---------------------------------------------------------------------------
# Constantes visuais
# ---------------------------------------------------------------------------
COLOR_BG = "#f5f6f8"
COLOR_PANEL = "#ffffff"
COLOR_GRID = "#334155"
COLOR_X = "#2563eb"
COLOR_O = "#dc2626"
COLOR_WIN = "#bbf7d0"
COLOR_INVALID = "#fecaca"
COLOR_LAST = "#fef9c3"
COLOR_TEXT = "#0f172a"
COLOR_MUTED = "#64748b"

COLOR_BEST = "#16a34a"
COLOR_AVG = "#2563eb"
COLOR_GLOBAL = "#ea580c"
COLOR_ELITE = "#f59e0b"
COLOR_INDIV = "#94a3b8"
MODE_BAND = {"medio": "#fef3c7", "dificil": "#ede9fe", "misto": "#dcfce7"}

MODE_LABEL = {"medio": "Médio", "dificil": "Difícil", "misto": "Misto"}

# Ordem de aplicação dos modos durante o treino (rótulo -> chave do AG)
TRAIN_ORDERS = {
    "Currículo (Médio → Difícil)": "curriculo",
    "Só Médio": "medio",
    "Só Difícil": "dificil",
    "Misto (metade de cada)": "misto",
}

STARTER_OPTIONS = ("Alternar", "Você", "Computador")

SYMBOL = {PLAYER_X: "X", PLAYER_O: "O"}


def winning_line(board):
    """Retorna a tupla de células da linha vencedora, ou None."""
    for line in WIN_LINES:
        a, b, c = line
        if board[a] != 0 and board[a] == board[b] == board[c]:
            return line
    return None


# ---------------------------------------------------------------------------
# Widgets reutilizáveis
# ---------------------------------------------------------------------------
class BoardCanvas(tk.Canvas):
    """Tabuleiro 3x3 desenhado em Canvas, com clique por célula."""

    def __init__(self, master, size=300, on_click=None, **kw):
        super().__init__(
            master, width=size, height=size, bg=COLOR_PANEL,
            highlightthickness=0, **kw,
        )
        self.size = size
        self.cell = size / 3
        self.on_click = on_click
        self.bind("<Button-1>", self._handle_click)
        self.draw([0] * 9)

    def _handle_click(self, event):
        if self.on_click is None:
            return
        col = int(event.x // self.cell)
        row = int(event.y // self.cell)
        if 0 <= col < 3 and 0 <= row < 3:
            self.on_click(row * 3 + col)

    def draw(self, board, last=None, invalid=None):
        self.delete("all")
        c = self.cell
        win = winning_line(board) or ()

        for i in range(9):
            r, col = divmod(i, 3)
            x0, y0 = col * c, r * c
            fill = None
            if i in win:
                fill = COLOR_WIN
            elif i == invalid:
                fill = COLOR_INVALID
            elif i == last:
                fill = COLOR_LAST
            if fill:
                self.create_rectangle(x0, y0, x0 + c, y0 + c, fill=fill, width=0)

        for k in (1, 2):
            self.create_line(k * c, 6, k * c, self.size - 6, width=4,
                             fill=COLOR_GRID, capstyle="round")
            self.create_line(6, k * c, self.size - 6, k * c, width=4,
                             fill=COLOR_GRID, capstyle="round")

        pad = c * 0.22
        for i, v in enumerate(board):
            r, col = divmod(i, 3)
            x0, y0 = col * c + pad, r * c + pad
            x1, y1 = (col + 1) * c - pad, (r + 1) * c - pad
            if v == PLAYER_X:
                self.create_line(x0, y0, x1, y1, width=7, fill=COLOR_X, capstyle="round")
                self.create_line(x0, y1, x1, y0, width=7, fill=COLOR_X, capstyle="round")
            elif v == PLAYER_O:
                self.create_oval(x0, y0, x1, y1, width=7, outline=COLOR_O)

        if invalid is not None:
            r, col = divmod(invalid, 3)
            self.create_text(col * c + c / 2, r * c + c - 12, text="inválida",
                             fill=COLOR_O, font=("Helvetica", 9, "bold"))


class OutputsCanvas(tk.Canvas):
    """Mapa de calor 3x3 com as 9 saídas da rede (tanh, -1..1)."""

    def __init__(self, master, size=150, **kw):
        super().__init__(master, width=size, height=size, bg=COLOR_PANEL,
                         highlightthickness=0, **kw)
        self.size = size
        self.draw(None)

    @staticmethod
    def _color(v):
        # -1 -> vermelho claro, 0 -> branco, +1 -> verde
        v = max(-1.0, min(1.0, v))
        if v >= 0:
            r, g, b = int(255 - 155 * v), int(255 - 40 * v), int(255 - 155 * v)
        else:
            v = -v
            r, g, b = 255, int(255 - 140 * v), int(255 - 140 * v)
        return f"#{r:02x}{g:02x}{b:02x}"

    def draw(self, scores, chosen=None):
        self.delete("all")
        c = self.size / 3
        for i in range(9):
            r, col = divmod(i, 3)
            x0, y0 = col * c, r * c
            fill = self._color(scores[i]) if scores else "#f1f5f9"
            self.create_rectangle(x0 + 1, y0 + 1, x0 + c - 1, y0 + c - 1,
                                  fill=fill, outline="#cbd5e1",
                                  width=3 if i == chosen else 1)
            if scores:
                self.create_text(x0 + c / 2, y0 + c / 2, text=f"{scores[i]:+.2f}",
                                 fill=COLOR_TEXT, font=("Helvetica", 10,
                                 "bold" if i == chosen else "normal"))


class LineChart(tk.Canvas):
    """Gráfico de linhas da evolução do fitness por geração."""

    def __init__(self, master, **kw):
        super().__init__(master, bg=COLOR_PANEL, highlightthickness=0, **kw)
        self.records = []
        self.bind("<Configure>", lambda e: self.redraw())

    def set_records(self, records):
        self.records = records
        self.redraw()

    def redraw(self):
        self.delete("all")
        w = max(self.winfo_width(), 200)
        h = max(self.winfo_height(), 120)
        left, right, top, bottom = 52, 14, 30, 28
        pw, ph = w - left - right, h - top - bottom

        self.create_text(left, 12, anchor="w", text="Evolução do fitness",
                         font=("Helvetica", 10, "bold"), fill=COLOR_TEXT)
        legend = [("Melhor da geração", COLOR_BEST), ("Média da população", COLOR_AVG),
                  ("Melhor global", COLOR_GLOBAL)]
        lx = w - right
        for label, color in reversed(legend):
            tid = self.create_text(lx, 12, anchor="e", text=label,
                                   font=("Helvetica", 9), fill=COLOR_MUTED)
            x0 = self.bbox(tid)[0]
            self.create_line(x0 - 20, 12, x0 - 4, 12, fill=color, width=3)
            lx = x0 - 30

        if not self.records:
            self.create_rectangle(left, top, left + pw, top + ph, outline="#e2e8f0")
            self.create_text(left + pw / 2, top + ph / 2,
                             text="Inicie o treinamento para ver a evolução",
                             fill=COLOR_MUTED)
            return

        values = []
        for rec in self.records:
            values += [rec["best_fitness"], rec["avg_fitness"], rec["global_best_fitness"]]
        vmin, vmax = min(values), max(values)
        if vmax - vmin < 1e-6:
            vmin, vmax = vmin - 1, vmax + 1
        margin = (vmax - vmin) * 0.08
        vmin, vmax = vmin - margin, vmax + margin
        n = len(self.records)
        xmax = max(n, 10)

        def sx(gen):
            return left + (gen - 0.5) / xmax * pw

        def sy(v):
            return top + (vmax - v) / (vmax - vmin) * ph

        # Faixas de fundo indicando o modo do Minimax em cada geração
        band_w = pw / xmax
        for rec in self.records:
            x0 = left + (rec["generation"] - 1) * band_w
            self.create_rectangle(x0, top, x0 + band_w, top + ph,
                                  fill=MODE_BAND.get(rec["mode"], "#f8fafc"), width=0)

        self.create_rectangle(left, top, left + pw, top + ph, outline="#e2e8f0")

        # Eixo Y
        for k in range(5):
            v = vmin + (vmax - vmin) * k / 4
            y = sy(v)
            self.create_line(left, y, left + pw, y, fill="#eef2f7")
            self.create_text(left - 6, y, anchor="e", text=f"{v:.0f}",
                             font=("Helvetica", 8), fill=COLOR_MUTED)
        if vmin < 0 < vmax:
            self.create_line(left, sy(0), left + pw, sy(0), fill="#94a3b8", dash=(3, 3))

        # Eixo X
        step = max(1, xmax // 10)
        for gen in range(step, xmax + 1, step):
            self.create_text(sx(gen), top + ph + 12, text=str(gen),
                             font=("Helvetica", 8), fill=COLOR_MUTED)
        self.create_text(left + pw / 2, h - 4, anchor="s", text="geração",
                         font=("Helvetica", 8), fill=COLOR_MUTED)

        for key, color, width in (("avg_fitness", COLOR_AVG, 2),
                                  ("best_fitness", COLOR_BEST, 2),
                                  ("global_best_fitness", COLOR_GLOBAL, 2)):
            pts = []
            for rec in self.records:
                pts += [sx(rec["generation"]), sy(rec[key])]
            if len(pts) >= 4:
                self.create_line(*pts, fill=color, width=width)
            else:
                x, y = pts
                self.create_oval(x - 3, y - 3, x + 3, y + 3, fill=color, outline="")

        # Legenda das faixas de modo
        bx = left
        for mode in ("medio", "dificil", "misto"):
            if any(r["mode"] == mode for r in self.records):
                self.create_rectangle(bx, h - 14, bx + 10, h - 4,
                                      fill=MODE_BAND[mode], outline="#cbd5e1")
                tid = self.create_text(bx + 14, h - 9, anchor="w",
                                       text=f"Minimax {MODE_LABEL[mode]}",
                                       font=("Helvetica", 8), fill=COLOR_MUTED)
                bx = self.bbox(tid)[2] + 12


class PopulationChart(tk.Canvas):
    """Barras com o fitness de cada indivíduo de uma geração (ordenado)."""

    def __init__(self, master, **kw):
        super().__init__(master, bg=COLOR_PANEL, highlightthickness=0, **kw)
        self.record = None
        self.elitism = 0
        self.bind("<Configure>", lambda e: self.redraw())

    def set_record(self, record, elitism):
        self.record = record
        self.elitism = elitism
        self.redraw()

    def redraw(self):
        self.delete("all")
        w = max(self.winfo_width(), 200)
        h = max(self.winfo_height(), 100)
        left, right, top, bottom = 52, 14, 30, 18
        pw, ph = w - left - right, h - top - bottom

        if not self.record:
            self.create_text(left, 12, anchor="w", text="População",
                             font=("Helvetica", 10, "bold"), fill=COLOR_TEXT)
            self.create_rectangle(left, top, left + pw, top + ph, outline="#e2e8f0")
            return

        fits = self.record["population_fitness"]
        gen = self.record["generation"]
        self.create_text(
            left, 12, anchor="w",
            text=f"População da geração {gen} — {len(fits)} indivíduos "
                 f"(Minimax {MODE_LABEL.get(self.record['mode'], '')})",
            font=("Helvetica", 10, "bold"), fill=COLOR_TEXT)
        tid = self.create_text(w - right, 12, anchor="e", text="demais",
                               font=("Helvetica", 9), fill=COLOR_MUTED)
        x0 = self.bbox(tid)[0]
        self.create_rectangle(x0 - 14, 7, x0 - 4, 17, fill=COLOR_INDIV, outline="")
        tid = self.create_text(x0 - 22, 12, anchor="e", text="elite",
                               font=("Helvetica", 9), fill=COLOR_MUTED)
        x0 = self.bbox(tid)[0]
        self.create_rectangle(x0 - 14, 7, x0 - 4, 17, fill=COLOR_ELITE, outline="")

        vmin = min(min(fits), 0.0)
        vmax = max(max(fits), 0.0)
        if vmax - vmin < 1e-6:
            vmax = vmin + 1
        span = vmax - vmin

        def sy(v):
            return top + (vmax - v) / span * ph

        self.create_rectangle(left, top, left + pw, top + ph, outline="#e2e8f0")
        for k in range(3):
            v = vmin + span * k / 2
            self.create_text(left - 6, sy(v), anchor="e", text=f"{v:.0f}",
                             font=("Helvetica", 8), fill=COLOR_MUTED)
        y0 = sy(0)
        self.create_line(left, y0, left + pw, y0, fill="#94a3b8")

        n = len(fits)
        bw = pw / n
        gap = 1 if bw > 3 else 0
        for i, f in enumerate(fits):
            x = left + i * bw
            color = COLOR_ELITE if i < self.elitism else COLOR_INDIV
            self.create_rectangle(x + gap, min(y0, sy(f)), x + bw - gap,
                                  max(y0, sy(f)), fill=color, width=0)


# ---------------------------------------------------------------------------
# Abas Humano vs IA (base comum)
# ---------------------------------------------------------------------------
class HumanVsAITab(ttk.Frame):
    """Lógica de partida Humano vs Computador. Subclasses definem `ai_move`."""

    ai_name = "Computador"
    ai_delay_ms = 350

    def __init__(self, master, app):
        super().__init__(master, padding=12)
        self.app = app
        self.game = TicTacToe()
        self.human = PLAYER_X
        self.turn = PLAYER_X
        self.playing = False
        self.last = None
        self.invalid = None
        self.games_played = 0
        self.score = {"human": 0, "ai": 0, "draw": 0}
        self._pending = None

        self.columnconfigure(1, weight=1)
        self.controls = ttk.Frame(self)
        self.controls.grid(row=0, column=0, sticky="nw", padx=(0, 16))

        center = ttk.Frame(self)
        center.grid(row=0, column=1, sticky="n")
        self.status = ttk.Label(center, text="", style="Status.TLabel")
        self.status.pack(pady=(0, 8))
        self.board = BoardCanvas(center, size=330, on_click=self.on_cell)
        self.board.pack()
        self.score_label = ttk.Label(center, text="", style="Muted.TLabel")
        self.score_label.pack(pady=(8, 0))

        self.side = ttk.Frame(self)
        self.side.grid(row=0, column=2, sticky="ne", padx=(16, 0))

        self.starter_var = tk.StringVar(value="Alternar")
        self._update_score()
        self.status.config(text="Clique em “Nova partida” para começar.")

    # -- controles padrão -------------------------------------------------
    def build_starter_control(self, parent):
        ttk.Label(parent, text="Quem começa (joga com X)").pack(anchor="w", pady=(10, 2))
        for opt in STARTER_OPTIONS:
            ttk.Radiobutton(parent, text=opt, value=opt,
                            variable=self.starter_var).pack(anchor="w")

    # -- fluxo da partida -------------------------------------------------
    def can_start(self):
        return True

    def new_game(self):
        if not self.can_start():
            return
        if self._pending:
            self.after_cancel(self._pending)
            self._pending = None
        choice = self.starter_var.get()
        if choice == "Você":
            human_starts = True
        elif choice == "Computador":
            human_starts = False
        else:
            human_starts = self.games_played % 2 == 0
        self.human = PLAYER_X if human_starts else PLAYER_O
        self.game.reset()
        self.turn = PLAYER_X
        self.playing = True
        self.last = None
        self.invalid = None
        self.on_new_game()
        self._redraw()
        if self.turn == self.human:
            self.status.config(text=f"Sua vez — você é {SYMBOL[self.human]}")
        else:
            self._schedule_ai()

    def on_new_game(self):
        pass

    def on_cell(self, idx):
        if not self.playing or self.turn != self.human:
            return
        if not self.game.make_move(idx, self.human):
            self.status.config(text="Casa ocupada — escolha outra.")
            return
        self.last = idx
        self.invalid = None
        self._redraw()
        if not self._check_end():
            self.turn = -self.turn
            self._schedule_ai()

    def _schedule_ai(self):
        self.status.config(text=f"{self.ai_name} pensando… ({SYMBOL[self.turn]})")
        self._pending = self.after(self.ai_delay_ms, self._do_ai)

    def _do_ai(self):
        self._pending = None
        if not self.playing:
            return
        move = self.ai_move(self.game, self.turn)
        self.game.make_move(move, self.turn)
        self.last = move
        self._redraw()
        if not self._check_end():
            self.turn = -self.turn
            self.status.config(text=f"Sua vez — você é {SYMBOL[self.human]}")

    def ai_move(self, game, player):
        raise NotImplementedError

    def _check_end(self):
        winner = self.game.check_winner()
        if winner is None:
            return False
        self.playing = False
        self.games_played += 1
        if winner == 0:
            self.score["draw"] += 1
            msg = "Empate!"
        elif winner == self.human:
            self.score["human"] += 1
            msg = "Você venceu!"
        else:
            self.score["ai"] += 1
            msg = f"{self.ai_name} venceu!"
        self.status.config(text=msg + "  Clique em “Nova partida”.")
        self._update_score()
        self.on_game_end(winner)
        return True

    def on_game_end(self, winner):
        pass

    def reset_score(self):
        self.score = {"human": 0, "ai": 0, "draw": 0}
        self.games_played = 0
        self._update_score()

    def _update_score(self):
        s = self.score
        self.score_label.config(
            text=f"Você {s['human']}  ·  Empates {s['draw']}  ·  {self.ai_name} {s['ai']}")

    def _redraw(self):
        self.board.draw(self.game.board, last=self.last, invalid=self.invalid)


class MinimaxTab(HumanVsAITab):
    """Modo 1 — Usuário vs Minimax."""

    ai_name = "Minimax"

    def __init__(self, master, app):
        super().__init__(master, app)
        self.minimax = MinimaxPlayer()
        self.mode_var = tk.StringVar(value="dificil")

        c = self.controls
        ttk.Label(c, text="Você vs Minimax", style="Title.TLabel").pack(anchor="w")
        ttk.Label(c, text="Dificuldade").pack(anchor="w", pady=(12, 2))
        ttk.Radiobutton(c, text="Médio (50% Minimax, 50% aleatório)", value="medio",
                        variable=self.mode_var).pack(anchor="w")
        ttk.Radiobutton(c, text="Difícil (100% Minimax)", value="dificil",
                        variable=self.mode_var).pack(anchor="w")
        self.build_starter_control(c)
        ttk.Button(c, text="Nova partida", style="Accent.TButton",
                   command=self.new_game).pack(fill="x", pady=(16, 4))
        ttk.Button(c, text="Zerar placar", command=self.reset_score).pack(fill="x")

    def ai_move(self, game, player):
        return self.minimax.choose_move(game, player, mode=self.mode_var.get())


class NetworkTab(HumanVsAITab):
    """Modo 3 — Usuário vs Rede treinada, com cálculo de acurácia."""

    ai_name = "Rede"

    def __init__(self, master, app):
        super().__init__(master, app)
        self.mlp = None
        self.net_moves = 0
        self.net_valid = 0

        c = self.controls
        ttk.Label(c, text="Você vs Rede Neural", style="Title.TLabel").pack(anchor="w")
        self.source_label = ttk.Label(c, text="", style="Muted.TLabel", wraplength=230)
        self.source_label.pack(anchor="w", pady=(6, 0))
        self.build_starter_control(c)
        self.new_btn = ttk.Button(c, text="Nova partida", style="Accent.TButton",
                                  command=self.new_game)
        self.new_btn.pack(fill="x", pady=(16, 4))
        ttk.Button(c, text="Zerar placar e acurácia",
                   command=self.reset_score).pack(fill="x")
        ttk.Separator(c).pack(fill="x", pady=12)
        ttk.Button(c, text="Salvar rede (.json)…", command=self.save).pack(fill="x", pady=2)
        ttk.Button(c, text="Carregar rede (.json)…", command=self.load).pack(fill="x", pady=2)

        # Painel lateral: saídas da rede + acurácia
        s = self.side
        ttk.Label(s, text="Saídas da rede na última jogada",
                  style="Section.TLabel").pack(anchor="w")
        self.outputs = OutputsCanvas(s, size=165)
        self.outputs.pack(anchor="w", pady=(4, 2))
        ttk.Label(s, text="Borda grossa = casa escolhida (argmax).\n"
                          "Verde = escore alto, vermelho = baixo.",
                  style="Muted.TLabel").pack(anchor="w")

        ttk.Label(s, text="Acurácia contra você", style="Section.TLabel").pack(
            anchor="w", pady=(14, 2))
        self.acc_label = ttk.Label(s, text="", justify="left")
        self.acc_label.pack(anchor="w")

        ttk.Label(s, text="Avaliação automática vs Minimax",
                  style="Section.TLabel").pack(anchor="w", pady=(14, 2))
        row = ttk.Frame(s)
        row.pack(anchor="w")
        ttk.Label(row, text="Partidas por modo:").pack(side="left")
        self.eval_games = tk.IntVar(value=20)
        ttk.Spinbox(row, from_=2, to=500, width=6,
                    textvariable=self.eval_games).pack(side="left", padx=4)
        self.eval_btn = ttk.Button(row, text="Avaliar", command=self.run_evaluation)
        self.eval_btn.pack(side="left")
        cols = ("modo", "v", "e", "d", "inv", "naoderrota", "validas")
        self.eval_tree = ttk.Treeview(s, columns=cols, show="headings", height=3)
        for col, txt, wd in zip(cols, ("Modo", "V", "E", "D", "Inv.", "Não-derrota",
                                       "Jog. válidas"),
                                (62, 32, 32, 32, 38, 80, 82)):
            self.eval_tree.heading(col, text=txt)
            self.eval_tree.column(col, width=wd, anchor="center")
        self.eval_tree.pack(anchor="w", pady=(6, 0))

        self.refresh_source()
        self._update_accuracy()

    # -- rede ---------------------------------------------------------------
    def refresh_source(self):
        info = self.app.best_info
        if self.app.best_weights is None:
            self.mlp = None
            self.source_label.config(
                text="Nenhuma rede disponível. Treine na aba “Treinamento” "
                     "ou carregue um arquivo salvo.")
            self.new_btn.state(["disabled"])
            self.eval_btn.state(["disabled"])
        else:
            self.mlp = MLP(hidden_size=self.app.hidden_size,
                           weights=self.app.best_weights)
            self.source_label.config(
                text=f"Rede: 9-{self.app.hidden_size}-9 "
                     f"({self.mlp.num_weights()} pesos)\n{info}")
            self.new_btn.state(["!disabled"])
            self.eval_btn.state(["!disabled"])

    def can_start(self):
        if self.mlp is None:
            messagebox.showinfo("Sem rede", "Treine a rede ou carregue um arquivo primeiro.")
            return False
        return True

    def on_new_game(self):
        self.outputs.draw(None)

    def ai_move(self, game, player):
        """Escolhe a jogada pela saída da rede.

        A escolha "crua" da rede é o argmax de todas as 9 saídas. Se ela cair
        em uma casa ocupada, conta como jogada inválida (erro de acurácia) e,
        para a partida poder continuar, usa-se a melhor casa livre.
        """
        scores = self.mlp.forward(game.encode_for_player(player))
        raw = max(range(9), key=lambda i: scores[i])
        self.net_moves += 1
        if game.board[raw] == 0:
            self.net_valid += 1
            move = raw
            self.invalid = None
        else:
            move = max(game.available_moves(), key=lambda i: scores[i])
            self.invalid = raw
        self.outputs.draw(scores, chosen=raw)
        self._update_accuracy()
        return move

    def _do_ai(self):
        # Mantém o destaque de jogada inválida após o redraw da base
        super()._do_ai()
        if self.invalid is not None and self.playing:
            self.status.config(
                text=f"A rede tentou jogar na casa ocupada {self.invalid + 1} "
                     f"(jogada inválida). Sua vez — você é {SYMBOL[self.human]}")

    def on_game_end(self, winner):
        self._update_accuracy()

    def reset_score(self):
        self.net_moves = 0
        self.net_valid = 0
        super().reset_score()
        self._update_accuracy()

    def _update_accuracy(self):
        s = self.score
        games = s["human"] + s["ai"] + s["draw"]
        valid = (self.net_valid / self.net_moves * 100) if self.net_moves else 0.0
        nonloss = ((s["ai"] + s["draw"]) / games * 100) if games else 0.0
        win = (s["ai"] / games * 100) if games else 0.0
        self.acc_label.config(text=(
            f"Jogadas da rede: {self.net_moves}  (válidas: {self.net_valid})\n"
            f"Acurácia de jogadas válidas: {valid:.1f}%\n"
            f"Partidas: {games}  →  V {s['ai']} · E {s['draw']} · D {s['human']}\n"
            f"Taxa de não-derrota da rede: {nonloss:.1f}%\n"
            f"Taxa de vitória da rede: {win:.1f}%"))

    def run_evaluation(self):
        if self.app.best_weights is None:
            return
        try:
            n = int(self.eval_games.get())
        except (tk.TclError, ValueError):
            messagebox.showerror("Valor inválido", "Informe um número de partidas.")
            return
        n = max(2, n)
        res = evaluate_accuracy(self.app.best_weights, games_per_mode=n,
                                hidden_size=self.app.hidden_size,
                                seed=random.randrange(1 << 30))
        self.eval_tree.delete(*self.eval_tree.get_children())
        for key, label in (("medio", "Médio"), ("dificil", "Difícil"), ("geral", "Geral")):
            r = res[key]
            self.eval_tree.insert("", "end", values=(
                label, r["wins"], r["draws"], r["losses"], r["invalid_games"],
                f"{r['non_loss_rate']:.1f}%", f"{r['valid_move_rate']:.1f}%"))

    # -- arquivo ------------------------------------------------------------
    def save(self):
        if self.app.best_weights is None:
            messagebox.showinfo("Sem rede", "Não há rede treinada para salvar.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json", filetypes=[("JSON", "*.json")],
            initialfile="rede_treinada.json")
        if not path:
            return
        data = {"hidden_size": self.app.hidden_size,
                "weights": self.app.best_weights,
                "info": self.app.best_info}
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
        messagebox.showinfo("Salvo", f"Rede salva em:\n{path}")

    def load(self):
        path = filedialog.askopenfilename(filetypes=[("JSON", "*.json")])
        if not path:
            return
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            hidden = int(data["hidden_size"])
            weights = [float(w) for w in data["weights"]]
            MLP(hidden_size=hidden, weights=weights)  # valida tamanho
        except (OSError, ValueError, KeyError, TypeError) as exc:
            messagebox.showerror("Erro ao carregar", f"Arquivo inválido:\n{exc}")
            return
        self.app.set_best(weights, hidden,
                          f"Carregada de arquivo — {data.get('info', '')}".strip(" —"))


# ---------------------------------------------------------------------------
# Aba de treinamento
# ---------------------------------------------------------------------------
class TrainingTab(ttk.Frame):
    """Modo 2 — a rede aprende a jogar contra o Minimax (evolução pelo AG)."""

    def __init__(self, master, app):
        super().__init__(master, padding=12)
        self.app = app
        self.ga = None
        self.running = False
        self._tick_id = None
        self.records = []

        # Partida de demonstração (melhor rede vs Minimax)
        self.demo_game = None
        self.demo_id = None
        self.demo_count = 0
        self.demo_minimax = MinimaxPlayer()

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_params()
        self._build_center()
        self._build_demo()
        self._refresh_buttons()

    # -- layout ---------------------------------------------------------------
    def _build_params(self):
        p = ttk.Frame(self)
        p.grid(row=0, column=0, sticky="nsw", padx=(0, 12))
        ttk.Label(p, text="Parâmetros do AG", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w", pady=(0, 8))

        self.vars = {}
        fields = [
            ("population_size", "População (N)", 60, 4, 1000, 1),
            ("max_generations", "Máx. gerações", 100, 1, 10000, 1),
            ("stagnation_limit", "Parada por estagnação", 25, 1, 10000, 1),
            ("elitism_count", "Elitismo", 4, 0, 100, 1),
            ("tournament_size", "Tamanho do torneio", 3, 2, 50, 1),
            ("mutation_rate", "Taxa de mutação", 0.08, 0.0, 1.0, 0.01),
            ("mutation_std", "Desvio da mutação", 0.25, 0.01, 5.0, 0.05),
            ("blx_alpha", "Alpha do BLX-α", 0.5, 0.0, 2.0, 0.05),
            ("hidden_size", "Neurônios ocultos", 18, 1, 200, 1),
        ]
        self.param_widgets = []
        for r, (key, label, default, lo, hi, inc) in enumerate(fields, start=1):
            ttk.Label(p, text=label).grid(row=r, column=0, sticky="w", pady=2)
            var = tk.StringVar(value=str(default))
            sb = ttk.Spinbox(p, from_=lo, to=hi, increment=inc, width=8, textvariable=var)
            sb.grid(row=r, column=1, sticky="e", pady=2)
            self.vars[key] = (var, type(default), lo, hi)
            self.param_widgets.append(sb)

        r = len(fields) + 1
        ttk.Label(p, text="Semente (vazio = aleatória)").grid(row=r, column=0, sticky="w", pady=2)
        self.seed_var = tk.StringVar(value="")
        e = ttk.Entry(p, textvariable=self.seed_var, width=9)
        e.grid(row=r, column=1, sticky="e", pady=2)
        self.param_widgets.append(e)

        r += 1
        ttk.Label(p, text="Ordem dos modos do Minimax").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(10, 2))
        self.order_var = tk.StringVar(value=next(iter(TRAIN_ORDERS)))
        cb = ttk.Combobox(p, values=list(TRAIN_ORDERS), textvariable=self.order_var,
                          state="readonly", width=28)
        cb.grid(row=r + 1, column=0, columnspan=2, sticky="we")
        self.param_widgets.append(cb)
        ttk.Label(p, text="A rede alterna entre X e O a cada partida,\n"
                          "então o Minimax nem sempre começa.",
                  style="Muted.TLabel").grid(row=r + 2, column=0, columnspan=2,
                                             sticky="w", pady=(4, 0))

        r += 3
        ttk.Label(p, text="Pausa entre gerações (ms)").grid(
            row=r, column=0, columnspan=2, sticky="w", pady=(10, 0))
        self.delay_var = tk.IntVar(value=150)
        ttk.Scale(p, from_=0, to=1500, variable=self.delay_var,
                  orient="horizontal").grid(row=r + 1, column=0, columnspan=2, sticky="we")

        r += 2
        btns = ttk.Frame(p)
        btns.grid(row=r, column=0, columnspan=2, sticky="we", pady=(12, 0))
        btns.columnconfigure((0, 1), weight=1)
        self.start_btn = ttk.Button(btns, text="▶ Iniciar", style="Accent.TButton",
                                    command=self.start)
        self.start_btn.grid(row=0, column=0, sticky="we", padx=(0, 3), pady=2)
        self.pause_btn = ttk.Button(btns, text="⏸ Pausar", command=self.pause)
        self.pause_btn.grid(row=0, column=1, sticky="we", padx=(3, 0), pady=2)
        self.step_btn = ttk.Button(btns, text="Passo (1 geração)", command=self.step)
        self.step_btn.grid(row=1, column=0, sticky="we", padx=(0, 3), pady=2)
        self.reset_btn = ttk.Button(btns, text="Resetar", command=self.reset)
        self.reset_btn.grid(row=1, column=1, sticky="we", padx=(3, 0), pady=2)

        r += 1
        self.state_label = ttk.Label(p, text="", style="Status.TLabel",
                                     wraplength=250, justify="left")
        self.state_label.grid(row=r, column=0, columnspan=2, sticky="w", pady=(12, 0))
        self.summary_label = ttk.Label(p, text="", justify="left")
        self.summary_label.grid(row=r + 1, column=0, columnspan=2, sticky="w", pady=(6, 0))

    def _build_center(self):
        c = ttk.Frame(self)
        c.grid(row=0, column=1, sticky="nsew")
        c.columnconfigure(0, weight=1)
        c.rowconfigure(0, weight=3)
        c.rowconfigure(1, weight=2)
        c.rowconfigure(2, weight=3)

        self.chart = LineChart(c, height=230)
        self.chart.grid(row=0, column=0, sticky="nsew")
        self.pop_chart = PopulationChart(c, height=150)
        self.pop_chart.grid(row=1, column=0, sticky="nsew", pady=(8, 0))

        log = ttk.Frame(c)
        log.grid(row=2, column=0, sticky="nsew", pady=(8, 0))
        log.columnconfigure(0, weight=1)
        log.rowconfigure(1, weight=1)
        ttk.Label(log, text="Histórico por geração (clique numa linha para ver a "
                            "população daquela geração)",
                  style="Muted.TLabel").grid(row=0, column=0, sticky="w")
        cols = ("ger", "modo", "melhor", "media", "global", "v", "e", "d", "inv", "estag")
        heads = ("Ger.", "Modo", "Melhor", "Média", "Melhor global", "Vitórias",
                 "Empates", "Derrotas", "Inválidas", "Estagnação")
        widths = (40, 55, 55, 55, 85, 55, 55, 60, 62, 72)
        self.tree = ttk.Treeview(log, columns=cols, show="headings", height=7)
        for col, h, w in zip(cols, heads, widths):
            self.tree.heading(col, text=h)
            self.tree.column(col, width=w, anchor="center", stretch=True)
        sb = ttk.Scrollbar(log, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=sb.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        sb.grid(row=1, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._on_select_row)

    def _build_demo(self):
        d = ttk.Frame(self)
        d.grid(row=0, column=2, sticky="nse", padx=(12, 0))
        ttk.Label(d, text="Melhor rede vs Minimax", style="Title.TLabel").pack(anchor="w")
        self.demo_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(d, text="Mostrar partidas durante o treino",
                        variable=self.demo_var).pack(anchor="w", pady=(4, 2))
        ttk.Label(d, text="Velocidade da partida (ms/jogada)").pack(anchor="w")
        self.demo_speed = tk.IntVar(value=250)
        ttk.Scale(d, from_=30, to=1000, variable=self.demo_speed,
                  orient="horizontal").pack(fill="x")
        self.demo_title = ttk.Label(d, text="", style="Section.TLabel", wraplength=240)
        self.demo_title.pack(anchor="w", pady=(10, 4))
        self.demo_board = BoardCanvas(d, size=240)
        self.demo_board.pack()
        self.demo_status = ttk.Label(d, text="", wraplength=240, justify="left")
        self.demo_status.pack(anchor="w", pady=(6, 0))
        self.demo_btn = ttk.Button(d, text="Jogar uma partida agora",
                                   command=lambda: self._start_demo(force=True))
        self.demo_btn.pack(fill="x", pady=(10, 0))
        ttk.Label(d, text="A rede usa os pesos do melhor cromossomo\n"
                          "encontrado até agora (melhor global).",
                  style="Muted.TLabel").pack(anchor="w", pady=(6, 0))

    # -- parâmetros -----------------------------------------------------------
    def _read_params(self):
        params = {}
        for key, (var, typ, lo, hi) in self.vars.items():
            try:
                val = typ(float(var.get())) if typ is int else float(var.get())
            except ValueError:
                raise ValueError(f"Valor inválido em “{key}”: {var.get()!r}")
            if not (lo <= val <= hi):
                raise ValueError(f"“{key}” deve estar entre {lo} e {hi}.")
            params[key] = val
        if params["elitism_count"] >= params["population_size"]:
            raise ValueError("O elitismo deve ser menor que o tamanho da população.")
        seed = self.seed_var.get().strip()
        params["seed"] = int(seed) if seed else None
        return params

    def _ensure_ga(self):
        if self.ga is not None:
            return True
        try:
            params = self._read_params()
        except ValueError as exc:
            messagebox.showerror("Parâmetros inválidos", str(exc))
            return False
        self.ga = GeneticAlgorithm(**params)
        self.order = TRAIN_ORDERS[self.order_var.get()]
        for w in self.param_widgets:
            w.state(["disabled"])
        return True

    # -- controle do treino ---------------------------------------------------
    def _finished(self):
        return bool(self.records) and self.records[-1]["converged"]

    def start(self):
        if self.running or self._finished() or not self._ensure_ga():
            return
        self.running = True
        self._refresh_buttons()
        self._tick()

    def pause(self):
        self.running = False
        if self._tick_id:
            self.after_cancel(self._tick_id)
            self._tick_id = None
        self._refresh_buttons()

    def step(self):
        if self.running or self._finished() or not self._ensure_ga():
            return
        self._do_generation()
        self._refresh_buttons()

    def reset(self):
        self.pause()
        self._stop_demo()
        self.ga = None
        self.records = []
        self.tree.delete(*self.tree.get_children())
        self.chart.set_records([])
        self.pop_chart.set_record(None, 0)
        self.demo_board.draw([0] * 9)
        self.demo_title.config(text="")
        self.demo_status.config(text="")
        for w in self.param_widgets:
            w.state(["!disabled"])
        self.param_widgets[-1].state(["readonly"])
        self._refresh_buttons()

    def _tick(self):
        self._tick_id = None
        if not self.running:
            return
        self._do_generation()
        if self._finished():
            self.running = False
            self._refresh_buttons()
            return
        self._tick_id = self.after(int(self.delay_var.get()), self._tick)

    def _do_generation(self):
        m = self.ga.step_generation(self.order)
        self.records.append(m)
        ga = self.ga

        self.tree.insert("", "end", iid=str(m["generation"]), values=(
            m["generation"], MODE_LABEL.get(m["mode"], m["mode"]),
            f"{m['best_fitness']:.1f}", f"{m['avg_fitness']:.1f}",
            f"{m['global_best_fitness']:.1f}", m["wins"], m["draws"],
            m["losses"], m["invalid_moves"],
            f"{m['stagnation']}/{ga.stagnation_limit}"))
        self.tree.see(str(m["generation"]))
        self.chart.set_records(self.records)
        self.pop_chart.set_record(m, ga.elitism_count)

        self.app.set_best(
            m["best_weights"], ga.hidden_size,
            f"Treinada: geração {m['generation']}, "
            f"fitness {m['global_best_fitness']:.1f}")

        if self.demo_var.get() and self.demo_game is None:
            self._start_demo()
        self._refresh_buttons()

    def _refresh_buttons(self):
        finished = self._finished()
        self.start_btn.config(text="▶ Continuar" if self.records else "▶ Iniciar")
        self.start_btn.state(["disabled"] if self.running or finished else ["!disabled"])
        self.step_btn.state(["disabled"] if self.running or finished else ["!disabled"])
        self.pause_btn.state(["!disabled"] if self.running else ["disabled"])
        self.demo_btn.state(["!disabled"] if self.records else ["disabled"])

        if not self.records:
            self.state_label.config(text="Pronto. Ajuste os parâmetros e clique em "
                                         "Iniciar ou Passo.")
            self.summary_label.config(text="")
            return

        m = self.records[-1]
        ga = self.ga
        if finished:
            if m["generation"] >= ga.max_generations:
                why = f"atingiu o máximo de {ga.max_generations} gerações"
            else:
                why = (f"convergiu — {ga.stagnation_limit} gerações sem melhorar "
                       "o melhor global")
            state = f"✔ Treino concluído: {why}. Jogue contra a rede na 3ª aba."
        elif self.running:
            state = "Treinando…"
        else:
            state = "Pausado."
        self.state_label.config(text=state)
        n_games = sum(m[k] for k in ("wins", "draws", "losses"))
        self.summary_label.config(text=(
            f"Geração: {m['generation']} / {ga.max_generations}\n"
            f"Modo do Minimax: {MODE_LABEL.get(m['mode'], m['mode'])}\n"
            f"Melhor da geração: {m['best_fitness']:.1f}\n"
            f"Média da população: {m['avg_fitness']:.1f}\n"
            f"Melhor global: {m['global_best_fitness']:.1f}\n"
            f"Partidas na geração: {n_games}\n"
            f"Estagnação: {m['stagnation']} / {ga.stagnation_limit}"))

    def _on_select_row(self, _event):
        sel = self.tree.selection()
        if not sel or self.ga is None:
            return
        gen = int(sel[0])
        rec = self.records[gen - 1]
        self.pop_chart.set_record(rec, self.ga.elitism_count)

    # -- partida de demonstração ---------------------------------------------
    def _start_demo(self, force=False):
        if not self.records:
            return
        if self.demo_game is not None and not force:
            return
        self._stop_demo()
        m = self.records[-1]
        mode = m["mode"]
        if mode == "misto":
            mode = "medio" if self.demo_count % 4 < 2 else "dificil"
        self.demo_mlp = MLP(hidden_size=self.ga.hidden_size, weights=m["best_weights"])
        self.demo_net = PLAYER_X if self.demo_count % 2 == 0 else PLAYER_O
        self.demo_mode = mode
        self.demo_count += 1
        self.demo_game = TicTacToe()
        self.demo_turn = PLAYER_X
        self.demo_title.config(
            text=f"Ger. {m['generation']}: Rede ({SYMBOL[self.demo_net]}) vs "
                 f"Minimax {MODE_LABEL[mode]} ({SYMBOL[-self.demo_net]})")
        self.demo_board.draw(self.demo_game.board)
        first = "a rede" if self.demo_net == PLAYER_X else "o Minimax"
        self.demo_status.config(text=f"Começa {first}.")
        self.demo_id = self.after(int(self.demo_speed.get()), self._demo_step)

    def _stop_demo(self):
        if self.demo_id:
            self.after_cancel(self.demo_id)
        self.demo_id = None
        self.demo_game = None

    def _demo_step(self):
        self.demo_id = None
        g = self.demo_game
        if g is None:
            return
        invalid = None
        if self.demo_turn == self.demo_net:
            # Mesmo critério do treino: argmax sobre as 9 saídas
            move = self.demo_mlp.choose_move(g, self.demo_net, allow_invalid=True)
            if not g.make_move(move, self.demo_net):
                invalid = move
        else:
            move = self.demo_minimax.choose_move(g, self.demo_turn, mode=self.demo_mode)
            g.make_move(move, self.demo_turn)
        self.demo_board.draw(g.board, last=None if invalid is not None else move,
                             invalid=invalid)

        if invalid is not None:
            self.demo_status.config(
                text=f"Rede escolheu casa ocupada ({invalid + 1}): jogada inválida, "
                     "derrota (como no treino).")
            self._end_demo()
            return
        winner = g.check_winner()
        if winner is not None:
            if winner == 0:
                txt = "Empate."
            elif winner == self.demo_net:
                txt = "A rede venceu!"
            else:
                txt = "O Minimax venceu."
            self.demo_status.config(text=txt)
            self._end_demo()
            return
        self.demo_turn = -self.demo_turn
        who = "Rede" if self.demo_turn == self.demo_net else "Minimax"
        self.demo_status.config(text=f"Vez de: {who}")
        self.demo_id = self.after(int(self.demo_speed.get()), self._demo_step)

    def _end_demo(self):
        # Mostra o resultado por um instante e libera para a próxima partida
        def release():
            self.demo_id = None
            self.demo_game = None
            if self.running and self.demo_var.get():
                self._start_demo()
        self.demo_id = self.after(max(600, int(self.demo_speed.get()) * 3), release)


# ---------------------------------------------------------------------------
# Janela principal
# ---------------------------------------------------------------------------
class App(ttk.Frame):
    def __init__(self, root):
        super().__init__(root)
        self.root = root
        self.best_weights = None
        self.hidden_size = 18
        self.best_info = ""

        root.title("T2IA — Jogo da Velha: Rede Neural + AG + Minimax")
        root.configure(bg=COLOR_BG)
        root.minsize(1200, 700)
        self._setup_style()

        self.pack(fill="both", expand=True)
        self.nb = ttk.Notebook(self)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)

        self.minimax_tab = MinimaxTab(self.nb, self)
        self.training_tab = TrainingTab(self.nb, self)
        self.network_tab = NetworkTab(self.nb, self)
        self.nb.add(self.minimax_tab, text="  1. Você vs Minimax  ")
        self.nb.add(self.training_tab, text="  2. Treinamento (Rede vs Minimax)  ")
        self.nb.add(self.network_tab, text="  3. Você vs Rede treinada  ")

    def _setup_style(self):
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure(".", background=COLOR_BG, foreground=COLOR_TEXT,
                        font=("Helvetica", 10))
        style.configure("TFrame", background=COLOR_BG)
        style.configure("TLabel", background=COLOR_BG)
        style.configure("Title.TLabel", font=("Helvetica", 13, "bold"))
        style.configure("Section.TLabel", font=("Helvetica", 10, "bold"))
        style.configure("Status.TLabel", font=("Helvetica", 11, "bold"))
        style.configure("Muted.TLabel", foreground=COLOR_MUTED, font=("Helvetica", 9))
        style.configure("Accent.TButton", font=("Helvetica", 10, "bold"),
                        foreground="white", background="#2563eb")
        style.map("Accent.TButton",
                  background=[("disabled", "#94a3b8"), ("active", "#1d4ed8")])
        style.configure("TNotebook.Tab", padding=(10, 6), font=("Helvetica", 10, "bold"))
        style.configure("Treeview", rowheight=22, background=COLOR_PANEL,
                        fieldbackground=COLOR_PANEL)
        style.configure("Treeview.Heading", font=("Helvetica", 9, "bold"))

    def set_best(self, weights, hidden_size, info):
        """Atualiza a rede disponível para a aba 'Você vs Rede'."""
        self.best_weights = list(weights)
        self.hidden_size = hidden_size
        self.best_info = info
        self.network_tab.refresh_source()


def run():
    root = tk.Tk()
    root.geometry("1280x780")
    App(root)
    root.mainloop()
