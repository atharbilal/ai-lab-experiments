"""
LAB 05: Minimax Algorithm for Game Playing (Tic-Tac-Toe)
Includes plain Minimax and Minimax with Alpha-Beta pruning.

Run:  python lab05_minimax.py          -> AI vs AI demo + comparison
      python lab05_minimax.py --play   -> play against the AI
"""
import math
import sys
import time

WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
             (0, 3, 6), (1, 4, 7), (2, 5, 8),
             (0, 4, 8), (2, 4, 6)]

AI, HUMAN = "X", "O"     # AI = maximizer, HUMAN = minimizer
nodes = 0                # counts nodes visited, to compare algorithms


def winner(b):
    for a, c, d in WIN_LINES:
        if b[a] != " " and b[a] == b[c] == b[d]:
            return b[a]
    return None


def terminal(b):
    return winner(b) is not None or " " not in b


def utility(b, depth):
    """+10 for AI win, -10 for HUMAN win, 0 draw. Depth term prefers faster wins / slower losses."""
    w = winner(b)
    if w == AI:
        return 10 - depth
    if w == HUMAN:
        return depth - 10
    return 0


def moves(b):
    return [i for i, c in enumerate(b) if c == " "]


# ---------------- Plain Minimax ----------------
def minimax(b, depth, maximizing):
    global nodes
    nodes += 1
    if terminal(b):
        return utility(b, depth)

    if maximizing:
        best = -math.inf
        for m in moves(b):
            b[m] = AI
            best = max(best, minimax(b, depth + 1, False))
            b[m] = " "
        return best
    else:
        best = math.inf
        for m in moves(b):
            b[m] = HUMAN
            best = min(best, minimax(b, depth + 1, True))
            b[m] = " "
        return best


# ---------------- Minimax with Alpha-Beta pruning ----------------
def alphabeta(b, depth, alpha, beta, maximizing):
    global nodes
    nodes += 1
    if terminal(b):
        return utility(b, depth)

    if maximizing:
        value = -math.inf
        for m in moves(b):
            b[m] = AI
            value = max(value, alphabeta(b, depth + 1, alpha, beta, False))
            b[m] = " "
            alpha = max(alpha, value)
            if alpha >= beta:          # beta cut-off
                break
        return value
    else:
        value = math.inf
        for m in moves(b):
            b[m] = HUMAN
            value = min(value, alphabeta(b, depth + 1, alpha, beta, True))
            b[m] = " "
            beta = min(beta, value)
            if alpha >= beta:          # alpha cut-off
                break
        return value


def best_move(b, player, use_alphabeta=True):
    """Return the best cell for `player` (AI maximizes, HUMAN minimizes)."""
    maximizing = player == AI
    best_val = -math.inf if maximizing else math.inf
    choice = None
    for m in moves(b):
        b[m] = player
        if use_alphabeta:
            val = alphabeta(b, 1, -math.inf, math.inf, not maximizing)
        else:
            val = minimax(b, 1, not maximizing)
        b[m] = " "
        if (maximizing and val > best_val) or (not maximizing and val < best_val):
            best_val, choice = val, m
    return choice, best_val


def show(b):
    print()
    for r in range(3):
        cells = [b[r * 3 + c] if b[r * 3 + c] != " " else str(r * 3 + c) for c in range(3)]
        print("  " + " | ".join(cells))
        if r < 2:
            print("  " + "-" * 9)
    print()


def compare_algorithms():
    global nodes
    print("=" * 55)
    print("Minimax vs Alpha-Beta: choosing the first move on an empty board")
    print("=" * 55)
    for name, ab in [("Minimax", False), ("Alpha-Beta", True)]:
        nodes = 0
        t0 = time.perf_counter()
        move, val = best_move([" "] * 9, AI, use_alphabeta=ab)
        dt = time.perf_counter() - t0
        print(f"{name:<11} move={move}  value={val:>3}  nodes={nodes:>7}  time={dt:.3f}s")


def ai_vs_ai():
    print("\n" + "=" * 55)
    print("AI (X) vs AI (O) - both play optimally")
    print("=" * 55)
    b = [" "] * 9
    player = AI
    while not terminal(b):
        m, _ = best_move(b, player)
        b[m] = player
        print(f"{player} plays cell {m}")
        player = HUMAN if player == AI else AI
    show(b)
    w = winner(b)
    print("Result:", f"{w} wins" if w else "Draw (expected: perfect play always draws)")


def play_human():
    b = [" "] * 9
    first = input("Do you want to go first? (y/n): ").strip().lower() == "y"
    turn = HUMAN if first else AI
    print(f"You are {HUMAN}, AI is {AI}. Enter a cell number 0-8.")
    show(b)
    while not terminal(b):
        if turn == HUMAN:
            try:
                m = int(input("Your move: "))
                if m not in moves(b):
                    raise ValueError
            except ValueError:
                print("Invalid move, try again.")
                continue
            b[m] = HUMAN
        else:
            m, _ = best_move(b, AI)
            b[m] = AI
            print(f"AI plays {m}")
        show(b)
        turn = AI if turn == HUMAN else HUMAN
    w = winner(b)
    print("AI wins!" if w == AI else "You win!" if w == HUMAN else "It's a draw.")


if __name__ == "__main__":
    if "--play" in sys.argv:
        play_human()
    else:
        compare_algorithms()
        ai_vs_ai()
