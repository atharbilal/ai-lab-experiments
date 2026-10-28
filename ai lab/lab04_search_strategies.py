"""
LAB 04: Basic Search Strategies (BFS, DFS, IDDFS)
Applied to: 8-Puzzle, 8-Queens, Tic-Tac-Toe
"""
from collections import deque
import time

# =============================================================
# PART A: 8-PUZZLE  (BFS, depth-limited DFS, Iterative Deepening)
# =============================================================
GOAL = (1, 2, 3,
        4, 5, 6,
        7, 8, 0)          # 0 = blank

MOVES = {"Up": -3, "Down": 3, "Left": -1, "Right": 1}


def is_solvable(state):
    """A state is solvable iff the number of inversions is even (3x3 board)."""
    tiles = [t for t in state if t != 0]
    inversions = sum(1 for i in range(len(tiles))
                     for j in range(i + 1, len(tiles)) if tiles[i] > tiles[j])
    return inversions % 2 == 0


def neighbours(state):
    """Yield (move_name, next_state) for every legal blank move."""
    z = state.index(0)
    row, col = divmod(z, 3)
    for name, delta in MOVES.items():
        if name == "Up" and row == 0:      continue
        if name == "Down" and row == 2:    continue
        if name == "Left" and col == 0:    continue
        if name == "Right" and col == 2:   continue
        nz = z + delta
        s = list(state)
        s[z], s[nz] = s[nz], s[z]
        yield name, tuple(s)


def reconstruct(parent, state):
    path = []
    while parent[state] is not None:
        prev, move = parent[state]
        path.append(move)
        state = prev
    return path[::-1]


def bfs_puzzle(start):
    """Breadth-First Search: complete and optimal (unit step cost)."""
    frontier = deque([start])
    parent = {start: None}
    expanded = 0
    while frontier:
        state = frontier.popleft()
        expanded += 1
        if state == GOAL:
            return reconstruct(parent, state), expanded
        for move, nxt in neighbours(state):
            if nxt not in parent:
                parent[nxt] = (state, move)
                frontier.append(nxt)
    return None, expanded


def dfs_puzzle(start, limit=25):
    """Depth-limited DFS (plain DFS can wander down an extremely deep branch)."""
    stack = [(start, [], {start})]
    expanded = 0
    while stack:
        state, path, on_path = stack.pop()
        expanded += 1
        if state == GOAL:
            return path, expanded
        if len(path) >= limit:
            continue
        for move, nxt in neighbours(state):
            if nxt not in on_path:                 # avoid cycles on current path
                stack.append((nxt, path + [move], on_path | {nxt}))
    return None, expanded


def iddfs_puzzle(start, max_depth=30):
    """Iterative Deepening DFS: optimal like BFS, memory like DFS."""
    total = 0
    for depth in range(max_depth + 1):
        path, expanded = dfs_puzzle(start, depth)
        total += expanded
        if path is not None:
            return path, total
    return None, total


def print_board(state):
    for r in range(3):
        print("   " + " ".join(str(t) if t else "_" for t in state[r * 3:r * 3 + 3]))


def run_8_puzzle():
    print("=" * 60)
    print("PART A: 8-PUZZLE")
    print("=" * 60)
    start = (1, 2, 3,
             0, 4, 6,
             7, 5, 8)
    print("Start state:")
    print_board(start)
    print("Goal state:")
    print_board(GOAL)

    if not is_solvable(start):
        print("This configuration is NOT solvable.")
        return

    for name, fn in [("BFS", bfs_puzzle),
                     ("DFS (depth limit 25)", lambda s: dfs_puzzle(s, 25)),
                     ("IDDFS", iddfs_puzzle)]:
        t0 = time.perf_counter()
        path, expanded = fn(start)
        dt = (time.perf_counter() - t0) * 1000
        print(f"\n{name}:")
        print(f"   Moves ({len(path)}): {' -> '.join(path)}")
        print(f"   Nodes expanded: {expanded}   Time: {dt:.2f} ms")


# =============================================================
# PART B: 8-QUEENS  (DFS with backtracking, BFS over partial boards)
# =============================================================
N = 8


def safe(queens, col):
    """queens[r] = column of the queen in row r. Can we put a queen in next row at col?"""
    row = len(queens)
    for r, c in enumerate(queens):
        if c == col or abs(c - col) == abs(r - row):
            return False
    return True


def dfs_queens(find_all=True):
    """DFS / backtracking: place one queen per row."""
    solutions, expanded = [], 0
    stack = [()]
    while stack:
        queens = stack.pop()
        expanded += 1
        if len(queens) == N:
            solutions.append(queens)
            if not find_all:
                break
            continue
        for col in reversed(range(N)):          # reversed -> explores col 0 first
            if safe(queens, col):
                stack.append(queens + (col,))
    return solutions, expanded


def bfs_queens():
    """BFS level by level (row by row). Finds all solutions but stores every level."""
    frontier = deque([()])
    solutions, expanded = [], 0
    while frontier:
        queens = frontier.popleft()
        expanded += 1
        if len(queens) == N:
            solutions.append(queens)
            continue
        for col in range(N):
            if safe(queens, col):
                frontier.append(queens + (col,))
    return solutions, expanded


def print_queens(q):
    for r in range(N):
        print("   " + " ".join("Q" if q[r] == c else "." for c in range(N)))


def run_8_queens():
    print("\n" + "=" * 60)
    print("PART B: 8-QUEENS")
    print("=" * 60)
    first, exp_first = dfs_queens(find_all=False)
    print(f"DFS - first solution found after expanding {exp_first} nodes:")
    print_queens(first[0])

    sols_dfs, exp_dfs = dfs_queens(find_all=True)
    sols_bfs, exp_bfs = bfs_queens()
    print(f"\nDFS - all solutions: {len(sols_dfs)} (nodes expanded: {exp_dfs})")
    print(f"BFS - all solutions: {len(sols_bfs)} (nodes expanded: {exp_bfs})")


# =============================================================
# PART C: TIC-TAC-TOE  (BFS / DFS over the game tree)
# =============================================================
WIN_LINES = [(0, 1, 2), (3, 4, 5), (6, 7, 8),
             (0, 3, 6), (1, 4, 7), (2, 5, 8),
             (0, 4, 8), (2, 4, 6)]


def winner(b):
    for a, c, d in WIN_LINES:
        if b[a] != " " and b[a] == b[c] == b[d]:
            return b[a]
    return None


def to_move(b):
    return "X" if b.count("X") == b.count("O") else "O"


def successors(b):
    p = to_move(b)
    for i, cell in enumerate(b):
        if cell == " ":
            nb = list(b)
            nb[i] = p
            yield i, tuple(nb)


def bfs_tictactoe_states():
    """BFS over all reachable board positions; count distinct states and terminal outcomes."""
    start = tuple(" " * 9)
    seen = {start}
    frontier = deque([start])
    outcomes = {"X": 0, "O": 0, "Draw": 0}
    while frontier:
        b = frontier.popleft()
        w = winner(b)
        if w:
            outcomes[w] += 1
            continue
        if " " not in b:
            outcomes["Draw"] += 1
            continue
        for _, nb in successors(b):
            if nb not in seen:
                seen.add(nb)
                frontier.append(nb)
    return len(seen), outcomes


def dfs_count_games(b=None):
    """DFS over the full game tree: count every possible complete game."""
    if b is None:
        b = tuple(" " * 9)
    w = winner(b)
    if w or " " not in b:
        return 1
    return sum(dfs_count_games(nb) for _, nb in successors(b))


def dfs_find_winning_line(b, depth=0, max_depth=3):
    """DFS: search for a forced win for the player to move within max_depth of their own moves."""
    me = to_move(b)
    for i, nb in successors(b):
        if winner(nb) == me:
            return [i]
    if depth >= max_depth - 1:
        return None
    for i, nb in successors(b):
        if " " not in nb:
            continue
        replies = list(successors(nb))
        plans = []
        for j, nb2 in replies:
            if winner(nb2):            # opponent wins -> this move fails
                plans = None
                break
            rest = dfs_find_winning_line(nb2, depth + 1, max_depth)
            if rest is None:
                plans = None
                break
            plans.append(rest)
        if plans:                       # every reply is answered with a win
            return [i]
    return None


def print_ttt(b):
    for r in range(3):
        print("   " + " | ".join(b[r * 3:r * 3 + 3]))
        if r < 2:
            print("   " + "-" * 9)


def run_tictactoe():
    print("\n" + "=" * 60)
    print("PART C: TIC-TAC-TOE")
    print("=" * 60)
    states, outcomes = bfs_tictactoe_states()
    print(f"BFS: {states} distinct reachable board positions")
    print(f"     Terminal positions -> X wins: {outcomes['X']}, "
          f"O wins: {outcomes['O']}, Draws: {outcomes['Draw']}")

    print(f"DFS: {dfs_count_games()} possible complete games in the full tree")

    board = ("X", "O", " ",
             " ", "X", " ",
             " ", " ", "O")
    print("\nPosition (X to move):")
    print_ttt(board)
    move = dfs_find_winning_line(board)
    if move:
        print(f"DFS found a forcing move for {to_move(board)}: cell {move[0]} "
              f"(row {move[0] // 3}, col {move[0] % 3})")
    else:
        print("No forced win found within the depth limit.")


if __name__ == "__main__":
    run_8_puzzle()
    run_8_queens()
    run_tictactoe()
