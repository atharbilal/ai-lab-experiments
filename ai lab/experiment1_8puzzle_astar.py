"""
Experiment 1: Intelligent 8-Puzzle Solver Using A* Search
Implements Manhattan-distance and misplaced-tile heuristics with performance comparison.
"""

import heapq
import time
from copy import deepcopy

class PuzzleState:
    def __init__(self, board, parent=None, move=None, g=0):
        self.board = board
        self.parent = parent
        self.move = move
        self.g = g  # Cost from start
        self.h = 0  # Heuristic cost
        self.f = 0  # Total cost (g + h)

    def __lt__(self, other):
        return self.f < other.f

    def __eq__(self, other):
        return self.board == other.board

    def __hash__(self):
        return hash(tuple(self.board))

    def get_blank_position(self):
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == 0:
                    return i, j
        return None

    def get_children(self):
        children = []
        blank_i, blank_j = self.get_blank_position()
        moves = [(-1, 0, 'UP'), (1, 0, 'DOWN'), (0, -1, 'LEFT'), (0, 1, 'RIGHT')]

        for di, dj, move_name in moves:
            new_i, new_j = blank_i + di, blank_j + dj
            if 0 <= new_i < 3 and 0 <= new_j < 3:
                new_board = deepcopy(self.board)
                new_board[blank_i][blank_j], new_board[new_i][new_j] = \
                    new_board[new_i][new_j], new_board[blank_i][blank_j]
                children.append(PuzzleState(new_board, self, move_name, self.g + 1))

        return children

def manhattan_distance(board):
    """Calculate Manhattan distance heuristic"""
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    distance = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] != 0:
                for gi in range(3):
                    for gj in range(3):
                        if goal[gi][gj] == board[i][j]:
                            distance += abs(i - gi) + abs(j - gj)
    return distance

def misplaced_tiles(board):
    """Calculate misplaced tiles heuristic"""
    goal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
    misplaced = 0
    for i in range(3):
        for j in range(3):
            if board[i][j] != 0 and board[i][j] != goal[i][j]:
                misplaced += 1
    return misplaced

def is_solvable(board):
    """Check if the puzzle is solvable using inversion count"""
    flat = [num for row in board for num in row if num != 0]
    inversions = 0
    for i in range(len(flat)):
        for j in range(i + 1, len(flat)):
            if flat[i] > flat[j]:
                inversions += 1
    return inversions % 2 == 0

def reconstruct_path(state):
    """Reconstruct the solution path from goal to start"""
    path = []
    current = state
    while current.parent is not None:
        path.append((current.move, current.board))
        current = current.parent
    path.reverse()
    return path

def a_star_search(initial_board, heuristic_func, heuristic_name):
    """A* search algorithm with specified heuristic"""
    start_time = time.time()
    initial_state = PuzzleState(initial_board)
    initial_state.h = heuristic_func(initial_board)
    initial_state.f = initial_state.g + initial_state.h

    open_list = []
    heapq.heappush(open_list, initial_state)
    closed_set = set()
    visited_costs = {tuple(map(tuple, initial_board)): initial_state.g}

    nodes_expanded = 0

    while open_list:
        current = heapq.heappop(open_list)

        if heuristic_func(current.board) == 0:
            end_time = time.time()
            path = reconstruct_path(current)
            return {
                'solution': path,
                'nodes_expanded': nodes_expanded,
                'time': end_time - start_time,
                'solution_length': len(path)
            }

        board_tuple = tuple(map(tuple, current.board))
        if board_tuple in closed_set:
            continue

        closed_set.add(board_tuple)
        nodes_expanded += 1

        for child in current.get_children():
            child.h = heuristic_func(child.board)
            child.f = child.g + child.h
            child_tuple = tuple(map(tuple, child.board))

            if child_tuple not in closed_set:
                if child_tuple not in visited_costs or child.g < visited_costs[child_tuple]:
                    visited_costs[child_tuple] = child.g
                    heapq.heappush(open_list, child)

    return None  # No solution found

def print_board(board):
    for row in board:
        print(' '.join(str(x) if x != 0 else '_' for x in row))
    print()

def main():
    # Test configurations
    test_cases = [
        [[1, 2, 3], [4, 5, 6], [7, 8, 0]],  # Already solved
        [[1, 2, 3], [4, 5, 0], [7, 8, 6]],  # Easy
        [[1, 2, 3], [4, 0, 5], [7, 8, 6]],  # Medium
        [[2, 8, 3], [1, 6, 4], [7, 0, 5]],  # Hard
    ]

    print("=" * 60)
    print("8-Puzzle Solver using A* Search Algorithm")
    print("=" * 60)

    for idx, initial_board in enumerate(test_cases, 1):
        print(f"\nTest Case {idx}:")
        print("Initial State:")
        print_board(initial_board)

        # Check solvability
        if not is_solvable(initial_board):
            print("This configuration is UNSOLVABLE!")
            continue

        # Run with Manhattan distance
        print("Running A* with Manhattan Distance heuristic...")
        result_manhattan = a_star_search(initial_board, manhattan_distance, "Manhattan")

        # Run with Misplaced Tiles
        print("Running A* with Misplaced Tiles heuristic...")
        result_misplaced = a_star_search(initial_board, misplaced_tiles, "Misplaced Tiles")

        # Compare results
        print("\n" + "-" * 60)
        print("PERFORMANCE COMPARISON")
        print("-" * 60)
        print(f"{'Metric':<25} {'Manhattan':<15} {'Misplaced Tiles':<15}")
        print("-" * 60)
        print(f"{'Nodes Expanded':<25} {result_manhattan['nodes_expanded']:<15} {result_misplaced['nodes_expanded']:<15}")
        print(f"{'Time (seconds)':<25} {result_manhattan['time']:.4f}{'':<11} {result_misplaced['time']:.4f}")
        print(f"{'Solution Length':<25} {result_manhattan['solution_length']:<15} {result_misplaced['solution_length']:<15}")
        print("-" * 60)

        # Print solution path for first heuristic
        print(f"\nSolution Path (Manhattan Distance):")
        for step, (move, board) in enumerate(result_manhattan['solution'], 1):
            print(f"Step {step}: {move}")
            print_board(board)

if __name__ == "__main__":
    main()
