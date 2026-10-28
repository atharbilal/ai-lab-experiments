"""
Experiment 3: Minimax with Alpha-Beta Pruning
Implements intelligent game-playing agent for Tic-Tac-Toe with performance comparison.
"""

import time
from copy import deepcopy

class TicTacToe:
    def __init__(self):
        self.board = [[' ' for _ in range(3)] for _ in range(3)]
        self.current_player = 'X'

    def print_board(self):
        print("\n  0   1   2")
        for i in range(3):
            print(f"{i} {' | '.join(self.board[i])}")
            if i < 2:
                print("  -----------")
        print()

    def make_move(self, row, col, player):
        if self.board[row][col] == ' ':
            self.board[row][col] = player
            return True
        return False

    def undo_move(self, row, col):
        self.board[row][col] = ' '

    def get_valid_moves(self):
        moves = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] == ' ':
                    moves.append((i, j))
        return moves

    def check_winner(self):
        # Check rows
        for row in self.board:
            if row[0] == row[1] == row[2] != ' ':
                return row[0]

        # Check columns
        for col in range(3):
            if self.board[0][col] == self.board[1][col] == self.board[2][col] != ' ':
                return self.board[0][col]

        # Check diagonals
        if self.board[0][0] == self.board[1][1] == self.board[2][2] != ' ':
            return self.board[0][0]
        if self.board[0][2] == self.board[1][1] == self.board[2][0] != ' ':
            return self.board[0][2]

        # Check for draw
        if all(cell != ' ' for row in self.board for cell in row):
            return 'Draw'

        return None

    def is_terminal(self):
        return self.check_winner() is not None

    def evaluate(self, player):
        """Heuristic evaluation function for non-terminal states"""
        winner = self.check_winner()
        if winner == player:
            return 10
        elif winner == 'Draw':
            return 0
        elif winner:
            return -10

        # Heuristic: count potential winning lines
        opponent = 'O' if player == 'X' else 'X'
        score = 0

        # Check rows
        for row in self.board:
            if row.count(player) == 2 and row.count(' ') == 1:
                score += 1
            if row.count(opponent) == 2 and row.count(' ') == 1:
                score -= 1

        # Check columns
        for col in range(3):
            column = [self.board[row][col] for row in range(3)]
            if column.count(player) == 2 and column.count(' ') == 1:
                score += 1
            if column.count(opponent) == 2 and column.count(' ') == 1:
                score -= 1

        # Check diagonals
        diag1 = [self.board[i][i] for i in range(3)]
        diag2 = [self.board[i][2-i] for i in range(3)]

        if diag1.count(player) == 2 and diag1.count(' ') == 1:
            score += 1
        if diag1.count(opponent) == 2 and diag1.count(' ') == 1:
            score -= 1

        if diag2.count(player) == 2 and diag2.count(' ') == 1:
            score += 1
        if diag2.count(opponent) == 2 and diag2.count(' ') == 1:
            score -= 1

        return score

    def minimax(self, depth, is_maximizing, player, alpha, beta, use_alpha_beta):
        """Minimax algorithm with optional Alpha-Beta pruning"""
        if self.is_terminal() or depth == 0:
            return self.evaluate(player)

        valid_moves = self.get_valid_moves()

        # Move ordering: prioritize center and corners for better pruning
        move_order = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)]
        valid_moves = sorted(valid_moves, key=lambda x: move_order.index(x) if x in move_order else 9)

        if is_maximizing:
            max_eval = float('-inf')
            for row, col in valid_moves:
                self.make_move(row, col, player)
                eval_score = self.minimax(depth - 1, False, player, alpha, beta, use_alpha_beta)
                self.undo_move(row, col)
                max_eval = max(max_eval, eval_score)

                if use_alpha_beta:
                    alpha = max(alpha, eval_score)
                    if beta <= alpha:
                        break  # Beta cutoff
            return max_eval
        else:
            min_eval = float('inf')
            opponent = 'O' if player == 'X' else 'X'
            for row, col in valid_moves:
                self.make_move(row, col, opponent)
                eval_score = self.minimax(depth - 1, True, player, alpha, beta, use_alpha_beta)
                self.undo_move(row, col)
                min_eval = min(min_eval, eval_score)

                if use_alpha_beta:
                    beta = min(beta, eval_score)
                    if beta <= alpha:
                        break  # Alpha cutoff
            return min_eval

    def find_best_move(self, player, depth=9, use_alpha_beta=True):
        """Find the best move using Minimax"""
        nodes_evaluated = [0]  # Use list to make it mutable in nested function

        def minimax_count(board_state, depth, is_maximizing, player, alpha, beta, use_alpha_beta):
            nodes_evaluated[0] += 1
            if board_state.is_terminal() or depth == 0:
                return board_state.evaluate(player)

            valid_moves = board_state.get_valid_moves()
            move_order = [(1, 1), (0, 0), (0, 2), (2, 0), (2, 2), (0, 1), (1, 0), (1, 2), (2, 1)]
            valid_moves = sorted(valid_moves, key=lambda x: move_order.index(x) if x in move_order else 9)

            if is_maximizing:
                max_eval = float('-inf')
                for row, col in valid_moves:
                    board_state.make_move(row, col, player)
                    eval_score = minimax_count(board_state, depth - 1, False, player, alpha, beta, use_alpha_beta)
                    board_state.undo_move(row, col)
                    max_eval = max(max_eval, eval_score)
                    if use_alpha_beta:
                        alpha = max(alpha, eval_score)
                        if beta <= alpha:
                            break
                return max_eval
            else:
                min_eval = float('inf')
                opponent = 'O' if player == 'X' else 'X'
                for row, col in valid_moves:
                    board_state.make_move(row, col, opponent)
                    eval_score = minimax_count(board_state, depth - 1, True, player, alpha, beta, use_alpha_beta)
                    board_state.undo_move(row, col)
                    min_eval = min(min_eval, eval_score)
                    if use_alpha_beta:
                        beta = min(beta, eval_score)
                        if beta <= alpha:
                            break
                return min_eval

        valid_moves = self.get_valid_moves()
        best_move = None
        best_value = float('-inf')

        for row, col in valid_moves:
            self.make_move(row, col, player)
            move_value = minimax_count(self, depth - 1, False, player, float('-inf'), float('inf'), use_alpha_beta)
            self.undo_move(row, col)

            if move_value > best_value:
                best_value = move_value
                best_move = (row, col)

        return best_move, nodes_evaluated[0]

def play_game(use_alpha_beta=True, depth=9):
    """Play a game against the AI"""
    game = TicTacToe()
    human = 'X'
    ai = 'O'

    print(f"Playing with {'Alpha-Beta Pruning' if use_alpha_beta else 'Pure Minimax'}")
    print(f"Search Depth: {depth}")
    print("You are X, AI is O")
    game.print_board()

    while not game.is_terminal():
        # Human move
        while True:
            try:
                row = int(input("Enter row (0-2): "))
                col = int(input("Enter column (0-2): "))
                if game.make_move(row, col, human):
                    break
                print("Invalid move! Try again.")
            except ValueError:
                print("Invalid input! Enter numbers 0-2.")

        game.print_board()

        if game.is_terminal():
            break

        # AI move
        print("AI is thinking...")
        start_time = time.time()
        best_move, nodes = game.find_best_move(ai, depth, use_alpha_beta)
        end_time = time.time()

        print(f"AI evaluated {nodes} nodes in {end_time - start_time:.4f} seconds")
        game.make_move(best_move[0], best_move[1], ai)
        print(f"AI plays: {best_move}")
        game.print_board()

    winner = game.check_winner()
    if winner == 'Draw':
        print("It's a draw!")
    elif winner == human:
        print("You win!")
    else:
        print("AI wins!")

def compare_algorithms():
    """Compare performance with and without Alpha-Beta pruning"""
    print("=" * 70)
    print("Minimax Algorithm Performance Comparison")
    print("=" * 70)

    test_scenarios = [
        [['X', ' ', ' '], [' ', 'O', ' '], [' ', ' ', ' ']],  # Near start
        [['X', 'O', 'X'], [' ', ' ', ' '], [' ', ' ', ' ']],  # Mid game
        [['X', 'O', 'X'], ['O', 'X', ' '], [' ', ' ', 'O']],  # Near end
    ]

    for idx, initial_board in enumerate(test_scenarios, 1):
        print(f"\nScenario {idx}:")
        game = TicTacToe()
        game.board = deepcopy(initial_board)
        game.print_board()

        # Test with Alpha-Beta pruning
        game_ab = TicTacToe()
        game_ab.board = deepcopy(initial_board)
        start_time = time.time()
        best_move_ab, nodes_ab = game_ab.find_best_move('X', depth=9, use_alpha_beta=True)
        time_ab = time.time() - start_time

        # Test without Alpha-Beta pruning
        game_no_ab = TicTacToe()
        game_no_ab.board = deepcopy(initial_board)
        start_time = time.time()
        best_move_no_ab, nodes_no_ab = game_no_ab.find_best_move('X', depth=9, use_alpha_beta=False)
        time_no_ab = time.time() - start_time

        print("-" * 70)
        print(f"{'Metric':<25} {'With Alpha-Beta':<20} {'Without Alpha-Beta':<20}")
        print("-" * 70)
        print(f"{'Nodes Evaluated':<25} {nodes_ab:<20} {nodes_no_ab:<20}")
        print(f"{'Time (seconds)':<25} {time_ab:.4f}{'':<15} {time_no_ab:.4f}")
        print(f"{'Best Move':<25} {str(best_move_ab):<20} {str(best_move_no_ab):<20}")
        print("-" * 70)

        if nodes_ab < nodes_no_ab:
            reduction = ((nodes_no_ab - nodes_ab) / nodes_no_ab) * 100
            print(f"Alpha-Beta reduced node evaluation by {reduction:.1f}%")

def main():
    print("\n" + "=" * 70)
    print("Tic-Tac-Toe with Minimax and Alpha-Beta Pruning")
    print("=" * 70)

    # Performance comparison
    compare_algorithms()

    # Interactive game (commented out for automated testing)
    # print("\n" + "=" * 70)
    # print("Interactive Game")
    # print("=" * 70)
    # play_game(use_alpha_beta=True, depth=9)

if __name__ == "__main__":
    main()
