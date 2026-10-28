def print_board(board):
    print()
    for i in range(3):
        print(" | ".join(board[i]))
        if i < 2:
            print("--+---+--")
    print()


def check_winner(board):
    # Rows
    for row in board:
        if row[0] == row[1] == row[2] and row[0] != " ":
            return row[0]

    # Columns
    for col in range(3):
        if board[0][col] == board[1][col] == board[2][col] \
                and board[0][col] != " ":
            return board[0][col]

    # Diagonals
    if board[0][0] == board[1][1] == board[2][2] \
            and board[0][0] != " ":
        return board[0][0]

    if board[0][2] == board[1][1] == board[2][0] \
            and board[0][2] != " ":
        return board[0][2]

    return None


def is_full(board):
    for row in board:
        if " " in row:
            return False
    return True


def play_game():
    board = [[" " for _ in range(3)] for _ in range(3)]

    player = "X"

    while True:
        print_board(board)

        row = int(input(f"Player {player}, enter row (0-2): "))
        col = int(input(f"Player {player}, enter column (0-2): "))

        if board[row][col] != " ":
            print("Position already occupied!")
            continue

        board[row][col] = player

        winner = check_winner(board)

        if winner:
            print_board(board)
            print("Winner:", winner)
            break

        if is_full(board):
            print_board(board)
            print("Draw!")
            break

        player = "O" if player == "X" else "X"


play_game()