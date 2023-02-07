"""
Tic Tac Toe Player
"""

import math
import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    if (terminal(board)):
        return None
    
    countX, countO = 0, 0
    for row in board:
        for cell in row:
            if cell == X:
                countX += 1
            elif cell == O:
                countO += 1
    
    if countX > countO:
        return O
    else:
        return X

    # raise NotImplementedError


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    if (terminal(board)):
        return None

    empty_cells = set()
    for i in range(3):
        for j in range(3):
            if board[i][j] == EMPTY:
                empty_cells.add((i,j))
    
    return empty_cells
            
    # raise NotImplementedError


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    if (action[0]<0 or action[0]>2 or action[1]<0 or action[1]>2):
        raise Exception("Invalid coordinate")
    if (board[action[0]][action[1]] != EMPTY):
        raise Exception("The cell is not empty")

    next_board = copy.deepcopy(board)
    next_board[action[0]][action[1]] = player(board)
    return next_board
    # raise NotImplementedError


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # horizontal
    for i in range(3):
        row = [board[i][0], board[i][1], board[i][2]]
        if row == [X, X, X]:
            return X
        elif row == [O, O, O]:
            return O

    # vertical
    for j in range(3):
        row = [board[0][j], board[1][j], board[2][j]]
        if row == [X, X, X]:
            return X
        elif row == [O, O, O]:
            return O

    # diagonal
    row = [board[0][0], board[1][1], board[2][2]]
    if row == [X, X, X]:
        return X
    elif row == [O, O, O]:
        return O

    row = [board[0][2], board[1][1], board[2][0]]
    if row == [X, X, X]:
        return X
    elif row == [O, O, O]:
        return O

    return None

    # raise NotImplementedError


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    if winner(board) is not None:
        return True
    
    for row in board:
        for cell in row:
            if cell == EMPTY:
                return False
    
    return True
    # raise NotImplementedError


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if winner(board) == X:
        return 1
    elif winner(board) == O:
        return -1 
    return 0
    # raise NotImplementedError


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    if terminal(board):
        return None

    possible_actions = actions(board)
    best_action = (0,0)
    if player(board) == X:
        vM = -math.inf
    else:
        vM = math.inf
    
    for action in possible_actions:
        if player(board) == X:
            v = minValue(result(board, action))
            if v > vM:
                vM = v
                best_action = action
        else:
            v = maxValue(result(board, action))
            if v < vM:
                vM = v
                best_action = action
    
    return best_action
    
    # raise NotImplementedError

def minValue(board):
    v = math.inf
    if terminal(board):
        return utility(board)

    possible_actions = actions(board)
    for action in possible_actions:
        v = min(v, maxValue(result(board, action)) )
    
    return v

def maxValue(board):
    v = -math.inf
    if terminal(board):
        return utility(board)

    possible_actions = actions(board)
    for action in possible_actions:
        v = max(v, minValue(result(board, action)) )
    
    return v
