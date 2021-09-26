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
    now_x = 0
    nox_o = 0

    for i in range(0, len(board)):
        for j in range(0, len(board[0])):
            if board[i][j] == X:
                now_x += 1
            elif board[i][j] == O:
                nox_o += 1

    if now_x > nox_o:
        return O
    else:
        return X


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """
    possiblity = set()

    for i in range(0, 3):
        for j in range(0, 3):
            if board[i][j] == EMPTY:
                possiblity.add((i, j))
    return possiblity


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Create new board, without modifying the original board received as input
    minresult = copy.deepcopy(board)
    minresult[action[0]][action[1]] = player(board)
    return minresult


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    if board[0][0] == board[1][0] and board[1][0] == board[2][0]:
        return board[0][0]
    elif board[0][1] == board[1][1] and board[1][1] == board[2][1]:
        return board[0][1]
    elif board[0][2] == board[1][2] and board[1][2] == board[2][2]:
        return board[0][2]
    elif board[0][0] == board[1][1] and board[1][1] == board[2][2]:
        return board[0][0]
    elif board[0][2] == board[1][1] and board[1][1] == board[2][0]:
        return board[0][2]
    elif all(i == board[0][0] for i in board[0]):
        return board[0][0]
    elif all(i == board[1][0] for i in board[1]):
        return board[1][0]
    elif all(i == board[2][0] for i in board[2]):
        return board[2][0]
    else:
        return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """

    if winner(board) is not None or (not any(EMPTY in sublist for sublist in board) and winner(board) is None):
        return True
    else:
        return False


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    if terminal(board):
        if winner(board) == X:
            return 1
        elif winner(board) == O:
            return -1
        else:
            return 0
    # Check how to handle exception when a non terminal board is received.


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # #极小极大算法实现
    # if terminal(board):    #判断是否为终局
    #     return None
    # else:
    #     if player(board) == X:  #判断场上的先手关系
    #         value, move = max_value(board)  #针对极大结点计算极大值
    #         return move
    #     else:
    #         value, move = min_value(board)  #针对极小结点计算极小值
    #         return move

    #alpha beta剪枝算法
    return alphbetaSearch(board)

def max_value(board):   #针对极大结点计算极大值
    if terminal(board):
        return utility(board), None

    v = float('-inf')   #设置为负无穷
    move = None
    for action in actions(board):
        # v = max(v, min_value(result(board, action)))
        term, act = min_value(result(board, action))
        if term > v:
            v = term
            move = action
            if v == 1:
                return v, move

    return v, move

def min_value(board):    #针对极小结点计算极小值
    if terminal(board):
        return utility(board), None

    v = float('inf')    #设置为正无穷
    move = None
    for action in actions(board):
        # v = max(v, min_value(result(board, action)))
        term, act = max_value(result(board, action))
        if term < v:
            v = term
            move = action
            if v == -1:
                return v, move

    return v, move


#alpha beta剪枝算法
def alphbetaSearch(board):
    if terminal(board):    #判断是否为终局
        return None
    else:
        pos_infinity = float('inf')
        neg_infinity = float('-inf')
        if player(board) == X:  #判断场上的先手关系
            value, move = alphaMax_value(neg_infinity,pos_infinity,board)  #针对极大结点计算极大值
            return move
        else:
            value, move = alphaMin_value(pos_infinity,neg_infinity,board)  #针对极小结点计算极小值
            return move


def alphaMax_value(alpha,beta,board):
    if terminal(board):
        return utility(board), None

    v = float('-inf')

    for action in actions(board):
        term, act = alphaMin_value(alpha,beta,result(board, action))
        if v>=beta:
            return v,action
        alpha = max(alpha,v);
    raise NotImplementedError

def alphaMin_value(alpha,beta,board):
    if terminal(board):
        return utility(board), None

    v = float('inf')

    for action in actions(board):
        term, act = alphaMax_value(alpha, beta, result(board, action))
        if v <= alpha:
            return v,action
        beta = min(beta,v)
    raise NotImplementedError