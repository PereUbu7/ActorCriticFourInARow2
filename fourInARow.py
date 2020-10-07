width = 7
height = 6
player = 1
board = []
winner = 0
latest = 0
state = "NotInitialized"


def init(player_in):
    global state
    global player
    global winner
    global board
    player = player_in
    winner = 0
    state = "Playing"
    board = [[0] * height for _ in range(width)]


def set_winner():
    global winner
    deltas = [1, 2, 3]
    for x in range(width):
        right = x < width - 3
        for y in range(height):
            upper = y < height - 3
            lower = y > 2
            if board[x][y] > 0 and \
                    ((upper and 3 == sum(1 for d in deltas if board[x][y] == board[x][y + d])) or \
                     (right and 3 == sum(1 for d in deltas if board[x][y] == board[x + d][y])) or \
                     (right and upper and 3 == sum(1 for d in deltas if board[x][y] == board[x + d][y + d])) or \
                     (right and lower and 3 == sum(1 for d in deltas if board[x][y] == board[x + d][y - d]))):
                winner = board[x][y]


def resolve_state():
    global state
    if state == "Playing":
        set_winner()
        if winner > 0:
            state = "Player" + str(winner) + "Won"
        elif width == sum(1 for x in board if sum(1 for y in x if y > 0) == height):
            state = "Tie"


def drop_disc(col):
    global board
    global player
    global latest
    if state != "Playing" or col < 0 or col >= width:
        return False
    col_height = sum([1 for x in board[col] if x > 0])
    if col_height < height:
        board[col][col_height] = player
        latest = col, col_height
        player ^= 3
    else:
        return False

    resolve_state()
    return True


def get_available_cols():
    avabCols = []
    if True:
        for i in range(width):
            if board[i][height-1] > 0:
                avabCols.append(0)
            else:
                avabCols.append(1)

    return avabCols

# usage:
# init() # this starts a new game
# get_available_cols() # returns an array of all possible values of 'n' in drop_disc(n)
# drop_disc(n) # this drops a disc for current player in column 'n' (index 0)
# board # this is your current board state, row-array of column-arrays, containing 0 if empty, or 1 / 2 if player put disc there
# player # this is 1 or 2, depending on whos turn it is, player 1 starts
# state # this can be "NotInitialized" when starting this script, or "Playing", while playing and "Tie", "Player1Won" or "Player2Won" if the game has ended

