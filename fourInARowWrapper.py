import numpy as np
from gym import spaces
import gym
import fourInARow
import copy
import tensorflow as tf

tf.compat.v1.disable_eager_execution()

class ActionSpace(spaces.Discrete):
    def __init__(self, size):
        self.high = fourInARow.width
        self.low = 0

        super().__init__(size)

class FourInARowWrapper(gym.Env):

    def __init__(self, player):
        self.player = player
        self.action_space = ActionSpace(fourInARow.width)
        #self.action_space = ActionSpace([0], [8])
        fourInARow.init(player)

        self.state = self.getHotEncodedState2d()

        self.checkWinConvSetup()

        self.sess = tf.compat.v1.Session()


    def ansi(self, style):
        return "\033[{0}m".format(style)

    def seed(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        return np.random.random_sample()

    def reset(self, player):
        fourInARow.init(player)
        self.player = player
        return self.getHotEncodedState2d()

    def step(self, action):
        fourInARow.drop_disc(int(action))
        reward = 0
        if fourInARow.state != "Playing":
            if fourInARow.winner == fourInARow.player:
                reward = -1
            elif fourInARow.winner == (fourInARow.player ^ 3):
                reward = 1
            else:
                reward = 0.5
            done = True
        else:
            done = False

        stateOneHotEncoded = self.getHotEncodedState2d()

        self.state = stateOneHotEncoded

        return (stateOneHotEncoded, reward, done, None)

    def robotStep(self, level):
        action = self.getRobotAction(level)
        #print("Robot action:", action+1)
        stateOneHotEncoded, reward, done, _ = self.step(action)
        return (stateOneHotEncoded, reward, done, action)

    def getRobotAction(self, level):
        import random

        rules = [self.ruleIsWinHyper(1), self.ruleIsLoseHyper(1), self.ruleIsLoseHyper(2), self.ruleIsWinHyper(2), self.ruleIsWinHyper(3)]
        if level > len(rules):
            print("Error! Level can't be bigger than", len(rules))
            raise Exception

        rules = rules[:level]
        col_points = [0 for _ in range(fourInARow.width)]
        candidates = list(range(fourInARow.width))
        # for col in columns:
        #     #l = [x for rule in rules for x in rule]
        #     col_points[col] = sum(rules[x](col) << (len(rules)-x) for x in range(len(rules)))
        for rule in rules:
            new_candidates = []

            for column in candidates:
                if not self.ruleIsAvaliable(column):
                    col_points[column] = -999999999
                    continue
                col_points[column] = rule(column)

            max_points = max(col_points)
            min_points = min(col_points)
            #print("Rule", rules.index(rule), "Column points:", col_points)
            for column in candidates:
                if col_points[column] == max_points:
                    new_candidates.append(column)
            candidates = new_candidates

            if len(candidates) == 1 or max_points >= 1:
                break

        candidates = [i for i in range(len(col_points)) if col_points[i] == max_points]
        ret_col = random.choice(candidates)
        #print("Cadidates:", candidates)

        #print("Column points:", col_points)
        return ret_col

    def ruleIsAvaliable(self, column):
        return fourInARow.get_available_cols()[column]

    # def recTheoreticalPlaying(self, depth, board, player, next_player):
    #     str = ""
    #     for c in range(depth):
    #         str += "-"
    #
    #     ret = False
    #
    #     if depth == 0:
    #         return self.checkWin(board, player)
    #
    #     for column in range(fourInARow.width):
    #         newBoard = copy.deepcopy(board)
    #
    #         col_height = sum([1 for x in newBoard[column] if x > 0])
    #
    #         # Check if column is full
    #         if col_height == fourInARow.height:
    #             continue
    #
    #         newBoard[column][col_height] = next_player
    #
    #         #self.renderBoardPlayer(newBoard, next_player)
    #
    #         #print(str, "column:", column, "player:", next_player, "depth:", depth, "checking win for player:", player)
    #
    #         if depth > 1:
    #             ret += self.recTheoreticalPlaying(depth-1, newBoard, player, next_player ^ 3)/(depth**fourInARow.width)
    #         else:
    #             ret += self.checkWin(newBoard, player)
    #             #if self.checkWin(newBoard, player):
    #                 #print(str, "win:", self.checkWin(newBoard, player), "---------------------------------------------------------------")
    #
    #     #print(str, "loop return:", ret)
    #     return ret

    def recTheoreticalPlaying(self, depth, board, player, next_player, it):
        ret = 0
        win = 0

        for column in range(fourInARow.width):
            it += 1
            newBoard = copy.deepcopy(board)

            col_height = sum([1 for x in newBoard[column] if x > 0])

            # Check if column is full
            if col_height == fourInARow.height:
                continue

            # Drop imaginary disc
            newBoard[column][col_height] = next_player

            if depth >= 1:
                win = self.checkWin(newBoard, player) * (depth + 1)
                if win == 0:
                    win, it = self.recTheoreticalPlaying(depth-1, newBoard, player, next_player ^ 3, it)

                ret += win

        return ret, it


    def recTheoreticalBinaryPlaying(self, depth, board, player, next_player, it):
        str = ""
        for c in range(depth):
            str += "-"

        ret = 0

        # ret = self.checkWin(board, player)
        #
        # if ret:
        #     return ret

        for column in range(fourInARow.width):
            it += 1
            newBoard = copy.deepcopy(board)

            col_height = sum([1 for x in newBoard[column] if x > 0])

            # Check if column is full
            if col_height == fourInARow.height:
                continue

            newBoard[column][col_height] = next_player

            #self.renderBoardPlayer(newBoard, next_player)

            #print(str, "column:", column, "player:", next_player, "depth:", depth, "checking win for player:", player, "Iteration:", it)

            if depth >= 1:
                ret, it = self.recTheoreticalBinaryPlaying(depth-1, newBoard, player, next_player ^ 3, it)
                if ret > 0:
                    return ret, it
            else:
                ret = self.checkWin(newBoard, player)*(depth+1)
                if ret > 0:
                    #print(str, "win:", ret, "column:", column, "player:", next_player, "depth:", depth, "checking win for player:", player)
                    return ret, it

        #print(str, "loop return:", ret)
        return ret, it

    def ruleIsWinHyper(self, drops):
        def ruleIsWinN(column):
            board = copy.deepcopy(fourInARow.board)
            col_height = sum([1 for x in board[column] if x > 0])

            if col_height == fourInARow.height:
                return False

            mePlayer = fourInARow.player
            opponent = mePlayer ^ 3

            # Drop disc
            board[column][col_height] = fourInARow.player
            #print("player", fourInARow.player, "tries column", column)

            #ret = self.recTheoreticalPlaying(depth=(drops-1)*2, board=board, player=mePlayer, next_player=opponent)
            ret, it = self.recTheoreticalPlaying(depth=(drops - 1) * 2, board=board, player=mePlayer, next_player=opponent, it=0)

            #print("points:", ret)
            #print("Iterations:", it)

            return ret*1

        return ruleIsWinN

    def ruleIsLoseHyper(self, drops):
        def ruleIsLoseN(column):
            board = copy.deepcopy(fourInARow.board)
            col_height = sum([1 for x in board[column] if x > 0])

            if col_height == fourInARow.height:
                return False

            # Drop disc
            board[column][col_height] = fourInARow.player
            #print("player", fourInARow.player, "tries column", column)

            mePlayer = fourInARow.player
            opponent = mePlayer ^ 3

            #ret = self.recTheoreticalPlaying(depth=drops*2-1, board=board, player=mePlayer ^ 3, next_player=opponent)
            ret, it = self.recTheoreticalPlaying(depth=drops * 2 - 1, board=board, player=mePlayer ^ 3,
                                             next_player=opponent, it=0)
            #print("Iterations:", it)
            #print("Points:", ret)

            return -(ret*1)

        return ruleIsLoseN

    def ruleIsWin(self, column):
        #print("rule1:", column)
        # Check if dropping in column is a winning turn
        board = copy.deepcopy(fourInARow.board)
        col_height = sum([1 for x in board[column] if x > 0])
        if col_height == fourInARow.height:
            return False
        # Drop disc
        board[column][col_height] = fourInARow.player

        return self.checkWin(board, fourInARow.player)

    def ruleIsBlockLose(self, column):
        # Check if opponent wins if I don't drop in a column
        board = copy.deepcopy(fourInARow.board)
        col_height = sum([1 for x in board[column] if x > 0])
        if col_height == fourInARow.height:
            return False
        # Inverse player, drop disc
        board[column][col_height] = fourInARow.player^3

        return self.checkWin(board, fourInARow.player)

    def ruleIsBlockLoseTwoStepsAhead(self, column):
        # Check if column is already full
        col_height = sum([1 for x in fourInARow.board[column] if x > 0])
        if col_height == fourInARow.height:
            return False

        for opponentColumn in range(fourInARow.width):
            board = copy.deepcopy(fourInARow.board)

    def checkWin(self, board, player):
        #return self.checkWinConv(board, player)
        deltas = [1, 2, 3]
        winner = 0
        for x in range(fourInARow.width):
            right = x < fourInARow.width - 3
            for y in range(fourInARow.height):
                upper = y < fourInARow.height - 3
                lower = y > 2
                if board[x][y] > 0 and \
                        ((upper and 3 == sum(1 for d in deltas if board[x][y] == board[x][y + d])) or \
                         (right and 3 == sum(1 for d in deltas if board[x][y] == board[x + d][y])) or \
                         (right and upper and 3 == sum(1 for d in deltas if board[x][y] == board[x + d][y + d])) or \
                         (right and lower and 3 == sum(1 for d in deltas if board[x][y] == board[x + d][y - d]))):
                    winner = board[x][y]
        if player == winner:
            return 1
        else:
            return 0



    def checkWinConvSetup(self):
        filterValues = np.zeros(shape=(4, 4, 1, 20), dtype="float32")
        for i in range(4):
            for j in range(4):
                filterValues[i, j, 0, i] = -1
                filterValues[j, i, 0, i+4] = -1

                filterValues[i, j, 0, i + 10] = 1
                filterValues[j, i, 0, i + 4 + 10] = 1

            filterValues[i, i, 0, 8] = -1
            filterValues[3-i, i, 0, 9] = -1

            filterValues[i, i, 0, 8 + 10] = 1
            filterValues[3 - i, i, 0, 9 + 10] = 1

        # for j in range(20):
        #     print("filter:", j)
        #     print(filterValues[:, :, 0, j])

        self.TFboard = tf.compat.v1.placeholder(tf.float32, (1, 7, 6, 1), "board")

        self.TFboardScale = 6.5 * tf.multiply(self.TFboard, self.TFboard) - 9.5 * self.TFboard

        self.winFilter = tf.constant(value=filterValues, name="winCheckFilter")

        self.conv = tf.compat.v1.nn.conv2d(
            input=self.TFboardScale,
            filter=self.winFilter,
            strides=(1, 1, 1, 1),
            padding="VALID",
            name="winConv"
        )

    def checkWinConv(self, board, player):

        res = self.sess.run(self.conv, {self.TFboard: np.expand_dims(np.expand_dims(board, -1), 0)})

        maxValueOne = res[0, :, :, :9]
        maxValueTwo = res[0, :, :, 10:]

        #print("maxValueOne:", maxValueOne, "maxValueTwo:", maxValueTwo)
        #print(tfBoard)

        # Player one wins
        if np.any(maxValueOne > 11.5) and np.any(maxValueOne < 12.5) and player < 1.5:
            #print("maxValueOne:", maxValueOne,"\nRes:", res[0, :, :, :9], "\nBoard:", board)
            return True

        # Player two wins
        elif np.any(maxValueTwo > 27.5) and np.any(maxValueTwo < 28.5) and player > 1.5:
            #print("maxValueTwo:", maxValueTwo, "\nRes:", res[0, :, :, 10:], "\nBoard:", board)
            return True

        else:
            return False

        #print("winner:", winner, "player:", player)


    def getAvaliableColumns(self):
        return np.reshape(np.array(fourInARow.get_available_cols()).astype(np.float32), (fourInARow.width))

    def render(self, mode='human'):
        self.renderBoardPlayer(fourInARow.board, fourInARow.player)
        # if fourInARow.player == 1:
        #     player = "X"
        # else:
        #     player = "O"
        #
        # print("Player:", player, "\n")
        # row = "  "
        # for n in range(fourInARow.width):
        #     row += str(n+1) + "   "
        # print(row)
        #
        # row = "|"
        # for _ in range(fourInARow.width):
        #     row += "---|"
        # print(row)
        #
        # for y in range(fourInARow.height):
        #     row = "|"
        #     for x in range(fourInARow.width):
        #         color = 30 + fourInARow.board[x][fourInARow.height-y-1]
        #         character = "   "
        #
        #         if fourInARow.board[x][fourInARow.height - y - 1] == 1:
        #             character = " X "
        #         elif fourInARow.board[x][fourInARow.height - y - 1] == 2:
        #             character = " O "
        #
        #         if fourInARow.latest == (x, fourInARow.height-y-1):
        #             color += 10
        #         row += self.ansi(color) + character + self.ansi(0) + "|"
        #
        #     print(row)
        #
        #     row = "|"
        #     for _ in range(fourInARow.width):
        #         row += "---|"
        #     print(row)

        #print("\n")

    def renderBoardPlayer(self, board, player):
        if player == 1:
            player = "X"
        else:
            player = "O"

        print("Player:", player, "\n")
        row = "  "
        for n in range(fourInARow.width):
            row += str(n+1) + "   "
        print(row)

        row = "|"
        for _ in range(fourInARow.width):
            row += "---|"
        print(row)

        for y in range(fourInARow.height):
            row = "|"
            for x in range(fourInARow.width):
                color = 30 + board[x][fourInARow.height-y-1]
                character = "   "

                if board[x][fourInARow.height - y - 1] == 1:
                    character = " X "
                elif board[x][fourInARow.height - y - 1] == 2:
                    character = " O "

                if fourInARow.latest == (x, fourInARow.height-y-1):
                    color += 10
                row += self.ansi(color) + character + self.ansi(0) + "|"

            print(row)

            row = "|"
            for _ in range(fourInARow.width):
                row += "---|"
            print(row)

        #print("\n")

    def close(self):
        pass

    def getHotEncodedState(self):
        board = np.reshape(np.array(fourInARow.board), fourInARow.height * fourInARow.width)
        boardOneHotEncoded = np.zeros(fourInARow.height * fourInARow.width * 2)

        player = fourInARow.player
        playerOneHotEncoded = np.zeros(2)

        if player == 1:
            playerOneHotEncoded[0] = 1
            playerOneHotEncoded[1] = 0
        elif player == 2:
            playerOneHotEncoded[0] = 0
            playerOneHotEncoded[1] = 1

        for i in range(board.size):
            if board[i] == 1:
                boardOneHotEncoded[2 * i] = 1
                boardOneHotEncoded[2 * i + 1] = 0
            elif board[i] == 2:
                boardOneHotEncoded[2 * i] = 0
                boardOneHotEncoded[2 * i + 1] = 1
            else:
                boardOneHotEncoded[2 * i] = 0
                boardOneHotEncoded[2 * i + 1] = 0

        return np.concatenate([playerOneHotEncoded, boardOneHotEncoded])

    def getHotEncodedState2d(self):
        board = np.array(fourInARow.board)
        boardOneHotEncoded = np.resize(np.expand_dims(np.zeros(board.shape), axis=2), (7,6,2))

        player = fourInARow.player
        playerOneHotEncoded = np.zeros(2)

        if player == 1:
            playerOneHotEncoded[0] = 1
            playerOneHotEncoded[1] = 0
        elif player == 2:
            playerOneHotEncoded[0] = 0
            playerOneHotEncoded[1] = 1

        for x in range(board.shape[0]):
            for y in range(board.shape[1]):
                if board[x][y] == 1:
                    boardOneHotEncoded[x][y][0] = 1
                    boardOneHotEncoded[x][y][1] = 0
                elif board[x][y] == 2:
                    boardOneHotEncoded[x][y][0] = 0
                    boardOneHotEncoded[x][y][1] = 1
                else:
                    boardOneHotEncoded[x][y][0] = 0
                    boardOneHotEncoded[x][y][1] = 0

        return (playerOneHotEncoded, boardOneHotEncoded)

    def getCurrentPlayer(self):
        return fourInARow.player

    def renderHotEncodedState(self, hotEncodedState):
        hotEncodedPlayer = hotEncodedState[0]
        hotEncodedBoard = hotEncodedState[1]

        print(hotEncodedPlayer)

        if hotEncodedPlayer[0] == 1:
            player = "X"
        elif hotEncodedPlayer[1] == 1:
            player = "O"
        else:
            print("No player in state")

        print("Player:", player, "\n")
        row = "  "
        for n in range(fourInARow.width):
            row += str(n+1) + "   "
        print(row)

        row = "|"
        for _ in range(fourInARow.width):
            row += "---|"
        print(row)

        for y in range(fourInARow.height):
            row = "|"
            for x in range(fourInARow.width):
                color = 30# + hotEncodedBoard[2*x + (fourInARow.height-2*y)*fourInARow.width-1]
                character = "   "

                if hotEncodedBoard[x][fourInARow.height-y-1][0] == 1:
                    character = " X "
                elif hotEncodedBoard[x][fourInARow.height-y-1][1] == 1:
                    character = " O "

                row += self.ansi(color) + character + self.ansi(0) + "|"

            print(row)

            row = "|"
            for _ in range(fourInARow.width):
                row += "---|"
            print(row)

def invertBoard(inBoard):
    invertedBoard = np.array(inBoard)

    board_shape = inBoard.shape

    #print("Shape:", board_shape)

    for x in range(board_shape[0]):
        for y in range(board_shape[1]):
            invertedBoard[x][y][0] = inBoard[x][y][1]
            invertedBoard[x][y][1] = inBoard[x][y][0]

    return invertedBoard
#
# env = FourInARowWrapper(1)
# #
# # ruleIsLoseInOne = env.ruleIsLoseHyper(1)
# # ruleIsWinInOne = env.ruleIsWinHyper(1)
# #
# while fourInARow.state == "Playing":
#
#     env.robotStep(4)
#     env.render()
#
#     key = input()
#     env.step(int(key) - 1)
#
#
#
# env.step(2)
# env.render()
# env.step(2)
#
# env.step(3)
# env.render()
# env.robotStep(4)
#
# env.step(4)
# env.render()
# env.robotStep(4)
#
# env.step(1)
# env.render()
#
#
#
# env.render()

    # for c in range(fourInARow.width):
    #     print("Win in one, column", c, ":", ruleIsWinInOne(c))
    #     print("Loose in onem column", c, ":", ruleIsLoseInOne(c))
