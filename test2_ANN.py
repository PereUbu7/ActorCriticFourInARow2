import tensorflow as tf
import numpy as np

def invertBoard(inBoard):
    invertedBoard = np.array(inBoard)

    board_shape = inBoard.shape

    #print("Shape:", board_shape)

    for x in range(board_shape[0]):
        for y in range(board_shape[1]):
            invertedBoard[x][y][0] = inBoard[x][y][1]
            invertedBoard[x][y][1] = inBoard[x][y][0]

    return invertedBoard

class ANN():
    """
    Function approximator
    """

    def __init__(self, scope="ANN", env=None, entropyFactor=0.1, perturbationSigma=0.1, valueLossFactor=0.1, policyLossFactor=1):
        if env is None:
            raise Exception("No environment")

        tf.reset_default_graph()

        with tf.variable_scope(scope):

            self.board = tf.placeholder(tf.float32, (None, 7, 6, 2), "board")
            self.policyTarget = tf.placeholder(dtype=tf.float32, name="policyTarget")
            self.valueTarget = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="valueTarget")
            self.validColumnsFilter = tf.placeholder(dtype=tf.float32, shape=(None, 7), name="validColumnsFilter")

            self.board_norm = tf.nn.batch_normalization(x=self.board, mean=0, variance=1, offset=1, scale=1,
                                                        variance_epsilon=1e-7)
            self.board_flat = tf.reshape(self.board_norm, (-1, 7 * 6 * 2))

            # Layers shared between policy and value functions
            # ---   Shared layer 1  ---
            self.sharedWeights1 = tf.get_variable(name="sharedWeights1", shape=(7*6*2, 800), dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=1))
            self.sharedBias1 = tf.get_variable(name="sharedBias1", shape=(1, 800), dtype=tf.float32, initializer=tf.zeros_initializer)

            # shape = (?, 800)
            self.sharedActivation1 = tf.nn.leaky_relu(features=tf.matmul(self.board_flat, self.sharedWeights1) + self.sharedBias1, alpha=0.1)

            # ---   Shared layer 2  ---
            self.sharedWeights2 = tf.get_variable(name="sharedWeights2", shape=(800, 500), dtype=tf.float32, initializer=tf.random_normal_initializer(mean=0.0, stddev=1))
            self.sharedBias2 = tf.get_variable(name="sharedBias2", shape=(1, 500), dtype=tf.float32, initializer=tf.zeros_initializer)

            # shape = (?, 500)
            self.sharedActivation2 = tf.nn.leaky_relu(
                features=tf.matmul(self.sharedActivation1, self.sharedWeights2) + self.sharedBias2, alpha=0.1)

            # Policy layers
            # ---   Policy layer 1  ---
            self.policyWeights1 = tf.get_variable(name="policyWeights1", shape=(500, 100), dtype=tf.float32,
                                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=1))
            self.policyBias1 = tf.get_variable(name="policyBias1", shape=(1, 100), dtype=tf.float32,
                                               initializer=tf.zeros_initializer)

            # shape = (?, 100)
            self.policyActivation1 = tf.nn.leaky_relu(
                features=tf.matmul(self.sharedActivation2, self.policyWeights1) + self.policyBias1, alpha=0.1)

            # ---   Policy layer 2  ---
            self.policyWeights2 = tf.get_variable(name="policyWeights2", shape=(100, env.action_space.high-env.action_space.low), dtype=tf.float32,
                                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=1))
            self.policyBias2 = tf.get_variable(name="policyBias2", shape=(1, env.action_space.high-env.action_space.low), dtype=tf.float32,
                                               initializer=tf.zeros_initializer)

            # shape = (?, 100)
            self.policyActivation2 = tf.nn.sigmoid(
                tf.matmul(self.policyActivation1, self.policyWeights2) + self.policyBias2)

            self.mu = tf.squeeze(self.policyActivation2)

            self.mu = tf.multiply(self.mu, self.validColumnsFilter) + 1e-6

            self.mu = tf.divide(self.mu, tf.reduce_sum(self.mu))

            self.dist = tf.contrib.distributions.Categorical(probs=self.mu, dtype=tf.float32)

            # Draw sample
            self.action = self.dist.sample()

            # Loss and train op. Add cross entropy cost to encourage exploration
            self.policyLoss = -self.dist.log_prob(self.action) * self.policyTarget + entropyFactor * self.dist.entropy()

            # Value function layers
            # ---   Value layer 1  ---
            self.valueWeights1 = tf.get_variable(name="valueWeights1", shape=(500, 100), dtype=tf.float32,
                                                  initializer=tf.random_normal_initializer(mean=0.0, stddev=1))
            self.valueBias1 = tf.get_variable(name="valueBias1", shape=(1, 100), dtype=tf.float32,
                                               initializer=tf.zeros_initializer)

            # shape = (?, 100)
            self.valueActivation1 = tf.nn.leaky_relu(
                features=tf.matmul(self.sharedActivation2, self.valueWeights1) + self.valueBias1, alpha=0.1, name="valueActivation1")

            # ---   Value layer 2  ---
            self.valueWeights2 = tf.get_variable(name="valueWeights2", shape=(100, 1), dtype=tf.float32,
                                                 initializer=tf.random_normal_initializer(mean=0.0, stddev=1))
            self.valueBias2 = tf.get_variable(name="valueBias2", shape=(1, 1), dtype=tf.float32,
                                              initializer=tf.zeros_initializer)

            # shape = (?, 1)
            self.valueActivation2 = tf.matmul(self.valueActivation1, self.valueWeights2, name="valueActivation2")

            self.value_estimate = tf.squeeze(self.valueActivation2)
            self.valueLoss = tf.squared_difference(self.value_estimate, self.valueTarget)

            self.loss = policyLossFactor * self.policyLoss + valueLossFactor * self.valueLoss

    def playerBoardCorrector(self, state):
        player = np.expand_dims(state[0], axis=0)

        if player[0][0] == 1:
            board = np.expand_dims(state[1], axis=0)
        else:
            board = np.expand_dims(invertBoard(state[1]), axis=0)

        return player, board

    def predictPolicy(self, env, sess=None):
        sess = sess or tf.get_default_session()

        player, board = self.playerBoardCorrector(env.state)

        action, mu = sess.run([self.action, self.mu], {self.board: board, self.validColumnsFilter: np.expand_dims(env.getAvaliableColumns(), axis=0)})
        return action, mu

    def predictValue(self, state, sess=None):
        sess = sess or tf.get_default_session()

        player, board = self.playerBoardCorrector(state)

        return sess.run(self.value_estimate, {self.board: board})

    def getLoss(self, env, sess=None):
        sess = sess or tf.get_default_session()

        player, board = self.playerBoardCorrector(env)

        loss = sess.run(self.loss, {self.board: board, self.validColumnsFilter: np.expand_dims(env.getAvaliableColumns(), axis=0)})

        return loss

    def assignNewMeanAndStdDev(self, mean, sigma):
