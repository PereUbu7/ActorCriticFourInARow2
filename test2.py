import gym
import itertools
import matplotlib as plt
import numpy as np
import sys
import tensorflow as tf
import collections
import pickle

from random import shuffle
from time import time

import os.path
from fourInARowWrapper import FourInARowWrapper

if "../" not in sys.path:
  sys.path.append("../")
from lib import plotting


plt.style.use('ggplot')

env = FourInARowWrapper(1)
NUM_STEPS = 10

def invertBoard(inBoard):
    invertedBoard = np.array(inBoard)

    board_shape = inBoard.shape

    #print("Shape:", board_shape)

    for x in range(board_shape[0]):
        for y in range(board_shape[1]):
            invertedBoard[x][y][0] = inBoard[x][y][1]
            invertedBoard[x][y][1] = inBoard[x][y][0]

    return invertedBoard

def printClockTimeSince(argString, timeSeconds):
    durationAllSeconds = time() - timeSeconds

    durationHours = int(durationAllSeconds / 60)
    durationSeconds = int(durationAllSeconds % 60)

    print(argString, "{:02d}:{:02d}".format(durationHours, durationSeconds))


class BatchGenerator(object):
    def __init__(self, data, num_steps, batch_size):
        self.data = data
        self.num_steps = num_steps
        self.batch_size = batch_size

    def generate(self):
        x = np.zeros((self.batch_size, self.num_steps))
        y = np.zeros((self.batch_size, self.num_steps, 7))

        while True:
            yield x, y

class RNN():
    def __init__(self):
        self.validColumnsFilter = np.zeros(7)
        self.boardSequence = np.zeros((1, NUM_STEPS, 84))
        self.rewardSequence = np.zeros((1, 1))
        self.muSequence = np.zeros((1, 7))
        self.actionSequence = np.zeros((1, 7))
        self.model = tf.keras.Sequential(
            [
                tf.keras.layers.LSTM(20, input_shape=(NUM_STEPS, 84), return_sequences='false'),
                tf.keras.layers.Dense(7, activation='softmax'),
                tf.keras.layers.Lambda(lambda x: x[:, -1])
            ]
        )

        print(self.model.summary())

        self.model.compile(loss=self.rlLoss, optimizer='rmsprop')

    def rlLoss(self, y, p):
        return -tf.math.multiply((tf.math.multiply(y, tf.math.log(p)) + tf.math.multiply((1 - y), tf.math.log(1 - p))), tf.square(y))

    def predictAction(self, env):
        player = np.expand_dims(env.state[0], axis=0)

        if player[0][0] == 1:
            board = np.expand_dims(np.reshape(env.state[1], 84), axis=0)
        else:
            board = np.expand_dims(np.reshape(invertBoard(env.state[1]), 84), axis=0)

        self.boardSequence[-1] = np.roll(self.boardSequence[-1], -1, axis=0)
        self.boardSequence[-1][-1] = board

        self.validColumnsFilter = np.expand_dims(env.getAvaliableColumns(), axis=0)
        print("Kör modell")
        mu = self.model(np.expand_dims(self.boardSequence[-1], axis=0))

        print("model.predict")
        mu = self.model.predict(np.expand_dims(self.boardSequence[-1], axis=0), batch_size=1)

        print(mu)

        print("Anpassa till tillgängliga kolumner")
        mu = np.multiply(mu, env.getAvaliableColumns()) + 1e-6
        print("Beräkna mu")
        #muValue = tf.keras.backend.eval(mu)[0]
        muValue = mu[0]
        print(muValue)
        self.muSequence[-1] = muValue

        print("RNN Mu shape:", mu.shape)
        print("Mu:", muValue)
        print("Pick action")
        action = tf.random.categorical(mu, 1)
        print("Eval action")
        actionValue = tf.keras.backend.eval(action)[0]
        print("RNN action value:", actionValue)

        actionOneHot = np.zeros((1, 7))
        actionOneHot[0][actionValue[0]] = 1
        self.actionSequence[-1] = actionOneHot

        print("RNN action:", actionValue)
        print("RNN history shape:", self.boardSequence.shape)
        return actionValue, muValue

    def update(self):
        print("update")
        print("boards")
        print(self.boardSequence.shape)
        print(self.boardSequence)
        print("rewards")
        print(self.rewardSequence.shape)
        print(self.rewardSequence)
        print("mus")
        print(self.muSequence.shape)
        print(self.muSequence)
        print("actions")
        print(self.actionSequence.shape)
        print(self.actionSequence)
        # skapa array som har 'reward' på index action och -'reward' på de andra
        target = -np.multiply(self.rewardSequence[-1], self.actionSequence[-1])
        print("targets")
        print(target.shape)
        print(target)
        self.model.fit(np.expand_dims(self.boardSequence[-1], axis=0), np.expand_dims(target, axis=0), epochs=2)

    def resetGame(self):
        #self.boardSequence = np.append(self.boardSequence, np.zeros((1, NUM_STEPS, 84)), axis=0)
        self.boardSequence = np.zeros((1, NUM_STEPS, 84))
        #self.rewardSequence = np.append(self.rewardSequence, np.zeros((1, 1)), axis=0)

        #self.actionSequence = np.append(self.actionSequence, np.zeros((1, 7)), axis=0)
        #self.muSequence = np.append(self.muSequence, np.zeros((1, 7)), axis=0)

    def finishGame(self, reward):
        self.rewardSequence[-1][0] = reward


class ANN():
    def __init__(self, learningRate=0.001, policyLossFactor=0.1, valueLossFactor=0.1, entropyFactor=0.1, dropout_keep_prob=1.0, TDRegularizingFactor=None, scope="ANN", global_step=None):
        with tf.compat.v1.variable_scope(scope):

            self.board = tf.compat.v1.placeholder(tf.float32, (None, 7, 6, 2), "board")
            self.policyTarget = tf.compat.v1.placeholder(dtype=tf.float32, name="policyTarget")
            self.valueTarget = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 1), name="valueTarget")
            self.validColumnsFilter = tf.compat.v1.placeholder(dtype=tf.float32, shape=(None, 7), name="validColumnsFilter")

            self.convFilter1 = tf.compat.v1.get_variable(name="convFilter1", shape=(4, 4, 2, 20), dtype=tf.float32, initializer=tf.random_normal_initializer)

            self.board_norm = tf.nn.batch_normalization(x=self.board, mean=0, variance=1, offset=None, scale=None, variance_epsilon=1e-7)
            self.board_flat = tf.reshape(self.board_norm, (-1, 7*6*2))

            self.convLayer1 = tf.compat.v1.nn.conv2d(input=self.board_norm,
                                           filter=self.convFilter1,
                                           strides=[1, 1, 1, 1],
                                           padding="SAME",
                                           data_format="NHWC",
                                           dilations=[1, 1, 1, 1],
                                           name="convLayer1")

            self.flat = tf.concat([tf.reshape(self.convLayer1, (-1, 7*6*20)), self.board_flat], 1)

            self.board_and_out = tf.compat.v1.layers.dense(
                inputs=self.flat,
                units=1500,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1)
            )

            self.board_and_out_relu = tf.nn.leaky_relu(features=self.board_and_out, alpha=0.1)

            self.l2 = tf.compat.v1.layers.dense(
                inputs=self.board_and_out_relu,
                units=1000,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1)
            )

            self.l2_relu = tf.nn.leaky_relu(features=self.l2, alpha=0.1)

            self.outLayer_pre = tf.compat.v1.layers.dense(
                inputs=self.l2_relu,
                units=800,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1)
            )

            self.outLayer = tf.nn.leaky_relu(features=self.outLayer_pre, alpha=0.1)

            # Shared layers end

            # Policy layer ahead

            self.policyl1 = tf.compat.v1.layers.dense(
                inputs=self.outLayer,
                units=500,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1)
            )

            self.policyl1_relu = tf.nn.leaky_relu(features=self.policyl1, alpha=0.1)

            self.policyl2 = tf.compat.v1.layers.dense(
                inputs=self.policyl1_relu,
                units=100,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1)
            )

            self.policyl2_relu = tf.nn.leaky_relu(features=self.policyl2, alpha=0.1)

            self.policyMu = tf.compat.v1.layers.dense(
                inputs=self.policyl2_relu,
                units=env.action_space.high - env.action_space.low,
                activation=tf.nn.sigmoid,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1))

            self.policyMu = tf.squeeze(self.policyMu)

            self.policyMu = tf.multiply(self.policyMu, self.validColumnsFilter) + 1e-6

            self.policyMu = tf.divide(self.policyMu, tf.reduce_sum(self.policyMu))

            self.policyDist = tf.compat.v1.distributions.Categorical(probs=self.policyMu, dtype=tf.float32)

            # Draw sample
            self.policyAction = self.policyDist.sample()

            #
            if TDRegularizingFactor is None:
                self.policyTDRegularizingLambda = tf.math.rsqrt(tf.reduce_mean(tf.square(self.policyTarget)))
            else:
                self.policyTDRegularizingLambda = tf.convert_to_tensor(TDRegularizingFactor)

            # Loss with TD-regularizing factor
            self.policyLoss = -self.policyDist.log_prob(self.policyAction) * (
                        self.policyTarget - self.policyTDRegularizingLambda * tf.square(self.policyTarget))

            # Add cross entropy cost to encourage exploration
            self.policyLoss -= entropyFactor * self.policyDist.entropy()

            self.policyLoss = tf.reduce_mean(self.policyLoss)

            # Policy layers end

            # Value (critic) layers ahead
            self.valuel1 = tf.compat.v1.layers.dense(
                inputs=self.outLayer,
                units=500,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1)
            )

            self.valuel1_relu = tf.compat.v1.nn.leaky_relu(features=self.valuel1, alpha=0.1)

            self.valuel1_dropout = tf.compat.v1.nn.dropout(self.valuel1_relu, keep_prob=dropout_keep_prob)

            self.valuel2 = tf.compat.v1.layers.dense(
                inputs=self.valuel1_dropout,
                units=100,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=1)
            )

            self.valuel2_relu = tf.nn.leaky_relu(features=self.valuel2, alpha=0.1)

            self.valuel2_dropout = tf.compat.v1.nn.dropout(self.valuel2_relu, keep_prob=dropout_keep_prob)

            self.valueOutput_layer = tf.compat.v1.layers.dense(
                inputs=self.valuel2_dropout,
                units=1,
                activation=None,
                kernel_initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01))

            self.value_estimate = tf.squeeze(self.valueOutput_layer)
            self.valueLoss = tf.reduce_mean(tf.compat.v1.squared_difference(self.value_estimate, self.valueTarget))

            # Summing it up
            self.loss = policyLossFactor * self.policyLoss + valueLossFactor * self.valueLoss

            self.optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learningRate)
            self.optimizerRMSProp = tf.compat.v1.train.RMSPropOptimizer(learning_rate=learningRate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=global_step)

    def update(self, board, td_target, td_error, action, avaliableColumns, sess=None):
        sess = sess or tf.compat.v1.get_default_session()

        feed_dict = {self.policyTarget: td_error, self.policyAction: action,
                             self.validColumnsFilter: avaliableColumns,
                             self.valueTarget: td_target, self.board: board}
        _, loss, pol_loss, value_loss, mean_squared_target = sess.run(
                    [self.train_op, self.loss, self.policyLoss, self.valueLoss, self.policyTDRegularizingLambda],
                    feed_dict)
        return loss, pol_loss, value_loss, mean_squared_target

    def predictAction(self, env, sess=None):
        sess = sess or tf.compat.v1.get_default_session()

        player = np.expand_dims(env.state[0], axis=0)

        if player[0][0] == 1:
            board = np.expand_dims(env.state[1], axis=0)
        else:
            board = np.expand_dims(invertBoard(env.state[1]), axis=0)

        action, mu = sess.run([self.policyAction, self.policyMu], {self.board: board,
                                                               self.validColumnsFilter: np.expand_dims(
                                                                   env.getAvaliableColumns(), axis=0)})
        return action, mu

    def predictValue(self, state, sess=None):
        sess = sess or tf.compat.v1.get_default_session()
        player = np.expand_dims(state[0], axis=0)

        if player[0][0] == 1:
            board = np.expand_dims(state[1], axis=0)
        else:
            board = np.expand_dims(invertBoard(state[1]), axis=0)

        return sess.run(self.value_estimate, {self.board: board})


def actor_critic(env, model1, model2, num_episodes, discount_factor=1.0, player2=True, positiveRewardFactor=1.0, negativeRewardFactor=1.0, batch_size=1, numberOfEpochsPerUpdate=1):
    """
    Actor Critic Algorithm. Optimizes the policy
    function approximator using policy gradient.

    Args:
        env: OpenAI environment.
        estimator_policy_X: Policy Function to be optimized
        estimator_value_X: Value function approximator, used as a critic
        trainer_X: our training class
        num_episodes: Number of episodes to run for
        discount_factor: Time-discount factor
        player2: True if computer plays player2, False if user does
        positiveRewardFactor: Factor bla bla bla reward
        negativeRewardFactor: Factor bla bla bla
        batch_size: Batch size

    Returns:
        An EpisodeStats object with two numpy arrays for episode_lengths and episode_rewards.
    """

    # Keeps track of useful statistics
    stats = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        episode_td_error=np.zeros(num_episodes),
        episode_value_loss=np.zeros(num_episodes),
        episode_policy_loss=np.zeros(num_episodes),
        episode_kl_divergence=np.zeros(num_episodes))

    Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

    batch_board_X = np.zeros((batch_size, 7, 6, 2))
    batch_player_X = np.zeros((batch_size, 2))
    batch_td_target_X = np.zeros((batch_size, 1))
    batch_td_error_X =np.zeros((batch_size, 1))
    batch_action_X =np.zeros((batch_size, 1))
    batch_avaliableColumns_X = np.zeros((batch_size, 7))


    batch_pos_X = 0

    game = 1

    startTime = time()

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset(i_episode % 2 + 1)
        #robotLevel = i_episode%2 + 3
        robotLevel = 2

        episode = []

        probas = None
        last_turn = False
        done = False
        last_state = None
        action = None
        reward = None

        # if game % 5000 == 10:
        #     player2 = True
        # elif game % 5000 == 0:
        #     player2 = False

        # if game >= num_episodes-3:
        #     player2 = False

        # One step in the environment
        for t in itertools.count():
            # Save avaliable columns
            if not done:
                avaliableColumns = env.getAvaliableColumns()

            currentPlayerBeforeStep = env.getCurrentPlayer()

            action_tmp = action

            # Model 1 always predict a step, even though it's not its turn - to be able to learn from the result
            if currentPlayerBeforeStep == 1 and not done:
                print("X ska bestämma")
                action, probas = model1.predictAction(env)
                print("X har bestämt")
                action = action[0]
                probas = probas[0]
            # elif not done:
            #     try:
            #         action = int(input("Give a column number: ")) - 1
            #     except ValueError:
            #         print("Wrong input! Setting action to 1")
            #         action = 0
            #     probas = None

            # If it player 2's turn, overwrite player1's step
            if currentPlayerBeforeStep == 2 and player2 and not done:
                #next_state, reward, step_done, action = env.robotStep(robotLevel)
                print("O ska bestämma")
                action, probas2 = model2.predictAction(env)
                print("O har bestämt")

            if not done:
                next_state, reward, step_done, _ = env.step(action)

                env.renderHotEncodedState(env.getHotEncodedState2d())

                if t > 0:
                    state_tmp = last_state
                    last_state = state
                    reward_tmp = -reward*negativeRewardFactor
                else:
                    state_tmp = state
                    last_state = state
                    reward_tmp = -reward*negativeRewardFactor


            elif done and not last_turn:
                state_tmp = episode[-2].next_state
                reward_tmp = reward*positiveRewardFactor
            else:
                break




            if t > 0:
                print("Ska spara ")
                episode.append(Transition(
                    state=state_tmp, action=action_tmp, reward=reward_tmp, next_state=next_state, done=done))
                print("Har sparat")

                player = None
                if episode[-1].state[0][0] == 1:
                    player = "X"
                elif episode[-1].state[0][1] == 1:
                    player = "O"
                # Update statistics
                stats.episode_lengths[i_episode] = t

                # If player 0 (X)
                # or True alternates between on-policy learning and off-policy learning
                if episode[-1].state[0][0] == 1:

                    if episode[-1].state[0][0] == 1:
                        stats.episode_rewards[i_episode] += episode[-1].reward
                    # Calculate TD Target (or advantage function)
                    value_next = model1.predictValue(episode[-1].next_state)

                    # Q value
                    td_target = episode[-1].reward + discount_factor * value_next
                    td_error = td_target - model1.predictValue(episode[-1].state)

                    if episode[-1].state[0][0] == 1:
                        batch_board_X[batch_pos_X] = episode[-1].state[1]
                    else:
                        batch_board_X[batch_pos_X] = invertBoard(episode[-1].state[1])
                    batch_player_X[batch_pos_X] = episode[-1].state[0]
                    batch_td_target_X[batch_pos_X] = td_target
                    batch_td_error_X[batch_pos_X] = td_error
                    batch_action_X[batch_pos_X] = episode[-1].action
                    batch_avaliableColumns_X[batch_pos_X] = avaliableColumns

                    batch_pos_X += 1

                stats.episode_td_error[i_episode] += td_error

                if batch_pos_X == batch_size:
                    # Update both networks
                    print("\n")
                    print("Uppdaterar X")
                    for n in range(numberOfEpochsPerUpdate):
                        loss_X, policyLoss, valueLoss, meanSquaredTDError = model1.update(batch_board_X, batch_td_target_X, batch_td_error_X, batch_action_X, batch_avaliableColumns_X)
                        print("\rUpdates network ({}). Value loss: {}. Policy loss: {}. Mean squared TD Error: {}".format(
                            n+1, valueLoss, policyLoss, meanSquaredTDError), end=" ")
                    print("X uppdaterat")
                    batch_pos_X = 0

                    print("Ska spara experience replay")
                    with open(experienceReplayFile, 'ab') as f:
                        pickle.dump( (batch_board_X, batch_td_target_X, batch_td_error_X, batch_action_X, batch_avaliableColumns_X) , f)
                    print("Sparat")

                    stats.episode_value_loss[i_episode] += valueLoss



                    if probas is not None and last_probas is not None:
                        kl_div = 0
                        for i in range(probas.size):
                            kl_div += probas[i]*np.log(probas[i]/last_probas[i])
                        stats.episode_kl_divergence[i_episode] += kl_div

                # Print out which step we're on, useful for debugging.
                print("Ska skriva player")
                print(
                    "\rPlayer {}: Action {}, Reward {:<4}, TD Error {:<20}, TD Target {:<20}, Value Next {:<20}, at Step {:<5} @ Episode {}/{} ({}). Batch step: {}".format(
                        player, int(episode[-1].action + 1), episode[-1].reward, td_error, td_target, value_next, t,
                        i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1], batch_pos_X), end="")
                print("Player skrivet")
                # if player == "X" and episode[-1].reward > 0 and robotLevel > 1:# or i_episode % 100 == 0:
                #     for i in range(t+1):
                #         print("Player:", batch_player_X[batch_pos_X-t+i-1], "Action:", int(batch_action_X[batch_pos_X-t+i-1])+1 )
                #     print("Robot level:", robotLevel)
                #     env.renderHotEncodedState( ((1, 0), batch_board_X[batch_pos_X-1]) )

            if game == num_episodes or env.getCurrentPlayer() == 2 and not player2:
                env.render()
                if probas is not None:
                    out = " "
                    for i in range(probas.size):
                        out += "%03d " % int(probas[i]*100+0.5)
                    print(out)

            last_probas = probas

            if done:
                last_turn = True
                game += 1

                print("Ska uppdatera O")
                # if winner is 'O' reward is 1 else -1
                model2.finishGame(1 if episode[-1].state[0][0] == 1 else -1)

                model2.update()
                print("O uppdaterat")

                model2.resetGame()

            if step_done:
                done = True

            state = next_state

    return stats

def runExperienceReplay(fileName, trainer, numberOfReplays, batchSize):

    numberOfAvailableReplays = 0

    stats = (np.zeros(numberOfReplays), np.zeros(numberOfReplays))
    return stats

    with open(fileName, 'rb') as f:
        try:
            data = list(pickle.load(f))

            while True:
                tempData = pickle.load(f)
                if len(data) == len(tempData):
                    for n in range(len(data)):
                        data[n] = np.append(data[n], tempData[n], axis=0)
                else:
                    raise RuntimeError("data error - sizes don't match")

                numberOfAvailableReplays = np.shape(data[0])[0]

        except EOFError:
            print("Number of available replays:", numberOfAvailableReplays)
        except RuntimeError:
            print("Quitting experience replay...")
            quit(-1)

    data = tuple(data)

    print("Running", numberOfReplays, "experience replay updates")

    for n in range(numberOfReplays):
        availableShuffledIndices = [*range(numberOfAvailableReplays)]
        shuffle(availableShuffledIndices)

        shuffledIndices = availableShuffledIndices[:batchSize]

        batch_board, batch_td_target, batch_td_error, batch_action, batch_avaliableColumns = data

        loss, policyLoss, valueLoss, meanSquareTDError = trainer.update(batch_board[shuffledIndices],
                                                     batch_td_target[shuffledIndices],
                                                     batch_td_error[shuffledIndices],
                                                     batch_action[shuffledIndices],
                                                     batch_avaliableColumns[shuffledIndices])

        # print(policyLoss)
        # print(valueLoss)

        stats[0][n] = policyLoss
        stats[1][n] = valueLoss

        if True:
            print("\rUpdating {} of {}. Batch with size {} is chosen. Policy loss: {}. Value loss: {}. Total loss: {}. Mean Square TD Error: {}".format(n, numberOfReplays, batchSize, policyLoss, valueLoss, loss, meanSquareTDError), end="")

    # print("\n")
    # print(shuffledIndices)
    # l = batch_action[shuffledIndices]
    # print(l)
    # print(l.shape)

    return stats



tf.python.framework.ops.reset_default_graph()

start = time()

experienceReplayFile = 'experienceReplayANN_1.pickle'
modelFile = "tmp/model_ANN_4.ckpt"

batch_size = 20
numberOfGames = 100

experienceReplayBatchSize = 100
experienceReplayNumberOfUpdates = 0

global_step = tf.Variable(0, name="global_step", trainable=False)
model1 = ANN(scope="ANN", learningRate=5e-4, entropyFactor=1e-5, dropout_keep_prob=0.99,
            TDRegularizingFactor=0.9, policyLossFactor=1e-0, valueLossFactor=1e-1, global_step=global_step)

model2 = RNN()

saver = tf.compat.v1.train.Saver(tf.compat.v1.all_variables())

with tf.compat.v1.Session() as sess:
    try:
        saver.restore(sess, modelFile)
        print("Restoring parameters")
    except ValueError:
        sess.run(tf.compat.v1.initializers.global_variables())
        print("Initializing parameters")

    stats = actor_critic(env, model1, model2, numberOfGames, discount_factor=0.75, player2=True, positiveRewardFactor=5,
                         negativeRewardFactor=1, batch_size=batch_size, numberOfEpochsPerUpdate=25)

    experienceReplayStats = runExperienceReplay(experienceReplayFile, model1, experienceReplayNumberOfUpdates, experienceReplayBatchSize)

    save_path = saver.save(sess, modelFile)
    print("Saving parameters")

end = time()

print("It took:", end-start, "seconds to do", numberOfGames, "games")

plotting.plot_episode_stats(stats, experienceReplayStats, smoothing_window=100)