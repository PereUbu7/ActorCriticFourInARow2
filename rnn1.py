import gym
import itertools
import matplotlib
import numpy as np
import sys
import tensorflow as tf
import collections

from time import time

import os.path
from fourInARowWrapper import FourInARowWrapper

if "../" not in sys.path:
  sys.path.append("../")
from lib import plotting


matplotlib.style.use('ggplot')

env = FourInARowWrapper(1)

def invertBoard(inBoard):
    invertedBoard = np.array(inBoard)

    board_shape = inBoard.shape

    #print("Shape:", board_shape)

    for x in range(board_shape[0]):
        for y in range(board_shape[1]):
            invertedBoard[x][y][0] = inBoard[x][y][1]
            invertedBoard[x][y][1] = inBoard[x][y][0]

    return invertedBoard

class CommonNetwork():
    def __init__(self, scope="common_net"):
        with tf.variable_scope(scope):
            self.board = tf.placeholder(tf.float32, (None, 7, 6, 2), "board")

            self.board_norm = tf.nn.batch_normalization(x=self.board, mean=0, variance=1, offset=1, scale=1, variance_epsilon=1e-7)
            self.board_flat = tf.reshape(self.board_norm, (-1, 7*6*2))

            self.board_and_out = tf.contrib.layers.fully_connected(
                inputs=self.board_flat,
                num_outputs=800,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1),
                scope="layer_1"
            )

            self.board_and_out_relu = tf.nn.leaky_relu(features=self.board_and_out, alpha=0.1)

            self.outLayer_pre = tf.contrib.layers.fully_connected(
                inputs=self.board_and_out_relu,
                num_outputs=500,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1),
                scope="outLayer"
            )

            self.outLayer_pre_relu = tf.nn.leaky_relu(features=self.outLayer_pre, alpha=0.1)

            self.outLayer = tf.contrib.layers.dropout(
                self.outLayer_pre_relu,
                keep_prob=0.9,
            )


class Trainer():
    def __init__(self, scope="trainer", learning_rate=0.001, commonNet=None, policy=None, policyLossFactor=0.1, value=None, valueLossFactor=0.1):
        with tf.variable_scope(scope):
            self.policy = policy
            self.value = value
            self.commonNet = commonNet
            self.policyLoss = policy.loss
            self.valueLoss = value.loss
            self.loss = policyLossFactor * policy.loss + valueLossFactor * value.loss

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            self.optimizerRMSProp = tf.train.RMSPropOptimizer(learning_rate=learning_rate)
            self.train_op = self.optimizer.minimize(
                self.loss, global_step=tf.contrib.framework.get_global_step())

    def update(self, board, td_target, td_error, action, avaliableColumns, sess=None):
        sess = sess or tf.get_default_session()

        feed_dict = {self.policy.target: td_error, self.policy.action: action,
                    self.policy.validColumnsFilter: avaliableColumns,
                     self.value.target: td_target, self.commonNet.board: board}
        _, loss, pol_loss, value_loss = sess.run([self.train_op, self.loss, self.policyLoss, self.valueLoss], feed_dict)
        return loss, pol_loss, value_loss

    def evalFilters(self, board, sess=None):
        sess = sess or tf.get_default_session()

        board_exp = np.expand_dims(board, axis=0)

        feed_dict = {self.commonNet.board: board_exp}
        layer1, layer2 = sess.run([self.commonNet.deconv1, self.commonNet.deconv2_1], feed_dict)

        return layer1, layer2


class PolicyEstimator():
    """
    Policy Function approximator.
    """

    def __init__(self, scope="policy_estimator", entropyFactor=0.1, shared_layers=None):
        with tf.variable_scope(scope):
            self.shared_layers = shared_layers
            if shared_layers is not None:
                # self.board = shared_layers.board
                # self.input = shared_layers.outLayer
                #self.player = shared_layers.player
                pass
            else:
                print("Needs shared_layers parameter")
                return -1

            self.target = tf.placeholder(dtype=tf.float32, name="target")
            self.validColumnsFilter = tf.placeholder(dtype=tf.float32, shape=(None, 7), name="validColumnsFilter")

            self.l1 = tf.contrib.layers.fully_connected(
                inputs=shared_layers.outLayer,
                num_outputs=100,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1),
                scope="l2"
            )

            self.l1 = tf.nn.leaky_relu(features=self.l1, alpha=0.1)

            # self.l1_dropout = tf.contrib.layers.dropout(
            #     self.l1,
            #     keep_prob=0.9,
            # )

            self.mu = tf.contrib.layers.fully_connected(
                inputs=self.l1,
                num_outputs=env.action_space.high-env.action_space.low,
                activation_fn=tf.nn.sigmoid,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1),
                scope="mu")

            self.mu = tf.squeeze(self.mu)

            self.mu = tf.multiply(self.mu, self.validColumnsFilter) + 1e-6

            self.mu = tf.divide(self.mu, tf.reduce_sum(self.mu))

            self.dist = tf.contrib.distributions.Categorical(probs=self.mu, dtype=tf.float32)

            # Draw sample
            self.action = self.dist.sample()

            # Loss and train op
            self.loss = -self.dist.log_prob(self.action) * self.target

            # Add cross entropy cost to encourage exploration
            self.loss -= entropyFactor * self.dist.entropy()


    def predict(self, env, sess=None):
        sess = sess or tf.get_default_session()

        player = np.expand_dims(env.state[0], axis=0)

        if player[0][0] == 1:
            board = np.expand_dims(env.state[1], axis=0)
        else:
            board = np.expand_dims(invertBoard(env.state[1]), axis=0)

        action, mu = sess.run([self.action, self.mu], {self.shared_layers.board: board, self.validColumnsFilter: np.expand_dims(env.getAvaliableColumns(), axis=0)})
        return action, mu


class ValueEstimator():
    """
    Value Function approximator.
    """

    def __init__(self, scope="value_estimator", shared_layers=None):
        with tf.variable_scope(scope):
            self.shared_layers = shared_layers
            if shared_layers is not None:
                # self.board = shared_layers.board
                # self.input = shared_layers.outLayer
                pass
            else:
                print("Needs shared_layers parameter")
                return -1

            #self.player = tf.placeholder(tf.float32, (None, 2), "player")
            self.target = tf.placeholder(dtype=tf.float32, shape=(None, 1), name="target")

            self.l1 = tf.contrib.layers.fully_connected(
                inputs=shared_layers.outLayer,
                num_outputs=100,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1),
                scope="l2"
            )

            self.l1 = tf.nn.leaky_relu(features=self.l1, alpha=0.1)

            # self.l1_dropout = tf.contrib.layers.dropout(
            #     self.l1,
            #     keep_prob=0.7,
            # )

            # This is just linear classifier
            self.output_layer = tf.contrib.layers.fully_connected(
                inputs=self.l1,
                num_outputs=1,
                activation_fn=None,
                weights_initializer=tf.random_normal_initializer(mean=0.0, stddev=1),
                scope="output_layer")

            self.value_estimate = tf.squeeze(self.output_layer)
            self.loss = tf.squared_difference(self.value_estimate, self.target)


    def predict(self, state, sess=None):
        sess = sess or tf.get_default_session()
        player = np.expand_dims(state[0], axis=0)

        if player[0][0] == 1:
            board = np.expand_dims(state[1], axis=0)
        else:
            board = np.expand_dims(invertBoard(state[1]), axis=0)
        #state = featurize_state(state)
        return sess.run(self.value_estimate, {self.shared_layers.board: board})


def actor_critic(env, estimator_policy_X, estimator_value_X, trainer_X, num_episodes, discount_factor=1.0, player2=True, positiveRewardFactor=1.0, negativeRewardFactor=1.0, batch_size=1):
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

    for i_episode in range(num_episodes):
        # Reset the environment and pick the first action
        state = env.reset(i_episode % 2 + 1)
        robotLevel = i_episode%2 + 3
        #robotLevel = 4

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

        # if game == num_episodes-3:
        #     player2 = False

        # One step in the environment
        for t in itertools.count():
            # Save avaliable columns
            if not done:
                avaliableColumns = env.getAvaliableColumns()

            currentPlayerBeforeStep = env.getCurrentPlayer()

            action_tmp = action

            # Take a step
            if currentPlayerBeforeStep == 1 and not done or currentPlayerBeforeStep == 2 and player2 and not done:
                action, probas = estimator_policy_X.predict(env)
                action = action[0]
                probas = probas[0]
            elif not done:
                try:
                    action = int(input("Give a column number: ")) - 1
                except ValueError:
                    print("Wrong input! Setting action to 1")
                    action = 0
                probas = None

            if currentPlayerBeforeStep == 2 and player2 and not done:
                next_state, reward, step_done, action = env.robotStep(robotLevel)
            elif not done:
                next_state, reward, step_done, _ = env.step(action)

            if not done:

                if game == num_episodes-3:
                    pass
                    #layer1, layer2 = trainer_X.evalFilters(next_state[1])
                    #plotting.plotNNFilter(next_state[1], layer1, layer2)


                if step_done:
                    pass

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
                episode.append(Transition(
                    state=state_tmp, action=action_tmp, reward=reward_tmp, next_state=next_state, done=done))

                player = None
                if episode[-1].state[0][0] == 1:
                    player = "X"
                elif episode[-1].state[0][1] == 1:
                    player = "O"
                # Update statistics
                stats.episode_lengths[i_episode] = t

                # If player 0 (X)
                if episode[-1].state[0][0] == 1 or True:

                    if episode[-1].state[0][0] == 1:
                        stats.episode_rewards[i_episode] += episode[-1].reward
                    # Calculate TD Target
                    value_next = estimator_value_X.predict(episode[-1].next_state)
                    td_target = episode[-1].reward + discount_factor * value_next
                    td_error = td_target - estimator_value_X.predict(episode[-1].state)

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
                # else:
                #     value_next = estimator_value_O.predict(episode[-1].next_state, )
                #     td_target = episode[-1].reward + discount_factor * value_next
                #     td_error = td_target - estimator_value_O.predict(episode[-1].state)
                #
                #     batch_player_O[batch_pos_O] = episode[-1].state[0]
                #     batch_board_O[batch_pos_O] = episode[-1].state[1]
                #     batch_td_target_O[batch_pos_O] = td_target
                #     batch_td_error_O[batch_pos_O] = td_error
                #     batch_action_O[batch_pos_O] = episode[-1].action
                #     batch_avaliableColumns_O[batch_pos_O] = avaliableColumns
                #
                #     batch_pos_O += 1


                stats.episode_td_error[i_episode] += td_error

                if batch_pos_X == batch_size:
                    # Update both networks
                    loss_X, policyLoss, valueLoss = trainer_X.update(batch_board_X, batch_td_target_X, batch_td_error_X, batch_action_X, batch_avaliableColumns_X)
                    loss_X = loss_X[0][0]
                    policyLoss = policyLoss[0][0]
                    valueLoss = valueLoss[0][0]
                    batch_pos_X = 0

                    print("Updates X network. Loss:", loss_X)
                    stats.episode_value_loss[i_episode] += valueLoss

                # if batch_pos_O == batch_size:
                #     # Update both networks
                #     loss_O = trainer_O.update(batch_board_O, batch_td_target_O, batch_td_error_O, batch_action_O,
                #                               batch_avaliableColumns_O)
                #     loss_O = loss_O[0][0]
                #     batch_pos_O = 0
                #
                #     print("Updates X network. Loss:", loss_O)
                #     stats.episode_value_loss[i_episode] += loss_O

                    if probas is not None and last_probas is not None:
                        kl_div = 0
                        for i in range(probas.size):
                            kl_div += probas[i]*np.log(probas[i]/last_probas[i])
                        stats.episode_kl_divergence[i_episode] += kl_div

                # Print out which step we're on, useful for debugging.
                print(
                    "\rPlayer {}: Action {}, Reward {:<4}, TD Error {:<20}, TD Target {:<20}, Value Next {:<20}, at Step {:<5} @ Game {} @ Episode {}/{} ({})".format(
                        player, int(episode[-1].action + 1), episode[-1].reward, td_error, td_target, value_next, t,
                        game, i_episode + 1, num_episodes, stats.episode_rewards[i_episode - 1]), end="")

                if player == "X" and episode[-1].reward > 0 and robotLevel > 1:# or i_episode % 100 == 0:
                    for i in range(t):
                        print("Player:", batch_player_X[batch_pos_X-t+i], "Action:", int(batch_action_X[batch_pos_X-t+i])+1 )
                    print("Robot level:", robotLevel)
                    env.renderHotEncodedState( ((1, 0), batch_board_X[batch_pos_X-1]) )

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


            if step_done:
                done = True

            state = next_state

    return stats

tf.reset_default_graph()

start = time()

batch_size = 2000

global_step = tf.Variable(0, name="global_step", trainable=False)
common_net_X = CommonNetwork("X_convNet")
policy_estimator_X = PolicyEstimator("X_policy", entropyFactor=1e-5, shared_layers=common_net_X)
value_estimator_X = ValueEstimator("X_value", shared_layers=common_net_X)
trainer_X = Trainer("X_trainer", learning_rate=1e-2, convNet=common_net_X, policy=policy_estimator_X, policyLossFactor=1, value=value_estimator_X, valueLossFactor=1e-2)

variables = tf.contrib.slim.get_variables_to_restore()
variables_to_restore = [v for v in variables if v.name.split('/')[0]!='trainer' and v.name.split('/')[0]!='policy_estimator' and v.name.split('/')[0]!='value_estimator']
variables_to_init = [v for v in variables if v.name.split('/')[0]!='conv_net']

saver = tf.train.Saver(variables)

with tf.Session() as sess:
    try:
        saver.restore(sess, "tmp/model10_t.ckpt")
        sess.run(tf.initializers.variables(variables))
        print("Restoring parameters")
        for v in variables:
            print(v)
    except ValueError:
        sess.run(tf.initializers.global_variables())
        print("Initializing parameters")

    stats = actor_critic(env, policy_estimator_X, value_estimator_X, trainer_X, 1000, discount_factor=0.99, player2=True, positiveRewardFactor=1, negativeRewardFactor=1, batch_size=batch_size)

    #filters = sess.run(conv_net_X.filter1)

    save_path = saver.save(sess, "tmp/model10_t.ckpt")
    print("Saving parameters")
    for v in variables:
        print(v)

end = time()

print("It took:", end-start, "seconds to do 5.000 games")

plotting.plot_episode_stats(stats, smoothing_window=100)

