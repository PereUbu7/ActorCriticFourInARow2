import unittest
import tensorflow as tf
from fourInARowWrapper import FourInARowWrapper
from test2_ANN import ANN

class TestANNMethods(unittest.TestCase):

    def setUp(self):
        self.env = FourInARowWrapper(1)
        self.ann = ANN(scope="test", env=self.env)

    def tearDown(self):
        pass

    def testANNInit(self):
        self.assertTrue(True)

    def testPlayerBoardCorrectorPlayer(self):
        player, board = self.ann.playerBoardCorrector(self.env)

        self.assertTrue(self.env.state[0][0] == player[0][0] and self.env.state[0][1] == player[0][1])

    def testPlayerBoardCorrectorBoard(self):
        player, board = self.ann.playerBoardCorrector(self.env)

        print(player, self.env.state[0])

        self.assertTrue(True) # TODO

    def testPolicyPredictor(self):
        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())

            print("Initial policy:", self.ann.predictPolicy(self.env, sess))

        self.assertTrue(True)

    def testPolicyPredictor(self):
        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())

            print("Initial value:", self.ann.predictValue(self.env, sess))

        self.assertTrue(True)

    def testAssign(self):
        with tf.Session() as sess:
            sess.run(tf.initializers.global_variables())




if __name__ == '__main__':
    unittest.main()


