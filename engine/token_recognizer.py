from token_ml_data import TokenMatchineLearningDataSet
import tensorflow as tf
import pickle
import numpy as np


class TokenRecognizer(object):
    # Private members
    _token_names = []
    _num_tokens = 0

    # Constructor
    def __init__(self, model_params_path, token_names_path):
        # Load token names
        self.read_token_names(token_names_path)

        # Load model parameters
        with open(model_params_path, "rb") as f:
            self.modelParams = pickle.load(f)

        shape_W1 = np.shape(self.modelParams["W1"])
        self._input_vector_len = shape_W1[0]

        shape_bo = np.shape(self.modelParams["bo"])
        assert shape_bo[0] == self.num_tokens

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.x = tf.placeholder("float", shape=[None, 78], name="theInput")

            self.W1 = tf.constant(self.modelParams["W1"], name="W1")
            self.b1 = tf.constant(self.modelParams["b1"], name="b1")
            self.Wo = tf.constant(self.modelParams["Wo"], name="Wo")
            self.bo = tf.constant(self.modelParams["bo"], name="bo")

            self.y1 = tf.nn.tanh(tf.matmul(self.x, self.W1) + self.b1, name="y1")
            self.y = tf.nn.softmax(tf.matmul(self.y1, self.Wo) + self.bo, name="softmaxOutput")

            self.tfSession = tf.InteractiveSession()
            self.tfSession.run(tf.initialize_all_variables())

    def read_token_names(self, token_names_path):
        with open(token_names_path, "rt") as f:
            txt = f.read().replace("\r\n", "\n").replace("\r", "\n")

        self._token_names = filter(None, txt.split("\n"))
        self._num_tokens = len(self._token_names)

    def recognize(self, input_vec):
        # Check the length of the input
        input_shape = np.shape(input_vec)

        if len(input_shape) != 2:
            raise ValueError("Input must be a two-dimensional array, with 1st dimension as the samples " +
                             "and the 2nd dimension as the features")

        if input_shape[1] != self.input_vector_len:
            raise ValueError("Incorrect input shape")

        n_inputs = input_shape[0]

        with self.graph.as_default():
            y_hat = self.y.eval(feed_dict={self.x: input_vec}) # Copied working. TODO: Copy to token_recognizer.py

        recog_winners = [""] * n_inputs
        recog_ps = [[]] * n_inputs
        assert np.shape(y_hat)[0] == n_inputs

        for i in range(n_inputs):
            recog_winners[i] = self._get_winner_token(y_hat[i])
            recog_ps[i] = self._make_recog_ps(y_hat[i])

        return recog_winners, recog_ps

    # Properties
    @property
    def input_vector_len(self):
        return self._input_vector_len

    @property
    def num_tokens(self):
        return self._num_tokens

    # Private functions
    def _get_winner_token(self, vec):
        return self._token_names[np.argmax(vec)]

    def _make_recog_ps(self, vec):
        recog_ps = [[]] * self.num_tokens
        for i in range(self.num_tokens):
            recog_ps[i] = [self._token_names[i], "%.6e" % vec[i]]

        return recog_ps

if __name__ == "__main__":
    # graphDef = tf.GraphDef()

    # Obtain the data set
    tokenDataSet = TokenMatchineLearningDataSet()

    model_params_path = "./model-params/model_N1_120_nIter_200_learnRate_0.02.pkl"
    token_names_path  = "./glyphoid-token-data/token_names.txt"

    tokenRecognizer = TokenRecognizer(model_params_path, token_names_path)

    print "Input vector length: %d" % tokenRecognizer.input_vector_len
    print "Number of tokens: %d" % tokenRecognizer.num_tokens

    # f = open("/home/scai/learn/tf/graphs/graph_N1_100_nIter_100_learnRate_0.02.pb", "rb")
    # graphDef.ParseFromString(f.read())

    # idx = 3
    test_input = tokenDataSet.validation_X

    (recog_winners, recog_ps) = tokenRecognizer.recognize(test_input[100:101])
    print recog_winners
    print recog_ps
