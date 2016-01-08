#!/usr/bin/python

from token_ml_data import TokenMatchineLearningDataSet

import tensorflow as tf
import pickle
import numpy as np
import sys
import os
import argparse

# Constants
LEARNING_RATE = 2.5e-2        # Training step
MODEL_PARAMS_NAME_TEMPLATE = "model_N1_%d_nIter_%d_learnRate_%g.pkl"

def weight_variable(shape, name):
    initial = tf.truncated_normal(shape, stddev=0.1)  # Initial symmetry breaker
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    initial = tf.constant(0.1, shape=shape)  # Add initial bias
    return tf.Variable(initial, name=name)


# graphProtoBufSaveDir = "./graphs"
# graphProtoButFileNameTemplate = "graph_N1_%d_nIter_%d_learnRate_%g.pb"

if __name__ == "__main__":
    # Input argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("N1", type=int, help="Hidden layer size")
    parser.add_argument("nIter",  type=int, help="Number of training iterations")
    parser.add_argument("tokenDataDir", type=str, help="Path to token data directory")
    parser.add_argument("modelParamsOutputDir", type=str, help="Path to model parameter (traing result) output directory")

    if len(sys.argv) == 1:
        parser.print_help()

    args = parser.parse_args()

    # Model parameters
    N1 = args.N1          # Hidden layer 1 size
    nIter = args.nIter    # Number of training iterations

    token_data_dir = args.tokenDataDir
    model_params_output_dir = args.modelParamsOutputDir

    # Verify that the output directory exists
    if not os.path.isdir(model_params_output_dir):
        raise ValueError("Model parameter output directory does not exist: \"%s\"" % model_params_output_dir)

    # Obtain the data set
    token_data_set = TokenMatchineLearningDataSet(token_data_dir)

    # TensorFlow model specification
    graph1 = tf.Graph()

    with graph1.as_default():
        sess = tf.InteractiveSession()

        x = tf.placeholder("float", shape=[None, token_data_set.num_dims])
        y_ = tf.placeholder("float", shape=[None, token_data_set.num_tokens])

        # Hidden layer
        W1 = weight_variable([token_data_set.num_dims, N1], "W1")
        b1 = bias_variable([N1], "b1")

        # Readout layer
        Wo = weight_variable([N1, token_data_set.num_tokens], "Wo")
        bo = bias_variable([token_data_set.num_tokens], "bo")

        with tf.name_scope("Wx_b") as scope:
            y1 = tf.nn.tanh(tf.matmul(x, W1) + b1)
            y = tf.nn.softmax(tf.matmul(y1, Wo) + bo)

        # Add summary ops to collect data
        w1_hist = tf.histogram_summary("weights", W1)
        b1_hist = tf.histogram_summary("biases1", b1)
        y1_hist = tf.histogram_summary("y1", y1)
        wo_hist = tf.histogram_summary("weights", Wo)
        bo_hist = tf.histogram_summary("biases", bo)
        y_hist = tf.histogram_summary("y", y)

        with tf.name_scope("xent") as scope:
            cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
            ce_summ = tf.scalar_summary("cross entropy", cross_entropy)

        with tf.name_scope("train") as scope:
            # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
            train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy)

        with tf.name_scope("validation") as scope:
            correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            accuracy_summary = tf.scalar_summary("accuracy", accuracy)

        # merged = tf.merge_all_summaries()
        # writer = tf.train.SummaryWriter("/tmp/glyphoid_token_logs", sess.graph_def)

        # Initialize all variables
        sess.run(tf.initialize_all_variables())

        validation_accuracies = []

        # Model training iterations
        for i in range(nIter):
            # batch = tokenDataSet.next_batch(1000)
            # train_step.run(feed_dict={x: batch[0], y_: batch[1]})

            train_feed = {x: token_data_set.X, y_: token_data_set.y}
            validation_feed = {x: token_data_set.validation_X, y_: token_data_set.validation_y}
            test_feed = {x: token_data_set.test_X, y_: token_data_set.test_y}

            train_step.run(feed_dict=train_feed)

            train_accuracy = accuracy.eval(feed_dict=train_feed)
            validation_accuracy = accuracy.eval(feed_dict=validation_feed)

            #yHat = y.eval(feed_dict={x: tokenDataSet.validationX[0:1]}) # TODO: Copy this

            validation_accuracies.append(validation_accuracy)

            train_err_pct = (1.0 - train_accuracy) * 100.0
            validation_err_pct = (1.0 - validation_accuracy) * 100.0

            print("Step %d of %d: training error = %g%%; validation error = %g%%" %
                  (i + 1, nIter, train_err_pct, validation_err_pct))

            # result = sess.run([merged, accuracy], feed_dict=validation_feed)
            # result = sess.run([accuracy], feed_dict=validation_feed)
            # summary_str = result[0]
            # acc = result[1]
            # writer.add_summary(summary_str, i)

            # acc = accuracy.eval(feed_dict={x: tokenDataSet.testX, y_: tokenDataSet.testy})
            # print "After training iteration %d: test accuracy = %f" % (i, acc)

        # Determine the best vaidation performance
        validation_accuracies = np.array(validation_accuracies)
        best_validation_acc_iter = np.argmax(validation_accuracies) # Iteration number of best validation accuracy
        best_validation_acc = np.max(validation_accuracies)         # Best validation accuracy

        # Calculate accuracy on test set
        test_accuracy = accuracy.eval(feed_dict=test_feed)

        print "Best validation error rate = %g (iteration %d)" % \
              ((1.0 - best_validation_acc) * 100.0, best_validation_acc_iter)
        print "Test error rate = %g%%" % ((1.0 - test_accuracy) * 100.0)

    # Store Variables as numpy arrays, to pkl file
    g1_W1 = sess.run(W1)
    g1_b1 = sess.run(b1)
    g1_Wo = sess.run(Wo)
    g1_bo = sess.run(bo)

    model_params = {
        "W1": g1_W1,
        "b1": g1_b1,
        "Wo": g1_Wo,
        "bo": g1_bo,
        "best_validation_acc_iter": best_validation_acc_iter,
        "best_validation_acc": best_validation_acc,
        "test_accuracy": test_accuracy
    }

    model_params_fn = os.path.join(model_params_output_dir, MODEL_PARAMS_NAME_TEMPLATE % (N1, nIter, LEARNING_RATE))
    with open(model_params_fn, "wb") as f:
        pickle.dump(model_params, f)

    print "Saved model parameters to file \"%s\"" % model_params_fn

    # graph2 = tf.Graph()
    # with graph2.as_default():
    #     g2_x = tf.placeholder("float", shape=[1, tokenDataSet.numDims], name="theInput")
    #     # g2_x = tf.Variable("float", shape=[None, tokenDataSet.numDims], name="input")
    #
    #     g2_W1 = tf.constant(np_W1, name="W1")
    #     g2_b1 = tf.constant(np_b1, name="b1")
    #     g2_Wo = tf.constant(np_Wo, name="Wo")
    #     g2_bo = tf.constant(np_bo, name="bo")
    #
    #     g2_y1 = tf.nn.tanh(tf.matmul(g2_x, g2_W1) + g2_b1, name="y1")
    #     g2_y = tf.nn.softmax(tf.matmul(g2_y1, g2_Wo) + g2_bo, name="softmaxOutput")
    #
    #     # test_input = tf.Variable(tf.truncated_normal([1, tokenDataSet.numDims], stddev=0.1))
    #     # test_input = np.zeros([tokenDataSet.numDims])
    #
    #     test_input = tf.constant(np.zeros([1, tokenDataSet.numDims]))
    #
    #     # sess1 = tf.Session()
    #     # sess1.run(tf.initialize_all_variables())
    #
    #     yHat = g2_y.eval(feed_dict={g2_x: test_input}) # Copied working. TODO: Copy to token_recognizer.py
    #
    #     # test_output = sess1.run(g2_y, feed_dict={g2_x: test_input})
    #     # test_g2_W1 = sess1.run(g2_W1)
    #     # test_g2_y = sess1.run(g2_y, feed_dict={g2_x: test_input})
    #     # test_g2_y = sess1.run(g2_y)
    #
    # graph3 = tf.Graph()
    # with graph3.as_default():
    #     graph2Def = graph2.as_graph_def()
    #     tf.import_graph_def(graph2.as_graph_def(), input_map={"input": g2_x}, return_elements=["softmaxOutput"])
    #
    #
    # graph3Proto = graph3.as_graph_def()
    # graph3ProtoStr = graph3Proto.SerializeToString()
    #
    #
    # # print graph3ProtoStr
    #
    # graphProtoBufFileName = graphProtoButFileNameTemplate % (N1, nIter, learnRate)
    # with open(os.path.join(graphProtoBufSaveDir, graphProtoBufFileName), "wb") as f:
    #     f.write(graph3ProtoStr)
    #     print "Trained graph written to: " + graphProtoBufFileName
    #
    #
