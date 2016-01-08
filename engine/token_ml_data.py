import numpy as np
import csv
import os


def read_csv(csvFileName):
    cr = csv.reader(open(csvFileName, "rt"))

    numDims = -1
    X = []
    for row in cr:
        if numDims == -1:
           numDims = len(row)
        xRow = [float(x) for x in row]
        X.append(xRow)

    return X, numDims


class TokenMatchineLearningDataSet(object):
    # Private members

    threshRowCountPerToken = 20

    def __init__(self, data_dir):
        self._data_dir = data_dir

        self.train_data_fn = os.path.join(self._data_dir, "sdve_train_data.csv")
        self.train_labels_fn = os.path.join(self._data_dir, "sdve_train_labels.csv")
        self.validation_data_fn = os.path.join(self._data_dir, "sdve_validation_data.csv")
        self.validation_labels_fn = os.path.join(self._data_dir, "sdve_validation_labels.csv")
        self.test_data_fn = os.path.join(self._data_dir, "sdve_test_data.csv")
        self.test_labels_fn = os.path.join(self._data_dir, "sdve_test_labels.csv")

        self.token_names_fn = os.path.join(self._data_dir, "token_names.txt")

        print "Training data file =", self.train_data_fn
        print "Training labels file =", self.train_labels_fn
        print "Token names file =", self.token_names_fn
    
        # Read the token names
        self.token_names = []
    
        with open(self.token_names_fn, "rt") as fileContent:
            self.token_names = fileContent.readlines()
        self.token_names = [tokenName.replace("\r", "").replace("\n", "") for tokenName in self.token_names]
    
        print "Number of unique tokens = %d" % len(self.token_names)
    
        # Read the training data
        (X, num_dims) = read_csv(self.train_data_fn)
        (validation_X, validation_num_dims) = read_csv(self.validation_data_fn)
        (test_X, test_num_dims) = read_csv(self.test_data_fn)

        assert(num_dims == validation_num_dims)
        assert(num_dims == test_num_dims)

        self.num_dims = num_dims
        self.num_rows = len(X)
        self.validation_num_rows = len(validation_X)
        self.test_num_rows = len(test_X)
    
        print "Number of dimensions in training data = %d" % self.num_dims
        print "Number of rows in training data = %d" % self.num_rows
        print "Number of rows in validation data = %d" % self.validation_num_rows
        print "Number of rows in test data = %d" % self.test_num_rows
    
        X = np.array(X)
        validation_X = np.array(validation_X)
        test_X = np.array(test_X)
    
        # Read the training labels
        (y, self.num_tokens) = read_csv(self.train_labels_fn)
        (validation_y, self.validation_num_tokens) = read_csv(self.validation_labels_fn)
        (test_y, self.test_num_tokens) = read_csv(self.test_labels_fn)

        assert(len(y) == self.num_rows)
        assert(len(validation_y) == self.validation_num_rows)
        assert(len(test_y) == self.test_num_rows)

        assert(self.num_tokens == len(self.token_names))
        assert(self.validation_num_tokens == len(self.token_names))
        assert(self.test_num_tokens == len(self.token_names))

        y = np.array(y)
        validation_y = np.array(validation_y)
        test_y = np.array(test_y)

        # Apply row count threshold to tokens
        self.row_counts_per_token = [0] * self.num_tokens
        for labelRow in y:
            idx = np.nonzero(labelRow)[0]
            self.row_counts_per_token[idx] += 1

        self.include_token_idx = []
        self.exclude_token_idx = []
        self.include_tokens = []

        for i in range(self.num_tokens):
            if self.row_counts_per_token[i] >= self.threshRowCountPerToken:
                self.include_token_idx.append(i)
                self.include_tokens.append(self.token_names[i])
            else:
                self.exclude_token_idx.append(i)
                print "Excluding token due to too few rows in data: %s (%d)" % (self.token_names[i], self.row_counts_per_token[i])

        # Filter training data
        include_rows = []
        for i in range(self.num_rows):
            labels = y[i]
            idx = np.nonzero(labels)[0]
            if not np.any(self.exclude_token_idx == idx):
                include_rows.append(i)

        X = X[include_rows]
        y = y[include_rows]
        self.X = X
        self.y = y

        self.num_rows = len(include_rows)

        print "After exclusion, number of training rows = %d" % self.num_rows

        # Filter validation data
        include_rows = []
        for i in range(self.validation_num_rows):
            labels = validation_y[i]
            idx = np.nonzero(labels)[0]
            if not np.any(self.exclude_token_idx == idx):
                include_rows.append(i)

        validation_X = validation_X[include_rows]
        validation_y = validation_y[include_rows]
        self.validation_X = validation_X
        self.validation_y = validation_y

        self.validation_num_rows = len(include_rows)

        print "After exclusion, number of validation rows = %d" % self.validation_num_rows

        # Filter test data
        include_rows = []
        for i in range(self.test_num_rows):
            labels = test_y[i]
            idx = np.nonzero(labels)[0]
            if not np.any(self.exclude_token_idx == idx):
                include_rows.append(i)

        test_X = test_X[include_rows]
        test_y = test_y[include_rows]
        self.test_X = test_X
        self.test_y = test_y

        self.test_num_rows = len(include_rows)

        print "After exclusion, number of test rows = %d" % self.test_num_rows

        # Initialize batch indices
        self._epochs_completed = 0
        self._index_in_epoch = 0

    def next_batch(self, batch_size):
        """Return the next `batch_size` examples from this data set."""
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self.num_rows:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self.num_rows)
            np.random.shuffle(perm)
            self.X = self.X[perm]
            self.y = self.y[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self.num_rows

        end = self._index_in_epoch
        print "end = %d" % end
        return self.X[start : end], self.y[start : end]