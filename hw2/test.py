import numpy as np


class DataPreProcessing:
    def __init__(self):
        pass

    @staticmethod
    def read_data(file, separator):
        """
        Use the Numpy function fromfile() to load the dataset from the given file.

        Parameters
        ----------
        file: string
            String path of the data file.
        separator: string
            Separator between items if file is a text file.

        Return
        ----------
        data: ndarray
            Data read from the given file.
        """
        # TODO: write your code here
        dataset = np.fromfile(file, sep=separator, dtype=float, count=-1)
        return dataset

    @staticmethod
    def train_test_split(data, train_size, shuffle=True):
        """
        Split the given data into random train and test subsets.

        Parameters
        ----------
        data: ndarray
            Input of the given data.
        train_size: float
            The proportion of the dataset to include in the train split.
        shuffle: bool
            Whether or not to shuffle the data before splitting.

        Return
        ----------
        train_data: ndarray
            Output of the training data.
        test_data: ndarray
            Output of the test data.
        """
        # TODO: write your code here
        if shuffle:
            np.random.shuffle(data)
        train_len = int(train_size * len(data))
        train_data = data[:train_len]
        test_data = data[train_len:]
        return train_data, test_data

    @staticmethod
    def minmax_scale(X, feature_range=(0, 1)):
        """
        Transform features X by scaling each feature to a given range.

        Parameters
        ----------
        data: ndarray
            Input of the given data.
        feature_range: tuple
            Desired range of transformed data.

        Return
        ----------
        X_scaled: ndarray
            Output of the scaled features.
        """
        # TODO: write your code here
        # Compute the minimum and maximum value for each feature
        val_min = np.min(X, axis=0)
        val_max = np.max(X, axis=0)

        # Compute the range for each feature
        feature_ranges = val_max - val_min

        # Compute the scaled features
        X_scaled = (X - val_min) / feature_ranges * (feature_range[1] - feature_range[0]) + feature_range[0]

        return X_scaled

# Init the train size
train_size = 0.8

# TODO: init an object of DataPreProcessing
preprocessor = DataPreProcessing()

# TODO: read the data from the data file
dataset = preprocessor.read_data('housing.data', separator=" ")

# TODO: split the data random train and test subsets
train_dataset, test_dataset = preprocessor.train_test_split(dataset, train_size)

# TODO: split the training subset into X_train and y_train
X_train = train_dataset[:-2]
y_train = train_dataset[-1]

# TODO: split the test subset into X_test and y_test
X_test = train_dataset[:-2]
y_test = train_dataset[-1]


# TODO: 0-1 scale X_train and X_test respectively
X_train = preprocessor.minmax_scale(X_train)
X_test = preprocessor.minmax_scale(X_test)

# TODO: reshape X_train, X_test, y_train, and y_test to satisfy the requirments
