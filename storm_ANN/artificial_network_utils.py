import random
import math


def between(minimum, maximum):
    """
    Return a real random value between the given min and max.
    :param minimum:
    :param maximum:
    :return:
    """
    return random.random() * (maximum - minimum) + minimum


def make_matrix(n, m):
    """
    Make an n rows by m columns matrix.
    :param n:
    :param m:
    :return:
    """
    return [[0 for i in range(m)] for i in range(n)]


def sigmoid(x):
    """
    This function squashes any number to a number btween 0 and 1
    :return:
    """
    return 1.0 / (1 + math.exp(-x))


def deriv_sigmoid(x):
    """
    This will return the value of the derivative with the input of X
    :param x:
    :return:
    """
    sgmd = sigmoid(x)
    return (1 - sgmd) * sgmd
