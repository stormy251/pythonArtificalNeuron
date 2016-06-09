import random
import math


def between(min, max):
    """
    Return a real random value between the given min and max.
    :param min:
    :param max:
    :return:
    """
    return random.random() * (max - min) + min


def make_matrix(n, m):
    """
    Make an N rows by M columns matrix.
    :param N:
    :param M:
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
    return (1-sgmd) * sgmd
