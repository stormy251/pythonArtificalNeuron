import random


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

