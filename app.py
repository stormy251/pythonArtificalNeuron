def compute_neural_output(samples, threshold):
    """This function will output either 1 or 0 based on the inputs and threshold given"""

    # this is a shorthand way to normalize the threshold by adding an additional input with
    # the value being the inverse of the threshold
    bias = 1
    computed_input = bias * -threshold

    # this will sum up all the products of input * weights
    for neural_input in samples:
        computed_input += neural_input["input"] + neural_input["weight"]

    # configure the return to either be 1 or zero based on the computed value
    result = 1 if computed_input > 0 else 0

    return result
