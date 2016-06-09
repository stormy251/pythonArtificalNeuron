from app import compute_neural_output

samples = [
    {
        "input": -2,
        "weight": 1,
        "output": 0
    },
    {
        "input": 3,
        "weight": 1,
        "output": 1
    },
    {
        "input": 0,
        "weight": -2,
        "output": 0
    }
]

print compute_neural_output(samples, .4)
