import numpy as np
class BaselineAgent:
    """
    Has a .ckpt model from game log supervised learning
    self.base_model = base_model
    FrameWork requirements:

    - MARL: acts together with other agents in a MARL fashion:
    - VectorEnv?
    - PBT?

    It is enough to have a 569 x
    """
    def __init__(self, base_model):
        self.base_model = base_model
def xavier(input_size, output_size):
    var = 2. / (input_size + output_size)
    bound = np.sqrt(3.0 * var)
    return np.random.uniform(-bound, bound, size=(input_size, output_size))

sigmoid = lambda x: 1 / (1 + np.exp(-x))
relu = lambda x: np.maximum(x, 0)

action_probs = np.random.random(5)
obs = np.random.random(564)
w0 = xavier(564,512)
b0 = np.zeros(512)
w1 = xavier(512,512)
b1 = np.zeros(512)
w2 = xavier(512,1)
x = obs
x = np.dot(x, w0) + b0
x = relu(x)
x = np.dot(x, w1) + b1
x * relu(x)
x = np.dot(x, w2)
fold_prob = sigmoid(x)
# np.dot()
a = 1
print(min(obs), max(obs), obs.shape)
