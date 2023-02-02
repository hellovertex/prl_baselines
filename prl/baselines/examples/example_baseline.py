import numpy as np
class BaselineAgent:
    """
    Has a .ckpt model from game log supervised learning
    self.base_model = base_model
    FrameWork requirements:

    - MARL: acts together with other agents in a MARL fashion:
    - VectorEnv?
    - PBT?
    # algo:
    # 0. define fitness function
    # 1. init 6 players with random weights
    # 2. play M games -- collect rewards per player
    # 3. evaluate players using rewards collected
    # 4. select the best player P1 using fitness function -- i) mutate ii) save weights to current best
    # 5. repeat 2,3,4
    It is enough to have a 569 x

    todo: figure out how we can use the trained .ckpt MLP quickly(vectorized/mp) with this numpy based approach
    """
    def __init__(self, base_model):
        self.base_model = base_model


def mutate(weights, mutation_rate=0.1, mutation_std=0.1):
    # Normalize the weights to have a unit norm
    norm = np.linalg.norm(weights)
    weights = weights / norm

    # Generate random noise with a standard deviation of mutation_std
    noise = np.random.normal(0, mutation_std, size=weights.shape)

    # Apply the mutation by adding the noise to the weights with a probability of mutation_rate
    mutation = np.random.binomial(1, mutation_rate, size=weights.shape)
    weights = weights + mutation * noise

    # Renormalize the weights to have the same norm as before
    weights = weights * norm

    return weights

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
