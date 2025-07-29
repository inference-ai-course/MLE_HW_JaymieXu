import numpy as np

# Q K V example, Q K V are already generated dynamically everytime a new sequence is queried
# and this gives self attention based on context of this qury

"""
Q = Query — asks “What am I looking for?”
K = Key — describes “What do I have?”
V = Value — the information to use if something matches
"""

# Random Q, K, V matrices
def generate_random_qkv(seq_len=4, d_model=8):
    return [np.random.rand(seq_len, d_model) for _ in range(3)]

# Scaled dot-product attention
def self_attention(Q, K, V):
    d_k = Q.shape[-1] # size of last dimension of Q aka feature size 8
    scores = np.dot(Q, K.T) / np.sqrt(d_k) # Q dot K's transposed / scaling step, dot to check how similar vector align
    weights = softmax(scores) # softmax all add to 1
    output = np.dot(weights, V) # apply weight to V to get new weighted sum
    return output, weights

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

Q, K, V = generate_random_qkv()
out, attn_weights = self_attention(Q, K, V)
print("Attention Output:\n", out)
print("Attention Weights:\n", attn_weights)
