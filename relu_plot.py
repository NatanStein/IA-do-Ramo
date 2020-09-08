import numpy as np
import matplotlib.pyplot as plt

#funcao relu
def relu(x) :
    return max(x, 0)

# Derivada da relu
def der_relu(x):
    if x <= 0 :
        return 0
    if x > 0 :
        return 1
    
x = np.linspace(-10,10,100)

plt.figure(figsize=(12,8))
plt.plot(x, list(map(lambda x: relu(x),x)), label="relu")
plt.plot(x, list(map(lambda x: der_relu(x),x)), label="derivative")
plt.title("ReLU")
plt.legend()
plt.show()

# Disadvantage	Problematic when we have lots of negative values, since the outcome is always 0 and leads to the death of the neuron