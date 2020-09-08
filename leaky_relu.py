import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10,10,100)

def leaky_relu(x):
    return max(0.01*x,x)

def der_leaky_relu(x):
    if x < 0 :
        return 0.01
    if x >= 0 :
        return 1
    
x = np.linspace(-10,10,100)    
    
plt.figure(figsize=(12,8))
plt.plot(x, list(map(lambda x: leaky_relu(x),x)), label="leaky-relu")
plt.plot(x, list(map(lambda x: der_leaky_relu(x),x)), label="derivative")
plt.title("Leaky-ReLU")
plt.legend()
plt.show()

# Resolve o problema da death of the neuron - Disadvantage	The factor 0.01 is arbitraty, and can be tuned (PReLU for parametric ReLU)