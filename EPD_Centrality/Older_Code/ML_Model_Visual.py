import numpy as np
import matplotlib.pyplot as plt

inputs = []
weights1 = []
weights2 = []
weights3 = [32, 0]
plt.scatter(weights3[0], weights3[1], s=20, marker='o', color='red')

for i in range(33):
    inputs.append([2, 1000000*(i-16)])
    plt.scatter(inputs[i][0], inputs[i][1], s=20,
                marker='o', color='red')

for i in range(64):
    weights1.append([12, 1000000*(i-32)])
    plt.scatter(weights1[i][0], weights1[i][1], s=20,
                marker='o', color='blue')


for i in range(32):
    weights2.append([22, 1000000*(i-16)])
    plt.scatter(weights2[i][0], weights2[i][1], s=20, marker='o', color='blue')

for i in range(64):
    for j in range(33):
        plt.plot([inputs[j][0], weights1[i][0]], [inputs[j][1],
                                                  weights1[i][1]], linewidth=0.05, color='black')

for i in range(64):
    for j in range(32):
        plt.plot([weights1[i][0], weights2[j][0]], [weights1[i]
                                                    [1], weights2[j][1]], linewidth=0.05, color='black')

for i in range(32):
    plt.plot([weights2[i][0], weights3[0]], [weights2[i]
                                             [1], weights3[1]], linewidth=0.1, color='black')

plt.show()
