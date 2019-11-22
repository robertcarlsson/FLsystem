import numpy as np 
import matplotlib.pyplot as plt

X = np.arange(0, 100, 1)
y = np.array([x**2 for x in X])

plt.plot(X, y)
plt.ylabel('some numbers')
plt.xlabel('Other numbers')
plt.show()