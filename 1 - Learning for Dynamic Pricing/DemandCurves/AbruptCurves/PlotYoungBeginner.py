import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('..')

data = np.load('./AC_YOUNG_BEGINNER_0.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, linestyle=':', color='blue')

data = np.load('./YOUNG_BEGINNER_0.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, color='blue', label='Tshirt')

"""
data = np.load('./AC_YOUNG_BEGINNER_1.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, label='Shorts')

data = np.load('./AC_YOUNG_BEGINNER_2.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, label='Twoel')

data = np.load('./AC_YOUNG_BEGINNER_3.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, label='Dumbbells')

data = np.load('./AC_YOUNG_BEGINNER_4.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, label='Preotein Powder')
"""

plt.legend()
plt.grid(True)
plt.show()