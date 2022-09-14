import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('..')

# OLD BEGINNER
data = np.load('./AC_OLD_BEGINNER_0.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, linestyle=':', color='blue')


data = np.load('./OLD_BEGINNER_0.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, color='blue', label='Tshirt')


data = np.load('./AC_OLD_BEGINNER_1.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, linestyle=':', color='orange')


data = np.load('./OLD_BEGINNER_1.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, color='orange', label='Shorts')

data = np.load('./AC_OLD_BEGINNER_2.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y,  linestyle=':', color='green')

data = np.load('./OLD_BEGINNER_2.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, color='green', label='Towel')


data = np.load('./AC_OLD_BEGINNER_3.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, linestyle=':', color='red')


data = np.load('./OLD_BEGINNER_3.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, color='red', label='Dumbbells')

data = np.load('./AC_OLD_BEGINNER_4.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, linestyle=':', color='purple')


data = np.load('./OLD_BEGINNER_4.npy')
x = data[:, 0]
y = data[:, 1]
plt.plot(x, y, color = 'purple', label='Preotein Powder')

plt.legend()
plt.grid(True)
plt.show()