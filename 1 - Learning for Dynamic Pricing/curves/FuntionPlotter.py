"""
This is a script that plots the demand curves of the three user classes for all four poducts.
"""

import matplotlib.pyplot as plt
import numpy as np

###### READING DATA ############################

#A
dataA_0 = np.load('A_0.npy')
dataA_1 = np.load('A_1.npy')
dataA_2 = np.load('A_2.npy')
dataA_3 = np.load('A_3.npy')
dataA_4 = np.load('A_4.npy')

xA_0 = dataA_0[:, 0]
yA_0 = dataA_0[:, 1]
xA_1 = dataA_1[:, 0]
yA_1 = dataA_1[:, 1]
xA_2 = dataA_2[:, 0]
yA_2 = dataA_2[:, 1]
xA_3 = dataA_3[:, 0]
yA_3 = dataA_3[:, 1]
xA_4 = dataA_4[:, 0]
yA_4 = dataA_4[:, 1]

#B
dataB_0 = np.load('B_0.npy')
dataB_1 = np.load('B_1.npy')
dataB_2 = np.load('B_2.npy')
dataB_3 = np.load('B_3.npy')
dataB_4 = np.load('B_4.npy')

xB_0 = dataB_0[:, 0]
yB_0 = dataB_0[:, 1]
xB_1 = dataB_1[:, 0]
yB_1 = dataB_1[:, 1]
xB_2 = dataB_2[:, 0]
yB_2 = dataB_2[:, 1]
xB_3 = dataB_3[:, 0]
yB_3 = dataB_3[:, 1]
xB_4 = dataB_4[:, 0]
yB_4 = dataB_4[:, 1]

#C
dataC_0 = np.load('C_0.npy')
dataC_1 = np.load('C_1.npy')
dataC_2 = np.load('C_2.npy')
dataC_3 = np.load('C_3.npy')
dataC_4 = np.load('C_4.npy')

xC_0 = dataC_0[:, 0]
yC_0 = dataC_0[:, 1]
xC_1 = dataC_1[:, 0]
yC_1 = dataC_1[:, 1]
xC_2 = dataC_2[:, 0]
yC_2 = dataC_2[:, 1]
xC_3 = dataC_3[:, 0]
yC_3 = dataC_3[:, 1]
xC_4 = dataC_4[:, 0]
yC_4 = dataC_4[:, 1]

###### PLOTTING DATA ############################

fig, axs = plt.subplots(3, sharex=True, sharey=True)
#A
axs[0].plot(xA_0, yA_0, label='A_0')
axs[0].plot(xA_1, yA_1, label='A_1')
axs[0].plot(xA_2, yA_2, label='A_2')
axs[0].plot(xA_3, yA_3, label='A_3')
axs[0].plot(xA_4, yA_4, label='A_4')
axs[0].legend(loc="upper right")
axs[0].set_title('User Class A')
axs[0].set_ylabel('Demand')
axs[0].grid(True)
#B
axs[1].plot(xB_0, yB_0, label='B_0')
axs[1].plot(xB_1, yB_1, label='B_1')
axs[1].plot(xB_2, yB_2, label='B_2')
axs[1].plot(xB_3, yB_3, label='B_3')
axs[1].plot(xB_4, yB_4, label='B_4')
axs[1].legend(loc="upper right")
axs[1].set_title('User Class B')
axs[1].set_ylabel('Demand')
axs[1].grid(True)
#C
axs[2].plot(xC_0, yC_0, label='C_0')
axs[2].plot(xC_1, yC_1, label='C_1')
axs[2].plot(xC_2, yC_2, label='C_2')
axs[2].plot(xC_3, yC_3, label='C_3')
axs[2].plot(xC_4, yC_4, label='C_4')
axs[2].legend(loc="upper right")
axs[2].set_title('User Class C')
axs[2].set_ylabel('Demand')
axs[2].grid(True)

plt.xlabel('Price')
plt.show()