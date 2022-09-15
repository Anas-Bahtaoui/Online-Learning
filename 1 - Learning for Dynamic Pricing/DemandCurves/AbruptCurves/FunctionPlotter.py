"""
This is a script that plots the demand curves of the three user classes for all four poducts.
"""

from turtle import color
import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('..')


###### READING DATA ############################

fig, axs = plt.subplots(3, sharex=True, sharey=True)
# OLD BEGINNER
data = np.load('./AC_OLD_BEGINNER_0.npy')
x = data[:, 0]
y = data[:, 1]
axs[2].plot(x, y, linestyle='--', color='blue')
#axs[2].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[2].set_title('OLD BEGINNER')
axs[2].set_ylabel('avg. Demand')
axs[2].grid(True)
plt.xlabel('Price')

data = np.load('./OLD_BEGINNER_0.npy')
x = data[:, 0]
y = data[:, 1]
axs[2].plot(x, y, color='blue', label='Tshirt')
#axs[2].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[2].set_title('OLD BEGINNER')
axs[2].set_ylabel('avg. Demand')
axs[2].grid(True)
plt.xlabel('Price')

data = np.load('./AC_OLD_BEGINNER_1.npy')
x = data[:, 0]
y = data[:, 1]
axs[2].plot(x, y, linestyle='--', color='orange')
#axs[2].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[2].set_title('OLD BEGINNER')
axs[2].set_ylabel('avg. Demand')
axs[2].grid(True)
plt.xlabel('Price')

data = np.load('./OLD_BEGINNER_1.npy')
x = data[:, 0]
y = data[:, 1]
axs[2].plot(x, y, color='orange', label='Shorts')
#axs[2].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[2].set_title('OLD BEGINNER')
axs[2].set_ylabel('avg. Demand')
axs[2].grid(True)
plt.xlabel('Price')

data = np.load('./AC_OLD_BEGINNER_2.npy')
x = data[:, 0]
y = data[:, 1]
axs[2].plot(x, y,  linestyle='--', color='green')
#axs[2].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[2].set_title('OLD BEGINNER')
axs[2].set_ylabel('avg. Demand')
axs[2].grid(True)
plt.xlabel('Price')

data = np.load('./OLD_BEGINNER_2.npy')
x = data[:, 0]
y = data[:, 1]
axs[2].plot(x, y, color='green', label='Towel')
#axs[2].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[2].set_title('OLD BEGINNER')
axs[2].set_ylabel('avg. Demand')
axs[2].grid(True)
plt.xlabel('Price')

data = np.load('./AC_OLD_BEGINNER_3.npy')
x = data[:, 0]
y = data[:, 1]
axs[2].plot(x, y, linestyle='--', color='red')
#axs[2].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[2].set_title('OLD BEGINNER')
axs[2].set_ylabel('avg. Demand')
axs[2].grid(True)
plt.xlabel('Price')

data = np.load('./OLD_BEGINNER_3.npy')
x = data[:, 0]
y = data[:, 1]
axs[2].plot(x, y, color='red', label='Dumbbells')
#axs[2].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[2].set_title('OLD BEGINNER')
axs[2].set_ylabel('avg. Demand')
axs[2].grid(True)
plt.xlabel('Price')

data = np.load('./AC_OLD_BEGINNER_4.npy')
x = data[:, 0]
y = data[:, 1]
axs[2].plot(x, y, linestyle='--', color='purple')
#axs[2].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[2].set_title('OLD BEGINNER')
axs[2].set_ylabel('avg. Demand')
axs[2].grid(True)
plt.xlabel('Price')

data = np.load('./OLD_BEGINNER_4.npy')
x = data[:, 0]
y = data[:, 1]
axs[2].plot(x, y, color = 'purple', label='Preotein Powder')
#axs[2].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[2].set_title('OLD BEGINNER')
axs[2].set_ylabel('avg. Demand')
axs[2].grid(True)
plt.xlabel('Price')

#YOUNG
data = np.load('./AC_YOUNG_BEGINNER_0.npy')
x = data[:, 0]
y = data[:, 1]
axs[1].plot(x, y, linestyle='--', color='blue')
#axs[1].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[1].set_title('YOUNG BEGINNER')
axs[1].set_ylabel('avg. Demand')
axs[1].grid(True)
plt.xlabel('Price')

data = np.load('./YOUNG_BEGINNER_0.npy')
x = data[:, 0]
y = data[:, 1]
axs[1].plot(x, y, color='blue', label='Tshirt')
#axs[1].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[1].set_title('YOUNG BEGINNER')
axs[1].set_ylabel('avg. Demand')
axs[1].grid(True)
plt.xlabel('Price')

data = np.load('./AC_YOUNG_BEGINNER_1.npy')
x = data[:, 0]
y = data[:, 1]
axs[1].plot(x, y, linestyle='--', color='orange')
#axs[1].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[1].set_title('YOUNG BEGINNER')
axs[1].set_ylabel('avg. Demand')
axs[1].grid(True)
plt.xlabel('Price')

data = np.load('./YOUNG_BEGINNER_1.npy')
x = data[:, 0]
y = data[:, 1]
axs[1].plot(x, y, color='orange', label='Shorts')
#axs[1].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[1].set_title('YOUNG BEGINNER')
axs[1].set_ylabel('avg. Demand')
axs[1].grid(True)
plt.xlabel('Price')

data = np.load('./AC_YOUNG_BEGINNER_2.npy')
x = data[:, 0]
y = data[:, 1]
axs[1].plot(x, y, linestyle='--', color='green')
#axs[1].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[1].set_title('YOUNG BEGINNER')
axs[1].set_ylabel('avg. Demand')
axs[1].grid(True)
plt.xlabel('Price')

data = np.load('./YOUNG_BEGINNER_2.npy')
x = data[:, 0]
y = data[:, 1]
axs[1].plot(x, y, color='green', label='Twoel')
#axs[1].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[1].set_title('YOUNG BEGINNER')
axs[1].set_ylabel('avg. Demand')
axs[1].grid(True)
plt.xlabel('Price')

data = np.load('./AC_YOUNG_BEGINNER_3.npy')
x = data[:, 0]
y = data[:, 1]
axs[1].plot(x, y, linestyle='--', color='red')
#axs[1].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[1].set_title('YOUNG BEGINNER')
axs[1].set_ylabel('avg. Demand')
axs[1].grid(True)
plt.xlabel('Price')

data = np.load('./YOUNG_BEGINNER_3.npy')
x = data[:, 0]
y = data[:, 1]
axs[1].plot(x, y,  color='red', label='Dumbbells')
#axs[1].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[1].set_title('YOUNG BEGINNER')
axs[1].set_ylabel('avg. Demand')
axs[1].grid(True)
plt.xlabel('Price')

data = np.load('./AC_YOUNG_BEGINNER_4.npy')
x = data[:, 0]
y = data[:, 1]
axs[1].plot(x, y, linestyle='--', color='purple')
#axs[1].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[1].set_title('YOUNG BEGINNER')
axs[1].set_ylabel('avg. Demand')
axs[1].grid(True)
plt.xlabel('Price')

data = np.load('./YOUNG_BEGINNER_4.npy')
x = data[:, 0]
y = data[:, 1]
axs[1].plot(x, y, color='purple', label='Preotein Powder')
#axs[1].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[1].set_title('YOUNG BEGINNER')
axs[1].set_ylabel('avg. Demand')
axs[1].grid(True)
plt.xlabel('Price')

####################
#               PRO                  #
####################
data = np.load('./AC_PROFESSIONAL_0.npy')
x = data[:, 0]
y = data[:, 1]
axs[0].plot(x, y, color='blue')
#axs[0].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[0].set_title('PROFESSIONAL')
axs[0].set_ylabel('avg. Demand')
axs[0].grid(True)
plt.xlabel('Price')

data = np.load('./PROFESSIONAL_0.npy')
x = data[:, 0]
y = data[:, 1]
axs[0].plot(x, y, linestyle='--', color='blue', label='Tshirt')
#axs[0].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[0].set_title('PROFESSIONAL')
axs[0].set_ylabel('avg. Demand')
axs[0].grid(True)
plt.xlabel('Price')


data = np.load('./AC_PROFESSIONAL_1.npy')
x = data[:, 0]
y = data[:, 1]
axs[0].plot(x, y, linestyle='--', color='orange')
#axs[0].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[0].set_title('PROFESSIONAL')
axs[0].set_ylabel('avg. Demand')
axs[0].grid(True)
plt.xlabel('Price')

data = np.load('./PROFESSIONAL_1.npy')
x = data[:, 0]
y = data[:, 1]
axs[0].plot(x, y, color = 'orange', label='Shorts')
#axs[0].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[0].set_title('PROFESSIONAL')
axs[0].set_ylabel('avg. Demand')
axs[0].grid(True)
plt.xlabel('Price')

data = np.load('./AC_PROFESSIONAL_2.npy')
x = data[:, 0]
y = data[:, 1]
axs[0].plot(x, y, linestyle='--', color='green')
#axs[0].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[0].set_title('PROFESSIONAL')
axs[0].set_ylabel('avg. Demand')
axs[0].grid(True)
plt.xlabel('Price')

data = np.load('./PROFESSIONAL_2.npy')
x = data[:, 0]
y = data[:, 1]
axs[0].plot(x, y, color = 'green', label='Twoel')
#axs[0].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[0].set_title('PROFESSIONAL')
axs[0].set_ylabel('avg. Demand')
axs[0].grid(True)
plt.xlabel('Price')

data = np.load('./AC_PROFESSIONAL_3.npy')
x = data[:, 0]
y = data[:, 1]
axs[0].plot(x, y, linestyle='--', color='red')
#axs[0].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[0].set_title('PROFESSIONAL')
axs[0].set_ylabel('avg. Demand')
axs[0].grid(True)
plt.xlabel('Price')

data = np.load('./PROFESSIONAL_3.npy')
x = data[:, 0]
y = data[:, 1]
axs[0].plot(x, y, color='red', label='Dumbbells')
#axs[0].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[0].set_title('PROFESSIONAL')
axs[0].set_ylabel('avg. Demand')
axs[0].grid(True)
plt.xlabel('Price')


data = np.load('./AC_PROFESSIONAL_4.npy')
x = data[:, 0]
y = data[:, 1]
axs[0].plot(x, y, linestyle='--', color='purple')
#axs[0].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[0].set_title('PROFESSIONAL')
axs[0].set_ylabel('avg. Demand')
axs[0].grid(True)
plt.xlabel('Price') 

data = np.load('./PROFESSIONAL_4.npy')
x = data[:, 0]
y = data[:, 1]
axs[0].plot(x, y, color='purple', label='Preotein Powder')
#axs[0].legend(loc='center left', bbox_to_anchor=(1, 2.5))
axs[0].set_title('PROFESSIONAL')
axs[0].set_ylabel('avg. Demand')
axs[0].grid(True)
plt.xlabel('Price') 


#SHOW AND SAVE
#Plot legend ounder the plot
plt.legend(loc='right', bbox_to_anchor=(1, 2.5))
plt.savefig('DemandCurves.png', bbox_inches="tight", dpi=300)
plt.show()
