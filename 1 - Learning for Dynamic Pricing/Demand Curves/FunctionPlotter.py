"""
This is a script that plots the demand curves of the three user classes for all four poducts.
"""

import matplotlib.pyplot as plt
import numpy as np
import sys

sys.path.append('..')
from parameters import CustomerClass, products

###### READING DATA ############################

fig, axs = plt.subplots(3, sharex=True, sharey=True)
for index, class_ in enumerate(list(CustomerClass)):
    for product in products:
        data = np.load(f"./curves/{class_.name}_{product.id}.npy")
        x = data[:, 0]
        y = data[:, 1]
        axs[index].plot(x, y, label=f'{class_.name}_{product.name}')
    axs[index].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    axs[index].set_title(f'User {class_.name}')
    axs[index].set_ylabel('Demand')
    axs[index].grid(True)
plt.xlabel('Price')
plt.savefig('DemandCurves.png', bbox_inches="tight",dpi = 300)
plt.show()
