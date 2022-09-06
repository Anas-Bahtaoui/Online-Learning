import numpy as np

from entities import CustomerClass
from production import product_configs
from itertools import product


for class_1, class_2, product_id_1, product_id_2 in product(list(CustomerClass), list(CustomerClass), range(len(product_configs)), range(len(product_configs))):
        data1 = np.load(f"./curves/{class_1.name}_{product_id_1}.npy")
        data2 = np.load(f"./curves/{class_2.name}_{product_id_2}.npy")
        if class_1.name == class_2.name and product_id_1 == product_id_2:
            continue
        if np.array_equal(data1, data2):
            print(f"{class_1.name}_{product_id_1} is equal to {class_2.name}_{product_id_2}")
