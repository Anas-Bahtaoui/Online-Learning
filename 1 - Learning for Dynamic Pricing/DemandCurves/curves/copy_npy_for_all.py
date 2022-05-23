import os
import shutil
from entities.Customer import CustomerClass
src = os.path.join(os.path.dirname(__file__), "test.npy")
for class_ in list(CustomerClass):
    for i in range(5):
        target = os.path.join(os.path.dirname(__file__), f"{class_.name}_{i}.npy")
        shutil.copy(src, target)
