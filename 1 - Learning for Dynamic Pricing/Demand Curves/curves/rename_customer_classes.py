import os
import shutil
from Customer import CustomerClass
mapping = {
    "A": CustomerClass.PROFESSIONAL,
    "B": CustomerClass.YOUNG_BEGINNER,
    "C": CustomerClass.OLD_BEGINNER,
}
current_dir = os.path.dirname(__file__)
originals = [path for path in os.listdir(current_dir) if path.endswith(".npy")]
for original in originals:
    name = original[:-6]
    rest = original[-6:]
    class_ = mapping[name]
    target = os.path.join(current_dir, f"{class_.name}{rest}")
    shutil.move(original, target)