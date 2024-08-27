x = [1]
from itertools import combinations_with_replacement as cmr
import numpy as np
combinations = cmr(x, 0)

product = [np.prod(x) for x in combinations]
print(product[0])