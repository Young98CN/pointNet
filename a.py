import torch
import torch.nn as nn
import numpy as np

a = ["ab_1", "bc_2", "cd_3"]
b = ["_".join(i.split("_")[0:-1]) for i in a]
c = [i.split("_")[0] for i in a]
print(b,"\n",c)


