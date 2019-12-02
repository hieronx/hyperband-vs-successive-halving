from algorithms.hyperband import Hyperband
from models.cifar_cnn import CifarCnn

import faulthandler

model = CifarCnn()
hb = Hyperband(model, 10, 3)

faulthandler.enable()
hb.tune()