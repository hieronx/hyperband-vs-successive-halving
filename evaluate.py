from algorithms.hyperband import Hyperband
from models.cifar_cnn import CifarCnn

model = CifarCnn()
hb = Hyperband(model, 10, 3)

hb.tune()