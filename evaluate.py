from algorithms.hyperband import Hyperband
from models.mnist_cnn import FashionMNISTCNN

model = FashionMNISTCNN()
hb = Hyperband(model, 10, 3)

hb.tune()