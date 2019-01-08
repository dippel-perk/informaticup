from classifier.neural_net import NeuralNet
from keras.utils import plot_model

net = NeuralNet()
model = net.get_model()
plot_model(model, to_file='../tmp/model.png', show_shapes=True)