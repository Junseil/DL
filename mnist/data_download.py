import matplotlib.pyplot as plt
import torchvision
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

mnist = torchvision.datasets.MNIST(root = './data', train=True, download=True)
plt.imshow(mnist.data[3], cmap='Greys', interpolation='nearest')
plt.show()