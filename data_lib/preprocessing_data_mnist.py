from torchvision.datasets import MNIST
from torchvision.transforms import transforms


def mnist_load_data():
    data_root = 'datasets/mnist'

    mean = 0.1307
    dev = 0.3081
    train_data = MNIST(root=data_root, train=True, download=True)
    test_data = MNIST(root=data_root, train=False, download=True)

    x_train = (train_data.data - mean) / dev
    x_train = x_train.numpy()
    y_train = train_data.targets.numpy()

    x_test = (test_data.data - mean) / dev
    x_test = x_test.numpy()
    y_test = test_data.targets.numpy()

    return (x_train, y_train), (x_test, y_test)


