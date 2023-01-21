from torchvision.datasets import MNIST
from torchvision.transforms import transforms


def mnist_load_data():
    data_root = 'datasets/mnist'

    train_data = MNIST(root=data_root, train=True, download=True)
    test_data = MNIST(root=data_root, train=False, download=True)

    x_train = train_data.data / 255.0
    x_train = x_train.unsqueeze(1)
    x_train = x_train.numpy()
    y_train = train_data.targets.numpy()

    x_test = test_data.data / 255.0
    x_test = x_test.unsqueeze(1)
    x_test = x_test.numpy()
    y_test = test_data.targets.numpy()

    return (x_train, y_train), (x_test, y_test)


