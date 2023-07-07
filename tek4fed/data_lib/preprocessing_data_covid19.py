from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import numpy as np


def get_paths(image_dirs, class_dict):
    image_paths = [x for x in os.listdir(image_dirs.replace('\\', '/')) if x.lower().endswith('png')]
    x_data = []
    y_data = []

    for path in image_paths:
        name = path.split('-')[0]
        x_data.append(path)
        y_data.append(class_dict[name.lower()])

    y_data = np.array(y_data, dtype=np.int32)
    return x_data, y_data


def covid_load_data():
    train_dir = "datasets/covid19/train"
    test_dir = "datasets/covid19/test"

    class_dict = {
        'normal': 0,
        'viral pneumonia': 1,
        'covid': 2
    }

    x_train, y_train = get_paths(train_dir, class_dict)
    x_test, y_test = get_paths(test_dir, class_dict)


    print("+++ covid 19 dataset: +++")
    print("\tNumber of training samples: ", len(y_train))
    print("\tNumber of testing samples: ", len(y_test))

    return (x_train, y_train), (x_test, y_test)


train_transform = transforms.Compose([
    # Converting images to the size that the model expects
    transforms.Resize(size=(224, 224)),
    transforms.RandomHorizontalFlip(),  # A RandomHorizontalFlip to augment our data
    transforms.ToTensor(),  # Converting to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    # Normalizing the data to the data that the ResNet18 was trained on
])

# Creating a Transformation Object
test_transform = transforms.Compose([
    # Converting images to the size that the model expects
    transforms.Resize(size=(224, 224)),
    # We don't do data augmentation in the test/val set
    transforms.ToTensor(),  # Converting to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    # Normalizing the data to the data that the ResNet18 was trained on
])


class ChestXRayDataset(Dataset):
    def __init__(self, image_paths, image_classes, transform_type):
        self.base_dir = "datasets/covid19/" + transform_type
        self.image_paths = image_paths
        self.image_classes = image_classes

        self.transform = None
        if transform_type == 'train':
            self.transform = train_transform
        elif transform_type == 'test':
            self.transform = test_transform

        self.class_dict = {
            0: 'normal',
            1: 'viral pneumonia',
            2: 'covid'
        }

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = os.path.join(self.base_dir, self.image_paths[index])
        image = Image.open(image_path).convert('RGB')
        return self.transform(image), self.image_classes[index]
