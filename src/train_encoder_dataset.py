from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from glob import glob
from math import ceil
import os
from random import shuffle
import torchvision.transforms as v2
import scipy.io

class train_encoder_dataset(Dataset):
    def __init__(self, operation, apply_data_augmentation, datasets, datasets_folder_path, transform, val_percent):
        if operation not in ["train", "val", "all"]:
            raise ValueError("Operation must be 'train', 'val', or 'all'.")
        
        self.operation = operation
        self.apply_data_augmentation = apply_data_augmentation
        self.transform = transform

        self.images = []

        for dataset in datasets:
            dataset_path = f"{datasets_folder_path}{dataset}/"

            match dataset:
                case 'cifar100' | 'cifar10':
                    classes = os.listdir(dataset_path + "train/")
                    images_per_class = {}
                    for _class in classes:
                        images_per_class[_class] = []
                        images_per_class[_class].extend(glob(dataset_path + "train/" + _class + "/*.png"))
                        images_per_class[_class].extend(glob(dataset_path + "test/" + _class + "/*.png"))

                        images_per_class[_class] = sorted(images_per_class[_class])

                        n_val_images = ceil(len(images_per_class[_class]) * val_percent)
                        
                        match self.operation:
                            case "train":
                                images_per_class[_class] = images_per_class[_class][:-n_val_images]
                            case "val":
                                images_per_class[_class] = images_per_class[_class][-n_val_images:]
                            case "all":
                                pass

                        self.images.extend(images_per_class[_class])

                case 'dtd':
                    classes = os.listdir(dataset_path + "dtd/dtd/images/")
                    images_per_class = {}
                    for _class in classes:
                        images_per_class[_class] = []
                        images_per_class[_class].extend(glob(dataset_path + "dtd/dtd/images/" + _class + "/*.jpg"))

                        images_per_class[_class] = sorted(images_per_class[_class])

                        n_val_images = ceil(len(images_per_class[_class]) * val_percent)
                        
                        match self.operation:
                            case "train":
                                images_per_class[_class] = images_per_class[_class][:-n_val_images]
                            case "val":
                                images_per_class[_class] = images_per_class[_class][-n_val_images:]
                            case "all":
                                pass

                        self.images.extend(images_per_class[_class])

                case 'fgvc-aircraft':
                    _images = glob(dataset_path + "fgvc-aircraft-2013b/data/images/*.jpg")
                    test_classes_file = f"{dataset_path}fgvc-aircraft-2013b/data/images_variant_test.txt"
                    trainval_classes_file = f"{dataset_path}fgvc-aircraft-2013b/data/images_variant_trainval.txt"
                    files = [test_classes_file, trainval_classes_file]

                    class_per_image = {}
                    for f in files:
                        file = open(f, "r")
                        for line in file:
                            line = line.strip()

                            if not line:
                                continue

                            values = line.split(" ")
                            image_name = values[0]
                            class_name = values[1:]
                            class_name = " ".join(class_name)

                            class_per_image[image_name] = class_name
                    
                    images_per_class = {}
                    for image in _images:
                        image_name = image.split("/")[-1].removesuffix(".jpg")
                        class_name = class_per_image[image_name]

                        if class_name not in images_per_class:
                            images_per_class[class_name] = []
                        images_per_class[class_name].append(image)
                    
                    for _class in images_per_class:
                        images_per_class[_class] = sorted(images_per_class[_class])

                        n_val_images = ceil(len(images_per_class[_class]) * val_percent)

                        match self.operation:
                            case "train":
                                images_per_class[_class] = images_per_class[_class][:-n_val_images]
                            case "val":
                                images_per_class[_class] = images_per_class[_class][-n_val_images:]
                            case "all":
                                pass

                        self.images.extend(images_per_class[_class])

                case 'flowers-102':
                    image_labels_file = f"{dataset_path}flowers-102/imagelabels.mat"
                    labels = []
                    with open(image_labels_file, "rb") as file:
                        data = scipy.io.loadmat(file)
                        labels.extend(data.get('labels')[0])

                    _images = sorted(glob(dataset_path + "flowers-102/jpg/*.jpg"))
                    
                    images_per_class = {}
                    for i, image in enumerate(_images):
                        if labels[i] not in images_per_class:
                            images_per_class[labels[i]] = []
                        images_per_class[labels[i]].append(image)

                    for _class in images_per_class:
                        images_per_class[_class] = sorted(images_per_class[_class])

                        n_val_images = ceil(len(images_per_class[_class]) * val_percent)

                        match self.operation:
                            case "train":
                                images_per_class[_class] = images_per_class[_class][:-n_val_images]
                            case "val":
                                images_per_class[_class] = images_per_class[_class][-n_val_images:]
                            case "all":
                                pass

                        self.images.extend(images_per_class[_class])

                case 'food-101':
                    classes = os.listdir(dataset_path + "food-101/food-101/images/")
                    images_per_class = {}
                    for _class in classes:
                        _images = glob(dataset_path + f"food-101/food-101/images/{_class}/*.jpg")
                        images_per_class[_class] = sorted(_images)

                        n_val_images = ceil(len(images_per_class[_class]) * val_percent)

                        match self.operation:
                            case "train":
                                images_per_class[_class] = images_per_class[_class][:-n_val_images]
                            case "val":
                                images_per_class[_class] = images_per_class[_class][-n_val_images:]
                            case "all":
                                pass

                        self.images.extend(images_per_class[_class])

                case 'oxford-pets':
                    _images = glob(dataset_path + "oxford-iiit-pet/images/*.jpg")
                    images_per_class = {}
                    for image in _images:
                        image_name = image.split("/")[-1]
                        values = image_name.split("_")
                        values.pop(-1)
                        class_name = " ".join(values)

                        if class_name not in images_per_class:
                            images_per_class[class_name] = []
                        images_per_class[class_name].append(image)
                    
                    for _class in images_per_class:
                        images_per_class[_class] = sorted(images_per_class[_class])

                        n_val_images = ceil(len(images_per_class[_class]) * val_percent)

                        match self.operation:
                            case "train":
                                images_per_class[_class] = images_per_class[_class][:-n_val_images]
                            case "val":
                                images_per_class[_class] = images_per_class[_class][-n_val_images:]
                            case "all":
                                pass

                        self.images.extend(images_per_class[_class])

                case 'stanford-cars':
                    train_annos = f"{dataset_path}/car_devkit/devkit/cars_train_annos.mat"
                    test_annos = f"{dataset_path}/car_devkit/devkit/cars_test_annos.mat"
                    train_images_path = f"{dataset_path}/cars_train/cars_train"
                    test_images_path = f"{dataset_path}/cars_test/cars_test"

                    train_images = glob(train_images_path + "/*.jpg")
                    test_images = glob(test_images_path + "/*.jpg")

                    train_data = scipy.io.loadmat(train_annos)
                    test_data = scipy.io.loadmat(test_annos)

                    train_labels = train_data['annotations'][0]
                    test_labels = test_data['annotations'][0]

                    images_per_class = {}

                    class_per_image = {}
                    for item in train_labels:
                        image_name = item[-1][0]
                        image_class = item[-2][0][0]

                        class_per_image[image_name] = image_class
                    
                    for image in train_images:
                        image_name = image.split("/")[-1]
                        class_name = class_per_image[image_name]
                        if class_name not in images_per_class:
                            images_per_class[class_name] = []
                        images_per_class[class_name].append(image)
                    
                    class_per_image = {}
                    for item in test_labels:
                        image_name = item[-1][0]
                        image_class = item[-2][0][0]

                        class_per_image[image_name] = image_class

                    for image in test_images:
                        image_name = image.split("/")[-1]
                        class_name = class_per_image[image_name]
                        if class_name not in images_per_class:
                            images_per_class[class_name] = []
                        images_per_class[class_name].append(image)

                    for _class in images_per_class:
                        images_per_class[_class] = sorted(images_per_class[_class])

                        n_val_images = ceil(len(images_per_class[_class]) * val_percent)

                        match self.operation:
                            case "train":
                                images_per_class[_class] = images_per_class[_class][:-n_val_images]
                            case "val":
                                images_per_class[_class] = images_per_class[_class][-n_val_images:]
                            case "all":
                                pass

                        self.images.extend(images_per_class[_class])
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        image = read_image(image_path, ImageReadMode.RGB)

        if self.apply_data_augmentation:
            x1 = self.transform(image)
            x2 = self.transform(image)

            return x1, x2
        
        else:
            image = self.transform(image)
            return image
