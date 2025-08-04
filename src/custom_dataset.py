from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, ImageReadMode
from glob import glob
from math import ceil
import os
from random import shuffle
import torchvision.transforms as v2
import scipy.io

CIFAR_VAL_PERCENT = 0.1

class custom_dataset(Dataset):
    def __init__(self, operation, apply_data_augmentation, datasets, datasets_folder_path, transform):
        if operation not in ["train", "val", "test"]:
            raise ValueError("Operation must be 'train', 'val', or 'test'.")
        
        self.operation = operation
        self.apply_data_augmentation = apply_data_augmentation
        self.transform = transform

        self.images = []
        self.labels = []

        for dataset in datasets:
            dataset_path = f"{datasets_folder_path}{dataset}/"

            match dataset:
                case 'cifar100' | 'cifar10':
                    '''
                    cifar100 and cifar10 don't have a validation set, so we use 10% of each class's training set (last 10% ordered images) as validation
                    '''
                    classes = sorted(os.listdir(dataset_path + "train/"))
                    images_per_class = {}
                    for class_idx, _class in enumerate(classes):
                        images_per_class[_class] = []
                        
                        match self.operation:
                            case "train":
                                images_per_class[_class].extend(glob(dataset_path + "train/" + _class + "/*.png"))
                                n_val_images = ceil(len(images_per_class[_class]) * CIFAR_VAL_PERCENT)
                                images_per_class[_class] = sorted(images_per_class[_class])
                                images_per_class[_class] = images_per_class[_class][:-n_val_images]
                            case "val":
                                images_per_class[_class].extend(glob(dataset_path + "train/" + _class + "/*.png"))
                                n_val_images = ceil(len(images_per_class[_class]) * CIFAR_VAL_PERCENT)
                                images_per_class[_class] = sorted(images_per_class[_class])
                                images_per_class[_class] = images_per_class[_class][-n_val_images:]
                            case "test":
                                images_per_class[_class].extend(glob(dataset_path + "test/" + _class + "/*.png"))


                        self.images.extend(images_per_class[_class])
                        self.labels.extend([class_idx] * len(images_per_class[_class]))

                case 'dtd':
                    labels_path = dataset_path + "dtd/dtd/labels/"
                    images_path = dataset_path + "dtd/dtd/images/"
                    train_labels = open(labels_path + "train1.txt", "r")
                    val_labels = open(labels_path + "val1.txt", "r")
                    test_labels = open(labels_path + "test1.txt", "r")

                    image_to_split = {}
                    for line in train_labels:
                        line = line.strip()
                        if not line:
                            continue
                        image_to_split[line] = "train"
                    
                    for line in val_labels:
                        line = line.strip()
                        if not line:
                            continue
                        image_to_split[line] = "val"
                    
                    for line in test_labels:
                        line = line.strip()
                        if not line:
                            continue
                        image_to_split[line] = "test"

                    classes = sorted(os.listdir(dataset_path + "dtd/dtd/images/"))
                    class_to_id = {cls: idx for idx, cls in enumerate(classes)}

                    for _class in classes:
                        class_id = class_to_id[_class]

                        _images = glob(images_path + _class + "/*.jpg")

                        for image in _images:
                            image_name = image.split("/")[-2:]
                            image_name = "/".join(image_name)
                            
                            if image_to_split.get(image_name) == self.operation:
                                self.images.append(image)
                                self.labels.append(class_id)

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
                    
                    classes = set(class_per_image.values())
                    classes = sorted(list(classes))
                    class_to_id = {cls: idx for idx, cls in enumerate(classes)}

                    train_images = open(f"{dataset_path}fgvc-aircraft-2013b/data/images_train.txt", "r")
                    val_images = open(f"{dataset_path}fgvc-aircraft-2013b/data/images_val.txt", "r")
                    test_images = open(f"{dataset_path}fgvc-aircraft-2013b/data/images_test.txt", "r")

                    image_to_split = {}
                    for line in train_images:
                        line = line.strip()
                        if not line:
                            continue
                        image_to_split[line] = "train"

                    for line in val_images:
                        line = line.strip()
                        if not line:
                            continue
                        image_to_split[line] = "val"

                    for line in test_images:
                        line = line.strip()
                        if not line:
                            continue
                        image_to_split[line] = "test"

                    for image in glob(dataset_path + "fgvc-aircraft-2013b/data/images/*.jpg"):
                        image_name = image.split("/")[-1].removesuffix(".jpg")
                        
                        if image_to_split.get(image_name) == self.operation:
                            class_name = class_per_image.get(image_name, None)
                            class_id = class_to_id.get(class_name)

                            self.images.append(image)
                            self.labels.append(class_id)

                case 'flowers-102':
                    image_labels_file = f"{dataset_path}flowers-102/imagelabels.mat"
                    setId_file = f"{dataset_path}flowers-102/setid.mat"

                    labels = []
                    with open(image_labels_file, "rb") as file:
                        data = scipy.io.loadmat(file)
                        labels.extend(data.get('labels')[0])

                    setId_file = scipy.io.loadmat(setId_file)
                    train_ids = setId_file.get('trnid')[0]
                    val_ids = setId_file.get('valid')[0]
                    test_ids = setId_file.get('tstid')[0]

                    image_to_split = {}
                    for idx in train_ids:
                        image_to_split[idx] = "train"
                    for idx in val_ids:
                        image_to_split[idx] = "val"
                    for idx in test_ids:
                        image_to_split[idx] = "test"

                    _images = sorted(glob(dataset_path + "flowers-102/jpg/*.jpg"))
                    
                    for i, image in enumerate(_images):
                        image_name = image.split("/")[-1].removesuffix(".jpg")
                        image_id = int(image_name.split("_")[-1])
                        
                        if image_to_split.get(image_id) == self.operation:
                            class_label = labels[i] - 1

                            self.images.append(image)
                            self.labels.append(class_label)

                case 'food-101':
                    classes_path = open(f"{dataset_path}food-101/food-101/meta/classes.txt", "r")
                    classes = []
                    for line in classes_path:
                        line = line.strip()
                        if not line:
                            continue
                        classes.append(line)
                    class_to_id = {cls: idx for idx, cls in enumerate(classes)}

                    train_images = open(f"{dataset_path}food-101/food-101/meta/train.txt", "r")
                    test_images = open(f"{dataset_path}food-101/food-101/meta/test.txt", "r")
                    image_to_split = {}
                    for line in train_images:
                        line = line.strip()
                        if not line:
                            continue
                        image_to_split[line] = "train"

                    for line in test_images:
                        line = line.strip()
                        if not line:
                            continue
                        image_to_split[line] = "test"

                    for _class in classes:
                        _images = sorted(glob(dataset_path + f"food-101/food-101/images/{_class}/*.jpg"))

                        train_images = []
                        test_images = []
                        for image in _images:
                            image_name = image.split("/")[-2:]
                            image_name = "/".join(image_name).removesuffix(".jpg")

                            if image_to_split.get(image_name) == "train":
                                train_images.append(image)
                            elif image_to_split.get(image_name) == "test":
                                test_images.append(image)
                        
                        val_images_n = ceil(len(train_images) * 0.1)
                        val_images = train_images[-val_images_n:]
                        train_images = train_images[:-val_images_n]

                        match self.operation:
                            case "train":
                                self.images.extend(train_images)
                                self.labels.extend([class_to_id[_class]] * len(train_images))
                            case "val":
                                self.images.extend(val_images)
                                self.labels.extend([class_to_id[_class]] * len(val_images))
                            case "test":
                                self.images.extend(test_images)
                                self.labels.extend([class_to_id[_class]] * len(test_images))

                case 'oxford-pets':
                    test_file = open(f"{dataset_path}oxford-iiit-pet/annotations/test.txt", "r")
                    trainval_file = open(f"{dataset_path}oxford-iiit-pet/annotations/trainval.txt", "r")

                    image_to_split = {}
                    classes = set()

                    for line in trainval_file:
                        line = line.strip()
                        if not line:
                            continue
                        image_name = line.split(" ")[0]
                        class_name = image_name.split("_")[-5:-1]
                        class_name = "_".join(class_name)
                        image_to_split[image_name] = "train"
                        classes.add(class_name)
                    
                    class_to_id = {cls: idx for idx, cls in enumerate(sorted(classes))}

                    for line in test_file:
                        line = line.strip()
                        if not line:
                            continue
                        image_name = line.split(" ")[0]
                        image_to_split[image_name] = "test"

                    _images = glob(dataset_path + "oxford-iiit-pet/images/*.jpg")

                    images_per_class = {}
                    for image in _images:
                        image_name = image.split("/")[-1]
                        values = image_name.split("_")
                        values.pop(-1)
                        class_name = "_".join(values)

                        if class_name not in images_per_class:
                            images_per_class[class_name] = []
                        images_per_class[class_name].append(image)
                    
                    for _class in images_per_class:
                        images_per_class[_class] = sorted(images_per_class[_class])

                        train_images = []
                        test_images = []
                        for image in images_per_class[_class]:
                            image_name = image.split("/")[-1].removesuffix(".jpg")
                            
                            if image_to_split.get(image_name) == "train":
                                train_images.append(image)
                            elif image_to_split.get(image_name) == "test":
                                test_images.append(image)

                        n_val_images = ceil(len(images_per_class[_class]) * 0.1)

                        val_images = train_images[-n_val_images:]
                        train_images = train_images[:-n_val_images]

                        match self.operation:
                            case "train":
                                self.images.extend(train_images)
                                self.labels.extend([class_to_id[_class]] * len(train_images))
                            case "val":
                                self.images.extend(val_images)
                                self.labels.extend([class_to_id[_class]] * len(val_images))
                            case "test":
                                self.images.extend(test_images)
                                self.labels.extend([class_to_id[_class]] * len(test_images))

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

                    train_images_per_class = {}

                    train_class_per_image = {}
                    for item in train_labels:
                        image_name = item[-1][0]
                        image_class = item[-2][0][0]

                        train_class_per_image[image_name] = image_class
                    
                    for image in train_images:
                        image_name = image.split("/")[-1]
                        class_name = train_class_per_image[image_name]
                        if class_name not in train_images_per_class:
                            train_images_per_class[class_name] = []
                        train_images_per_class[class_name].append(image)

                    test_images_per_class = {}

                    test_class_per_image = {}
                    for item in test_labels:
                        image_name = item[-1][0]
                        image_class = item[-2][0][0]

                        test_class_per_image[image_name] = image_class

                    for image in test_images:
                        image_name = image.split("/")[-1]
                        class_name = test_class_per_image[image_name]
                        if class_name not in test_images_per_class:
                            test_images_per_class[class_name] = []
                        test_images_per_class[class_name].append(image)

                    if self.operation == "test":
                        for _class in test_images_per_class:
                            self.images.extend(test_images_per_class[_class])
                            self.labels.extend([int(_class) - 1] * len(test_images_per_class[_class]))
                    else:
                        for _class in train_images_per_class:
                            class_images = train_images_per_class[_class]
                            class_images = sorted(class_images)

                            n_val_images = ceil(len(class_images) * 0.1)
                            val_images = class_images[-n_val_images:]
                            train_images = class_images[:-n_val_images]

                            match self.operation:
                                case "train":
                                    self.images.extend(train_images)
                                    self.labels.extend([int(_class) - 1] * len(train_images))
                                case "val":
                                    self.images.extend(val_images)
                                    self.labels.extend([int(_class) - 1] * len(val_images))

                case 'birdsnap':
                    pass

                case 'caltech101':
                    pass

                case 'caltech256':
                    pass
                    
                case 'imagenet':
                    validation_gd_path = f"{dataset_path}ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt"
                    meta_path = f"{dataset_path}ILSVRC2012_devkit_t12/data/meta.mat"

                    data = scipy.io.loadmat(open(meta_path, "rb"))
                    synsets = data.get('synsets')

                    id_to_wnid = {}
                    wnid_to_class = {}
                    for synset in synsets:
                        if int(synset['num_children'][0][0][0]) == 0:
                            wnid = synset['WNID'][0][0]
                            words = synset['words'][0][0]

                            wnid_to_class[wnid] = words
                            id_to_wnid[int(synset['ILSVRC2012_ID'][0][0][0])] = wnid

                    classes = sorted(wnid_to_class.values())
                    class_to_id = {cls: idx for idx, cls in enumerate(classes)}

                    match self.operation:
                        case "train":
                            train_wnid = os.listdir(dataset_path + "train/")
                            for wnid in train_wnid:
                                _images = glob(dataset_path + "train/" + wnid + "/*.JPEG")
                                self.images.extend(_images)
                                class_name = wnid_to_class[wnid]
                                class_id = class_to_id[class_name]
                                self.labels.extend([class_id] * len(_images))

                        case "val":
                            val_images = sorted(glob(dataset_path + "val/*.JPEG"))

                            idx_to_wnid = {}
                            with open(validation_gd_path, "r") as file:
                                for idx, line in enumerate(file):
                                    line = line.strip()
                                    if not line:
                                        continue
                                    wnid = id_to_wnid[int(line)]
                                    idx_to_wnid[idx] = wnid
                            
                            for idx, image in enumerate(val_images):
                                wnid = idx_to_wnid[idx]

                                self.images.append(image)
                                class_name = wnid_to_class[wnid]
                                class_id = class_to_id[class_name]
                                self.labels.append(class_id)

                        case "test":
                            test_images = sorted(glob(dataset_path + "test/*.JPEG"))
                            self.images.extend(test_images)
                            self.labels.extend([None] * len(test_images))  # Labels are not available

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_path = self.images[idx]

        image = read_image(image_path, ImageReadMode.RGB)

        if self.apply_data_augmentation:
            x1 = self.transform(image)
            x2 = self.transform(image)

            return x1, x2
        
        image = self.transform(image)
        return image, self.labels[idx]
