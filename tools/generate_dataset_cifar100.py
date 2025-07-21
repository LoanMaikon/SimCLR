from glob import glob
import shutil
from math import ceil
import sys
import os

def main(cifar100_folder, val_percentage, output_folder):
    os.makedirs(output_folder + "cifar100_splitted", exist_ok=True)

    os.system(f"cp -r {cifar100_folder}test {output_folder}cifar100_splitted")

    for class_folder in os.listdir(cifar100_folder + "train"):
        class_images = sorted(glob(f"{cifar100_folder}train/{class_folder}/**/*.png", recursive=True))

        num_val_images = ceil(len(class_images) * val_percentage)

        os.makedirs(f"{output_folder}cifar100_splitted/train/{class_folder}", exist_ok=True)
        os.makedirs(f"{output_folder}cifar100_splitted/val/{class_folder}", exist_ok=True)

        for image in class_images[:-num_val_images]:
            shutil.copy(image, f"{output_folder}cifar100_splitted/train/{class_folder}")

        for image in class_images[-num_val_images:]:
            shutil.copy(image, f"{output_folder}cifar100_splitted/val/{class_folder}")

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python generate_dataset_cifar100.py <cifar100_folder> <val_percentage> <output_folder>")
        sys.exit(1)

    cifar100_folder = sys.argv[1]
    val_percentage = float(sys.argv[2])
    output_folder = sys.argv[3]

    cifar100_folder += "/" if not cifar100_folder.endswith('/') else ""
    output_folder += "/" if not output_folder.endswith('/') else ""

    main(cifar100_folder, val_percentage, output_folder)
