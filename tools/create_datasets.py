import sys
import os
import kagglehub
from torchvision.datasets import OxfordIIITPet, Flowers102, FGVCAircraft, DTD

def main():
    if len(sys.argv) != 2:
        print("Usage: python create_datasets.py <output_path>")
        sys.exit(1)

    output_path = sys.argv[1]

    if os.path.exists(output_path):
        print(f"Output path '{output_path}' already exists. Please choose a different path.")
        sys.exit(1)
    
    output_path += "/" if not output_path.endswith('/') else ""
    os.makedirs(output_path)
    
    # CIFAR100 and CIFAR10
    os.system(f"cifar2png cifar100 {output_path}cifar100")
    os.system(f"cifar2png cifar10 {output_path}cifar10")
    os.system(f"rm ./*.tar.gz")

    # Food101
    os.makedirs(output_path + "food-101", exist_ok=True)
    path = kagglehub.dataset_download("dansbecker/food-101")
    os.system(f"mv {path}/* {output_path}food-101/")

    # Stanford Cars
    os.makedirs(output_path + "stanford-cars", exist_ok=True)
    path = kagglehub.dataset_download("eduardo4jesus/stanford-cars-dataset")
    os.system(f"mv {path}/* {output_path}stanford-cars/")

    # Oxford Pets
    os.makedirs(output_path + "oxford-pets", exist_ok=True)
    _ = OxfordIIITPet(root=output_path + "oxford-pets", download=True)

    # Flowers102
    os.makedirs(output_path + "flowers-102", exist_ok=True)
    _ = Flowers102(root=output_path + "flowers-102", download=True)

    # FGVCAircraft
    os.makedirs(output_path + "fgvc-aircraft", exist_ok=True)
    _ = FGVCAircraft(root=output_path + "fgvc-aircraft", download=True)

    # DTD
    os.makedirs(output_path + "dtd", exist_ok=True)
    _ = DTD(root=output_path + "dtd", download=True)

    # Get ImageNet 2012 from the ImageNet website and rename it to imagenet
    # Get Caltech-101 from https://data.caltech.edu/records/mzrjq-6wc02

if __name__ == "__main__":
    main()
