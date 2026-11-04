import os
import gdown
import zipfile
import argparse
import shutil

# URLs for the datasets and pretrained models
datasets_url = "https://drive.google.com/uc?export=download&id=1h8dWZ88Xsu-OZLbIyXksXzZhThn8l6Rb"
single_model = "https://drive.google.com/uc?export=download&id=1l4R1G98yTKuNJSwrsmpUtlxDhRLbW09k"
cifar_adam_001_pretrained_url = "https://drive.google.com/uc?export=download&id=1XZFbW_zFT6CoLzZfZxZz_nOgvKtu3RIE"
cifar_adam_005_pretrained_url = "https://drive.google.com/uc?export=download&id=1SRYcSy3rLml60Xu4yVkRUqo_iU5gD4ko"
cifar_sgd_001_pretrained_url = "https://drive.google.com/uc?export=download&id=1DBb5qX1GhFzhrIaewAhmc37bKeTjeg5p"
lenet5_pretrained_url = "https://drive.google.com/uc?export=download&id=1Y6QLosaVKw_MChCmOfW7L6rqOYmky4Dn"
adam_001_wd_pretrained_url = "https://drive.google.com/uc?export=download&id=1_x6VQ_2LOVD9fhJ2rsljMZZOvVO64Ras"
imagenet_pretrained_url = "https://drive.google.com/uc?export=download&id=1o1n5d3qF3E49l7NxhDCuYQQ16Kyj91Fe"
# Mapping of argument names to URLs, target directories, and final filenames
download_options = {
    "datasets": (datasets_url, "./data", "datasets"),
    "single_model": (single_model, "./models/pretrained", "vgg16_cifar_standard.bin"),
    "cifar_adam_001": (cifar_adam_001_pretrained_url, "./models/pretrained", "Cifar_adam_001"),
    "cifar_adam_005": (cifar_adam_005_pretrained_url, "./models/pretrained", "Cifar_adam_005"),
    "cifar_sgd_001": (cifar_sgd_001_pretrained_url, "./models/pretrained", "Cifar_sgd_001"),
    "adam_001_wd": (adam_001_wd_pretrained_url, "./models/pretrained", "adam_001_wd"),
    "lenet": (lenet5_pretrained_url, "./models/pretrained", "MNIST_LeNet5"),
    "imagenet": (imagenet_pretrained_url, "./models/pretrained", "ImageNet")
    
}

def download_and_extract(url, target_dir, final_filename=None, is_zip=True):
    os.makedirs(target_dir, exist_ok=True)
    temp_dir = os.path.join(target_dir, "temp_extracted")
    os.makedirs(temp_dir, exist_ok=True)
    
    if is_zip:
        custom_zip_file = os.path.join(temp_dir, "tmp.zip")
        gdown.download(url, custom_zip_file, quiet=False)
        with zipfile.ZipFile(custom_zip_file, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)
        os.remove(custom_zip_file)
        
        if final_filename:
            # Move the extracted directory or file to the final filename
            extracted_items = os.listdir(temp_dir)
            for item in extracted_items:
                item_path = os.path.join(temp_dir, item)
                final_path = os.path.join(target_dir, final_filename)
                if os.path.exists(final_path):
                    if os.path.isdir(final_path):
                        shutil.rmtree(final_path)
                    else:
                        os.remove(final_path)
                shutil.move(item_path, final_path)
    else:
        file_name = os.path.join(target_dir, final_filename if final_filename else os.path.basename(url))
        gdown.download(url, file_name, quiet=False)

    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download specified files.")
    parser.add_argument("files", nargs="+", help="List of files to download (options: all, datasets, single_model, cifar_adam_001, cifar_adam_005, cifar_sgd_001, lenet, adam_001_wd)")

    args = parser.parse_args()
    files_to_download = args.files

    if "all" in files_to_download:
        files_to_download = download_options.keys()

    for file_key in files_to_download:
        if file_key in download_options:
            url, target_dir, final_filename = download_options[file_key]
            is_zip = not (file_key == "single_model")
            download_and_extract(url, target_dir, final_filename, is_zip)
        else:
            print(f"Unknown file specified: {file_key}")

    if "all" or "imagenet" in args.files:
        #move files to the correct location
        if os.path.exists("./models/pretrained/ImageNet/imagenet_128-256_labels.bin"):
            os.makedirs("./data/datasets/", exist_ok=True)
            shutil.move("./models/pretrained/ImageNet/imagenet_128-256_labels.bin", "./data/datasets/imagenet_128-256_labels.bin")
        if os.path.exists("./models/pretrained/ImageNet/imagenet_128-256_images.bin"):
            os.makedirs("./data/datasets/", exist_ok=True)
            shutil.move("./models/pretrained/ImageNet/imagenet_128-256_images.bin", "./data//datasets/imagenet_128-256_images.bin")

    print("Download and extraction complete.")

