import os
import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import vgg16, VGG16_Weights
import numpy as np
import struct
import os

models = [
        alexnet(weights=AlexNet_Weights.DEFAULT),
        vgg16(weights=VGG16_Weights.DEFAULT)
        ]

# Define transforms
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load ImageNet validation dataset
batch_size = 32
first_image = 128
last_image = 256

val_dir = './data/ILSVRC2012_val'
val_dataset = ImageFolder(val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')
for model in models:
    model.to(device)
    model.eval()

# Check if accuracy is correct by testing 5 batches
    correct = 0
    total = 0

    with torch.no_grad():
        counter = 0
        for images, labels in val_loader:
            #skip first 4 batches to avoid 0 labels
            counter += 1
            if counter > first_image/batch_size:

                images = images.to(device)
                labels = labels.to(device)
                
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print("Label: ", labels, "Predicted: ", predicted)
            if counter == last_image/batch_size:
                break

        accuracy = 100 * correct / total
        # print(f'Accuracy of {model.__class__.__name__} on 128 images: {accuracy:.2f}%')
        print(f'Accuracy of {model.__class__.__name__} on images {first_image}-{last_image} of the validation dataset: {accuracy:.2f}%')


def save_weights_compatible_with_cpp(model, filepath):
    with open(filepath, 'wb') as f:
        all_params = []
        # Process each module according to its type
        for module in model.modules():
            # print(module)
            if isinstance(module, torch.nn.Linear):
                all_params.append(module.weight.data.cpu().numpy().ravel())
                if module.bias is not None:
                    all_params.append(module.bias.data.cpu().numpy().ravel())
            elif isinstance(module, torch.nn.Conv2d):
                all_params.append(module.weight.data.cpu().numpy().ravel())
                if module.bias is not None:
                    all_params.append(module.bias.data.cpu().numpy().ravel())
            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                all_params.append(module.running_mean.cpu().numpy().ravel())
                all_params.append(module.running_var.cpu().numpy().ravel())
                all_params.append(module.weight.data.cpu().numpy().ravel())
                all_params.append(module.bias.data.cpu().numpy().ravel())

        # Flatten all parameters and convert to numpy array
        all_params_flat = np.concatenate(all_params).astype(np.float32)

        # Write total number of parameters and then write the parameters
        f.write(np.array([all_params_flat.size], dtype=np.int32).tobytes())
        f.write(all_params_flat.tobytes())

# Function to export dataset
def export_dataset(dataset, images_filename, labels_filename):
    with open(images_filename, 'wb') as img_file, open(labels_filename, 'wb') as lbl_file:
        counter = 0
        for image, label in dataset:
            counter += 1
            if counter > first_image:
                # Convert image to numpy array, ensure it's float32, and flatten
                img_data = image.numpy().astype(np.float32).flatten()

                # Write the flattened image data to the binary file
                img_file.write(struct.pack('f' * len(img_data), *img_data))

                # Ensure label is an int and write to the binary file
                # print(label)
                lbl_data = np.array(label).astype(np.uint32)
                lbl_file.write(lbl_data.tobytes())
            if counter == last_image:
                return

# Export the dataset
# os.makedirs('data/datasets', exist_ok=True)
# os.makedirs('models/pretrained/ImageNet', exist_ok=True)
# export_dataset(val_dataset, 'data/datasets/imagenet_128-256_images.bin', 'data/datasets/imagenet_128-256_labels.bin')
# print('Exported dataset to data/datasets/imagenet_128-256_images.bin and data/datasets/imagenet_128-256_labels.bin')
# for model in models:
#     save_weights_compatible_with_cpp(model, f'models/pretrained/ImageNet/{model.__class__.__name__}_imagenet.bin')
#     print(f'Saved weights for {model.__class__.__name__} to models/pretrained/{model.__class__.__name__}_imagenet.bin')
