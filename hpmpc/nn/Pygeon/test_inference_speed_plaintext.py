import torch
import time
from torch.utils.data import DataLoader, TensorDataset
from torchvision.models import alexnet, AlexNet_Weights
from torchvision.models import vgg16, VGG16_Weights
from tqdm import tqdm
import cnns

# List of models
cifar_models = [
        cnns.AlexNet(num_classes=10),
        cnns.VGG16(num_classes=10),
        cnns.ResNet18_avg(num_classes=10),
        cnns.ResNet50_avg(num_classes=10),
        cnns.ResNet101_avg(num_classes=10),
        cnns.ResNet152_avg(num_classes=10)
]


imagenet_models = [
        cnns.AlexNet(num_classes=1000),
        cnns.VGG16(num_classes=1000),
        cnns.ResNet18_avg(num_classes=1000),
        cnns.ResNet50_avg(num_classes=1000),
        cnns.ResNet101_avg(num_classes=1000),
        cnns.ResNet152_avg(num_classes=1000),
        alexnet(), # predefined PyTorch models
        vgg16() # predefined PyTorch models
        
        
]
batch_size = 192

# Set device (GPU or CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Using device: {device}')

def reset_weights(model):
    for layer in model.children():
        if hasattr(layer, 'reset_parameters'):
            layer.reset_parameters()

# Reset weights of the models
for model in imagenet_models:
    reset_weights(model)

# Generate 10,000 dummy images (3 channels, 224x224) and random labels (1000 classes)
num_images = 10000
image_size_imagenet = (3, 224, 224)  # Channels, Height, Width
num_classes_imagenet = 1000

image_size_cifar = (3, 32, 32)  # Channels, Height, Width
num_classes_cifar = 10

# Create random dummy images and labels
dummy_images_imagenet = torch.randn(num_images, *image_size_imagenet)
dummy_labels_imagenet = torch.randint(0, num_classes_imagenet, (num_images,))
dummy_images_cifar = torch.randn(num_images, *image_size_cifar)
dummy_labels_cifar = torch.randint(0, num_classes_cifar, (num_images,))

# Create a dataset from the dummy data
dummy_dataset_imagenet = TensorDataset(dummy_images_imagenet, dummy_labels_imagenet)
dummy_loader_imagenet = DataLoader(dummy_dataset_imagenet, batch_size=batch_size, shuffle=False, num_workers=4)
dummy_dataset_cifar = TensorDataset(dummy_images_cifar, dummy_labels_cifar)
dummy_loader_cifar = DataLoader(dummy_dataset_cifar, batch_size=batch_size, shuffle=False, num_workers=4)


# Iterate over models and perform inference
for model in cifar_models:
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    # Start timing the inference
    start_time = time.time()

    with torch.no_grad():
        for images, labels in tqdm(dummy_loader_cifar, desc=f'Running inference on {model.__class__.__name__}'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Stop timing
    end_time = time.time()
    inference_time = end_time - start_time

    print(f'Inference time for {model.__class__.__name__} on 10,000 CIFAR-10 images: {inference_time:.2f} seconds')


# Iterate over models and perform inference
for model in imagenet_models:
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    # Start timing the inference
    start_time = time.time()

    with torch.no_grad():
        for images, labels in tqdm(dummy_loader_imagenet, desc=f'Running inference on {model.__class__.__name__}'):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Stop timing
    end_time = time.time()
    inference_time = end_time - start_time

    print(f'Inference time for {model.__class__.__name__} on 10,000 ImageNet images: {inference_time:.2f} seconds')

