# PyGEON

PyGEON exports models and datasets from PyTorch to `.bin` files compatible with [PIGEON](https://github.com/chart21/hpmpc). 
The models can either be trained from scratch with PyGEON or loaded from existing `.pth` files.
We provide pretrained models and datasets for popular architectures and datasets such as VGG16, various ResNets, AlexNe on CIFAR10/MNIST.
New models and datasets can be added to `cnns.py` and `datasets.py`, respectively.


## Requirements

PyGEON requires PyTorch to train models and gdown to download pretrained models and datasets.
To install PyTorch, follow the instructions on the [official website](https://pytorch.org/get-started/locally/).
Our versions are contained in the `requirements.txt` file.
The following commands may suffice to install the dependencies:

```sh
pip install torch torchvision # required for exporting models and datasets
pip install gdown # required for downloading pretrained models and datasets
```



## Train and export models from PyTorch

PyGEON can train models from sractch or load existing models from `.pth` files. The trained models can be exported to `.bin` files compatible with PIGEON. The resulting model gets stored in the `./models` directory. 
Similarly, datasets can be downloaded and transformed in PyTorch and exported to `.bin` files compatible with PIGEON. The transformed dataset gets stored in the `./data` directory.

### Examples


Train a model and export both the model and the transformed dataset to a `.bin` file compatible with PIGEON:

```sh
python main.py --action train --export_model --export_dataset --transform standard --model VGG16 --num_classes 10 --dataset_name CIFAR-10 --modelpath ./models/vgg16_cifar --num_epochs 40 --lr 0.001 --criterion CrossEntropyLoss --optimizer Adam
```

Load an existing model from a `.pth` file and export it to a `.bin` file compatible with PIGEON:

```sh
python main.py --action import --export_model --modelpath ./models/my_pretrained_model --model ResNet18_avg --num_classes 10
```

Train all predefined models and export them to `.pth` files.

```sh
python main.py --action train_all --num_epochs 40 --lr 0.001 --criterion CrossEntropyLoss --optimizer Adam
```


### Command-Line Arguments

- `--model`: Model to use (default: `LeNet5`)
- `--num_classes`: Number of classes (default: 10)
- `--dataset_name`: Dataset name (default: `MNIST`)
- `--modelpath`: Path to save/load the model (default: `./models/lenet5_mnist`)
- `--num_epochs`: Number of epochs (default: 80)
- `--lr`: Learning rate (default: 0.01)
- `--criterion`: Loss function (default: `CrossEntropyLoss`)
- `--optimizer`: Optimizer (default: `Adam`)
- `--dropout`: Dropout rate (default: 0.0)
- `--weight_decay`: Weight decay (default: 0.0)
- `--batch_size`: Batch size (default: 32)
- `--action`: Action to perform on the model (choices: `train`, `import`, `train_all`, `test`, `none`; default: `train`)
- `--export_model`: Export the model as a .bin file for PIGEON (flag)
- `--export_dataset`: Export the test dataset as a .bin file for PIGEON (flag)
- `--transform`: Type of transformation to apply (choices: `custom`, `standard`; default: `standard`)


## Download pretrained models and datasets

We provide pretrained models for popular architectures such as VGG16, various ResNets, AlexNet and transformed datasets such as CIFAR10, and MNIST.

### Examples

1. Download all files:
    ```sh
    python download_files.py all
    ```

2. Download specific files:
    ```sh
    python download_files.py datasets lenet single_model
    ```


Datasets get saved in the `./data` directory, and models get saved in the `./models/pretrained` directory.

### ImageNet Models and Datasets

The `imagenet_files.py` script exports PyTorch's pretrained VGG16 and AlexNet models and converts 128 images from the ImageNet validation dataset to `.bin` files compatible with PIGEON. To get started, download the ImageNet ILSCVR2012 validation dataset from the [official website](http://www.image-net.org/challenges/LSVRC/2012/), extract the downloaded `.tar.gz` file and move the `ILSVRC2012_val` directory to the `./data` directory. Then, run the following commands to save the models and datasets to the `./models/pretrained` and `./data` directories, respectively.

```sh
cp valprep.sh ./data/ILSVRC2012_val/
cd ./data/ILSVRC2012_val/
./valprep.sh
cd ../..
python3 imagenet_files.py
```

The script performs a few test inferences in PyTorch to ensure that the dataset was perpared correctly. The resulting models can then be used for inference with PIGEON. Execute the following commands from the `hpmpc` base directory to verify the accuracy of the models:
```sh
export MODEL_DIR=nn/Pygeon/models/pretrained/ImageNet
export DATA_DIR=nn/Pygeon/data/datasets
export SAMPLES_FILE=imagenet_128-256_images.bin
export LABELS_FILE=imagenet_128-256_labels.bin
export MODEL_FILE=AlexNet_imagenet.bin # use pretrained AlexNet
# export MODEL_FILE=VGG_imagenet.bin # use pretrained VGG16
make -j PARTY=all FUNCTION_IDENTIFIER=85 NUM_INPUTS=10 DATTYPE=64 BITLENGTH=64 FRACTIONAL=14 TRUNC_APPROACH=1 PROTOCOL=5 MODELOWNER=P_0 DATAOWNER=P_1 # Test 10 inferences of AlexNet on ImageNet, use FUNCTION_IDENTIFIER=86 for VGG16
scripts/run.sh -p all -n 3
```

If you achieve 70% accuracy with AlexNet, the setup is correct.

### Command-Line Arguments

The script accepts a list of files to download as command-line arguments. 
Available options are:

- `all`: Downloads and extracts all datasets and models.
- `datasets`: Downloads all datasets.
- `single_model`: Downloads a single VGG16 pretrained model.
- `lenet`: Downloads LeNetmodels pretrained on MNIST.
- `cifar_adam_001`: Downloads various models pretrained with CIFAR Adam optimizer (lr=0.001).
- `cifar_adam_005`: Downloads various models pretrained with CIFAR Adam optimizer (lr=0.005).
- `cifar_sgd_001`: Downloads various models pretrained with CIFAR SGD optimizer (lr=0.001).


