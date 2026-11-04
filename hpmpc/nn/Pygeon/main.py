#main function
import cnns
import data_load
import train
import parameter_export
import torch
import numpy as np
import os
import argparse
from torch import nn
from torch import optim
# from deepreduce_models.resnet import *

def save_model(model,filepath):
    parameter_export.save_weights_compatible_with_cpp(model, filepath+'.bin')

def export_pth_model(model,filepath):
    torch.save(model.state_dict(), filepath+'.pth')

def train_model(model,dataset_name,num_epochs=80,lr=0.001, transform = "standard", criterion=nn.CrossEntropyLoss, optimizer=optim.Adam, weight_decay=0.0, dropout=0.0, batch_size=32):
    train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name, transform, batch_size)
    train.train_and_evaluate(model, train_loader, test_loader, num_epochs, optimizer, lr, criterion, weight_decay, dropout)

def train_test_model(model,dataset_name,num_epochs=80,lr=0.001, transform = "standard", model_name="LeNet5", criterion=nn.CrossEntropyLoss, optimizier=optim.Adam, weight_decay=0.0, dropout=0.0, batch_size=32):
    train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name, transform, batch_size)
    train.train_test(model, train_loader, test_loader, num_epochs, optimizier, lr, criterion,  model_name, dataset_name, transform, weight_decay, dropout)


def load_model(model,filepath):
    if os.path.exists(filepath+'.bin'):
        parameter_export.read_params(model, filepath+'.bin')
    elif os.path.exists(filepath+'.pth'):
        model.load_state_dict(torch.load(filepath+'.pth', map_location=torch.device('cpu')))
    elif os.path.exists(filepath+'.pth.tar'):
        load_checkpoint(model,filepath)
    else:
        print("Could not find {}.bin or {}.pth or {}.pth.tar".format(filepath,filepath,filepath))

def load_checkpoint(model,filepath):
    checkpoint = torch.load(filepath+'.pth.tar', map_location=torch.device('cpu'))
    model.load_state_dict({k.replace('module.',''): v for k, v in checkpoint['snet'].items()})


def test_model(model,dataset_name,transform="standard",batch_size=32, criterion=nn.CrossEntropyLoss, test_batches=-1):
    train_loader, test_loader,num_classes = data_load.load_dataset(dataset_name, transform, batch_size)
    train.evaluate(model, test_loader, criterion_class=nn.CrossEntropyLoss, test_batches=test_batches)

def export_test_dataset(dataset_name,transform="standard",batch_size=32):
    train_set, test_set,num_classes = data_load.load_dataset(dataset_name, transform, batch_size)
    data_load.export_dataset(test_set,'./data/'+dataset_name+'_test_images.bin','./data/'+dataset_name+'_test_labels.bin')

def train_all(num_epochs=80,lr=0.001,criterion=nn.CrossEntropyLoss,optimizer=optim.Adam, weight_decay=0.0, dropout=0.0, batch_size=32):
    models_mnist = [(cnns.LeNet5(num_classes=10,dropout=dropout),"LeNet5")]
    models_cifar10 = [(cnns.AlexNet(num_classes=10,dropout=dropout),"AlexNet_CryptGPU"),(cnns.AlexNet_32(num_classes=10,dropout=dropout),"AlexNet_32"),(cnns.VGG16(num_classes=10,dropout=dropout),"VGG16"),(cnns.ResNet18_avg(num_classes=10,dropout=dropout),"ResNet18_avg"),(cnns.ResNet50_avg(num_classes=10,dropout=dropout),"ResNet50_avg"),(cnns.ResNet101_avg(num_classes=10,dropout=dropout),"ResNet101_avg"),(cnns.ResNet152_avg(num_classes=10,dropout=dropout),"ResNet152_avg"),(cnns.ResNet50(num_classes=10,dropout=dropout),"ResNet50"),(cnns.ResNet101(num_classes=10,dropout=dropout),"ResNet101"),(cnns.ResNet152(num_classes=10,dropout=dropout),"ResNet152")]   
    models_cifar100 = [(cnns.AlexNet_32(num_classes=100,dropout=dropout),"AlexNet_32"),(cnns.ResNet18_avg(num_classes=100,dropout=dropout),"ResNet18_avg"),(cnns.ResNet50_avg(num_classes=100,dropout=dropout),"ResNet50_avg"),(cnns.ResNet101_avg(num_classes=100,dropout=dropout),"ResNet101_avg"),(cnns.ResNet152_avg(num_classes=100,dropout=dropout),"ResNet152_avg"),(cnns.ResNet50(num_classes=100,dropout=dropout),"ResNet50"),(cnns.ResNet101(num_classes=100,dropout=dropout),"ResNet101"),(cnns.ResNet152(num_classes=100,dropout=dropout),"ResNet152")]
    #for model, model_name in models_mnist:
    #    transform = "standard"
    #    train_test_model(model,"MNIST", num_epochs, lr, transform, model_name, criterion, optimizer, weight_decay, dropout, batch_size)
    #    # train_model(model,"MNIST", num_epochs, lr, transform, criterion, optimizer, weight_decay, dropout, batch_size)
    #    transform = "custom"
    #    train_test_model(model,"MNIST", num_epochs, lr, transform, model_name, criterion, optimizer, weight_decay, dropout, batch_size)
    #    # train_model(model,"MNIST", num_epochs, lr, transform, criterion, optimizer, weight_decay, dropout, batch_size)
    #for model, model_name in models_cifar10:
    #    transform = "standard"
    #    train_test_model(model,"CIFAR-10", num_epochs, lr, transform, model_name, criterion, optimizer, weight_decay, dropout, batch_size)
    #    # train_model(model,"CIFAR-10", num_epochs, lr, transform, criterion, optimizer, weight_decay, dropout, batch_size)
    #    transform = "custom"
    #    train_test_model(model,"CIFAR-10", num_epochs, lr, transform, model_name, criterion, optimizer, weight_decay, dropout, batch_size)
    #    # train_model(model,"CIFAR-10", num_epochs, lr, transform, criterion, optimizer, weight_decay, dropout, batch_size)
    for model, model_name in models_cifar100:
        transform = "standard"
        train_test_model(model,"CIFAR-100", num_epochs, lr, transform, model_name, criterion, optimizer, weight_decay, dropout, batch_size)
        # train_model(model,"CIFAR-100", num_epochs, lr, transform, criterion, optimizer, weight_decay, dropout, batch_size)
        transform = "custom"
        train_test_model(model,"CIFAR-100", num_epochs, lr, transform, model_name, criterion, optimizer, weight_decay, dropout, batch_size)
        # train_model(model,"CIFAR-100", num_epochs, lr, transform, criterion, optimizer, weight_decay, dropout, batch_size)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Train a model and export it for PIGEON')
    parser.add_argument('--model', type=str, default='LeNet', help='Model to use, default is LeNet')
    parser.add_argument('--num_classes', type=int, default=10, help='Number of classes, default is 10')
    parser.add_argument('--dataset_name', type=str, default='MNIST', help='Dataset name, default is MNIST')
    parser.add_argument('--modelpath', type=str, default='./models/lenet5_mnist', help='Path to save/load the model, default is ./models/lenet5_mnist')
    parser.add_argument('--num_epochs', type=int, default=5, help='Number of epochs, default is 5')
    parser.add_argument('--lr', type=float, default=0.01, help='Learning rate, default is 0.01')
    parser.add_argument('--criterion', type=str, default='CrossEntropyLoss', help='Loss function, default is CrossEntropyLoss')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Optimizer, default is Adam')
    parser.add_argument('--dropout', type=float, default=0.0, help='Dropout Rate, default is 0.0')
    parser.add_argument('--weight_decay', type=float, default=0.0, help='Weight Decay Rate, default is 0.0')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch Size, default is 32')
    parser.add_argument('--test_batches', type=int, default=-1, help='Number of test batches to evaluate, default is -1 (whole test set)')

    # Add arguments for actions
    parser.add_argument('--action', type=str, choices=['train', 'import', 'train_all', 'test', 'none'], default='train', help='Action to perform on the model. Options are train (default), import, train_all, test, none')
    parser.add_argument('--export_model', action='store_true', help='Export the model as a .bin file for PIGEON, default is True')
    parser.add_argument('--export_dataset', action='store_true', help='Export the test dataset as a .bin file for PIGEON, default is True')
    
    # Add argument for transformation
    parser.add_argument('--transform', type=str, choices=['custom', 'standard'], default='standard', help='Type of transformation to apply, default is standard')

    args = parser.parse_args()

    # Create directories if they do not exist
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    weight_decay = args.weight_decay
    dropout = args.dropout
    batch_size = args.batch_size

    # Dynamically create model, criterion, and optimizer
    model_class = getattr(cnns, args.model)
    model = model_class(num_classes=args.num_classes, dropout=dropout)
    
    criterion = getattr(nn, args.criterion)
    
    optimizer = getattr(optim, args.optimizer)




    # Perform the specified action
    if args.action == 'train':
        print("Parameters set to: num_epochs={}, lr={}, transform={}, criterion={}, optimizer={}, weight_decay={}, dropout={}, batch_size={}".format(args.num_epochs, args.lr, args.transform, args.criterion, args.optimizer, args.weight_decay, args.dropout, args.batch_size))
        train_model(model, args.dataset_name, args.num_epochs, args.lr, args.transform, criterion, optimizer, weight_decay, dropout, batch_size)
    elif args.action == 'import':
        load_model(model, args.modelpath)
    elif args.action == 'train_all':
        print("Parameters set to: num_epochs={}, lr={}, criterion={}, optimizer={}, weight_decay={}, dropout={}, batch_size={}".format(args.num_epochs, args.lr, args.criterion, args.optimizer, args.weight_decay, args.dropout, args.batch_size))
        train_all(args.num_epochs, args.lr, criterion, optimizer, weight_decay, dropout, batch_size)
    elif args.action == 'test':
        load_model(model, args.modelpath)
        test_model(model, args.dataset_name, args.transform, batch_size, criterion, args.test_batches)
    elif args.action == 'none':
        print("No action specified for the model.")

    # Export the test dataset if specified
    if args.export_dataset:
        export_test_dataset(args.dataset_name, args.transform, batch_size)
        print("Exported the test dataset for PIGEON.")
    
    # Export the model if specified
    if args.export_model:
        parameter_export.save_weights_compatible_with_cpp(model, args.modelpath + '.bin')
        print("Exported the model for PIGEON.")

if __name__ == '__main__':
    main()
