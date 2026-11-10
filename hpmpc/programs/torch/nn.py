"""Define the neural network here for torch_to_hpmpc.py to read"""

from torch import nn

SIZES = [5, 50, 50, 50, 50, 50, 50, 5]

class AcasXu(nn.Module):
    def __init__(self):
        super().__init__()
        layers = []
        for i, (a,b) in enumerate(zip(SIZES, SIZES[1:])):
            layers.append(nn.Linear(a,b))
            if i + 2 < len(SIZES):
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


# Specify NIDS architecture here, used for generating training code in C++
# Make sure this matches the .onnx architecture if finetuning a model
