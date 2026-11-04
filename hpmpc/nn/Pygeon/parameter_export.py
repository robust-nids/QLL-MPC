import torch
import numpy as np

def save_weights_compatible_with_cpp(model, filepath):
    with open(filepath, 'wb') as f:
        all_params = []

        # Process each module according to its type
        for module in model.modules():
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

def count_exported_params(model):
    total_params = 0
    for module in model.modules():  # `modules()` will iterate over all modules in the network
        if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
            # Count weights and biases
            total_params += module.weight.data.nelement()
            if module.bias is not None:
                total_params += module.bias.data.nelement()

        elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            # Count all BatchNorm parameters
            total_params += module.running_mean.nelement()
            total_params += module.running_var.nelement()
            if module.weight is not None:
                total_params += module.weight.data.nelement()
            if module.bias is not None:
                total_params += module.bias.data.nelement()

    print("Total number of parameters: {}".format(total_params))
    return total_params


def write_params(model, filename):
    total_params = count_exported_params(model)
    with open(filename, 'wb') as f:
        # Write the total number of parameters first
        np.array([total_params], dtype=np.int32).tofile(f)

        # Then write the parameters for each module
        for module in model.modules():
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d)):
                # Flatten and save weights
                np_weights = module.weight.data.numpy().flatten()
                np_weights.tofile(f)

                # Save biases
                if module.bias is not None:
                    np_bias = module.bias.data.numpy()
                    np_bias.tofile(f)

            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                # Save BatchNorm parameters
                np_mu = module.running_mean.numpy()
                np_var = module.running_var.numpy()
                np_gamma = module.weight.data.numpy()
                np_beta = module.bias.data.numpy()

                np_mu.tofile(f)
                np_var.tofile(f)
                np_gamma.tofile(f)
                np_beta.tofile(f)



def read_params(model, filename):
    total_params = count_exported_params(model)
    with open(filename, 'rb') as f:
        # Read the total number of parameters first
        num_params = np.fromfile(f, dtype=np.int32, count=1)[0]
        if num_params != total_params:
            raise ValueError("Expected {} parameters, but found {}.".format(total_params, num_params))
        for module in model.modules():
            
            if isinstance(module, torch.nn.Linear):
                # Read and reshape weights, then biases
                read_to_tensor(f, module.weight)
                if module.bias is not None:
                    read_to_tensor(f, module.bias)

            elif isinstance(module, torch.nn.Conv2d):
                # Read and reshape kernel weights, then biases
                read_to_tensor(f, module.weight)
                if module.bias is not None:
                    read_to_tensor(f, module.bias)

            elif isinstance(module, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
                # Read BatchNorm parameters in order: moving mean, variance, gamma, beta
                read_to_tensor(f, module.running_mean)
                read_to_tensor(f, module.running_var)
                if module.weight is not None:
                    read_to_tensor(f, module.weight)
                if module.bias is not None:
                    read_to_tensor(f, module.bias)


def read_to_tensor(file, tensor):
    num_elements = tensor.numel()
    tensor_data = np.fromfile(file, dtype=np.float32, count=num_elements)
    tensor.data.copy_(torch.from_numpy(tensor_data).view_as(tensor))


