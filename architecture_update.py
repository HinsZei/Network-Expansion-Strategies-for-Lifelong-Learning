import numpy as np
import torch
import torch.nn as nn
import random
import copy


def update_weights(net, output_layer_name_prefix, amnt_new_classes, device):
    weights = net.state_dict()[f"{output_layer_name_prefix}.weight"].cpu().detach().numpy()
    w_mean = np.mean(weights, axis=0)
    w_std = np.std(weights, axis=0)
    new_weights = np.pad(weights, ((0, amnt_new_classes), (0, 0)), mode="constant", constant_values=0)
    for i in reversed(range(amnt_new_classes)):
        for j in range(new_weights.shape[1]):
            new_weights[new_weights.shape[0] - 1 - i][j] = np.random.normal(w_mean[j], w_std[j])
    return new_weights


def update_bias(net, output_layer_name_prefix, amnt_new_classes, device):
    bias = net.state_dict()[f"{output_layer_name_prefix}.bias"].cpu().detach().numpy()
    b_mean = np.mean(bias)
    b_std = np.std(bias)
    new_bias = np.zeros(len(bias) + amnt_new_classes, dtype="f")
    new_bias[:len(bias)] = bias
    for i in range(amnt_new_classes):
        new_bias[-1 - i] = np.random.normal(b_mean, b_std)
    return new_bias

def expand_layer(layer1, layer2, new_width, normalise_flag=False, random_init=False):
    weight1 = layer1.weight.data
    weight2 = layer2.weight.data
    bias1 = layer1.bias.data
    old_width = weight1.size(0)
    new_weight1 = layer1.weight.data.clone()
    new_weight2 = layer2.weight.data.clone()
    assert weight1.size(0) == weight2.size(1)
    assert new_width > weight1.size(0)
    if new_weight1.dim() < 4:
        new_weight1.resize_(new_width, new_weight1.size(1))
        new_weight2.resize_(new_weight2.size(0), new_width)
    if bias1 is not None:
        nb1 = layer1.bias.data.clone()
        nb1.resize_(new_width)
    weight2 = weight2.transpose(0, 1)
    new_weight2 = new_weight2.transpose(0, 1)

    new_weight1.narrow(0, 0, old_width).copy_(weight1)
    new_weight2.narrow(0, 0, old_width).copy_(weight2)
    nb1.narrow(0, 0, old_width).copy_(bias1)

    if normalise_flag:
        for i in range(old_width):
            norm = weight1.select(0, i).norm()
            weight1.select(0, i).div_(norm)

    # contain how many times a node's weight was selected to init the new nodes.
    record = dict()
    for i in range(old_width, new_width):
        idx = np.random.randint(0, old_width)
        try:
            record[idx].append(i)
        except:
            record[idx] = [idx]
            record[idx].append(i)
        if random_init:
            n1 = layer1.out_features * layer1.in_features
            n2 = layer2.out_features * layer2.in_features
            new_weight1.select(0, i).normal_(0, np.sqrt(2. / n1))
            new_weight2.select(0, i).normal_(0, np.sqrt(2. / n2))
        else:
            new_weight1.select(0, i).copy_(weight1.select(0, idx).clone())
            new_weight2.select(0, i).copy_(weight2.select(0, idx).clone())
        nb1[i] = bias1[idx]
    if not random_init:
        for idx, new_nodes in record.items():
            for item in new_nodes:
                new_weight2[item].div_(len(new_nodes))
    weight2.transpose_(0, 1)
    new_weight2.transpose_(0, 1)
    new_layer1 = nn.Linear(layer1.in_features, new_width)
    new_layer2 = nn.Linear(new_width, layer2.out_features)
    new_layer1.weight.data = nn.Parameter(new_weight1)
    new_layer1.bias.data = nn.Parameter(nb1)
    new_layer2.weight.data = nn.Parameter(new_weight2)
    return new_layer1, new_layer2


def extend_layer(layer1, nonlinear, count):
    if 'Linear' in layer1.__class__.__name__:
        layer2 = nn.Linear(layer1.out_features, layer1.out_features)
        layer2.weight.data.copy_(torch.eye(layer1.out_features))
    else:
        layer2 = nn.Linear(layer1[0].out_features, layer1[0].out_features)
        layer2.weight.data.copy_(torch.eye(layer1[0].out_features))
    layer2.bias.data.zero_()
    if 'Linear' in layer1.__class__.__name__:
        sequential = torch.nn.Sequential(
            layer1,
            nonlinear(),
            layer2
        )
        return sequential
    else:
        layer1.add_module('nonlinear' + str(count), nonlinear())
        layer1.add_module('fc' + str(count), layer2)
        return layer1


def update_output_layer(layer, tasks_number):
    weight = layer.weight.data
    bias = layer.bias.data
    old_width = weight.size(0)
    input_number = weight.size(1)
    new_bias = copy.deepcopy(bias)
    new_weight = copy.deepcopy(weight)
    weight_numpy = weight.cpu().detach().numpy()
    mean = np.mean(weight_numpy, axis=1)
    standard = np.std(weight_numpy, axis=1)

    if weight.dim() < 4:
        new_weight.resize_(old_width + tasks_number, input_number)
    if bias is not None:
        new_bias.resize_(old_width + tasks_number)

    for i in range(old_width, old_width + tasks_number):
        new_weight.select(0, i).normal_(0, 1)
    new_layer = nn.Linear(input_number, old_width + tasks_number)
    new_layer.weight.data = nn.Parameter(new_weight)
    if bias is not None:
        new_layer.bias.data = nn.Parameter(new_bias)
    return new_layer


def update_model(NET_FUNCTION, net, data_shape, amount_new_classes, device):
    """
    Adds neurons to output layer，temporarily abandoned, as I found that the weights of output layer always totally
    changed, it is no need to keep the distribution
    """
    output_layer_nr = len(list(net.children()))
    network_layer_names = list(net.state_dict().keys())
    # get the name of output layer like 'fc4'
    output_layer_name_prefix = network_layer_names[-1][:network_layer_names[-1].index(".")]
    amnt_old_classes = list(net.children())[-1].out_features
    amnt_neurons = list(net.children())[-2].out_features

    if amount_new_classes >= 1:
        new_output_layer_w = update_weights(net, output_layer_name_prefix, amount_new_classes, device)
        new_output_layer_bias = update_bias(net, output_layer_name_prefix, amount_new_classes, device)
        new_model = NET_FUNCTION(data_shape, amnt_old_classes + amount_new_classes, amnt_neurons)

        for i, l in enumerate(zip(new_model.children(), net.children())):

            if i == len(list(new_model.children())) - 1:
                l[0].weight = torch.nn.Parameter(torch.from_numpy(new_output_layer_w))
                l[0].bias = torch.nn.Parameter(torch.from_numpy(new_output_layer_bias))
            else:
                l[0].weight = l[1].weight
                l[0].bias = l[1].bias
        del net
        new_model = new_model.to(device)

        return new_model
    else:
        return net


def check_model_integrity(old_model, new_model, verbose=False):
    """
    Checks whether the model is consistent after updating
    """
    for i in old_model.state_dict().keys():
        if (np.array_equal(old_model.state_dict()[i].cpu().numpy(), new_model.state_dict()[i].cpu().numpy())):
            if verbose:
                print(f"key {i} is the same for both nets")
        else:
            if verbose:
                print("\n", i, "\n")
            for h in range(len(old_model.state_dict()[i])):
                try:
                    if np.array_equal(old_model.state_dict()[i][h].cpu().numpy(),
                                      new_model.state_dict()[i][h].cpu().numpy()):
                        if verbose:
                            print(f"key {i} weights of neuron {h} are the same for both nets\n")
                    else:

                        print(f"key {i} weights of neuron {h} are different for both nets\n Differces at:")
                        print(old_model.state_dict()[i][h].cpu().numpy() - new_model.state_dict()[i][h].cpu().numpy())
                        print("\n")
                        return False
                except:
                    print("PROBLEM")
    return True
