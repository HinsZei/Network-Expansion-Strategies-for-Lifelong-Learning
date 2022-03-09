from torchvision import transforms, datasets
import torch
from scipy.spatial.distance import cdist
import numpy as np
import json
from pytorch_util import get_class_indices


def class_similarity_dict(dataset, device):
    if dataset == 'CIFAR10':
        data_loader = datasets.CIFAR10("../Data", train=True, download=False,
                                       transform=transforms.Compose([transforms.ToTensor()]))
    else:
        data_loader = datasets.MNIST("../Data", train=True, download=False,
                                     transform=transforms.Compose([transforms.ToTensor()]))
    classes_target = []
    for value, key in enumerate(data_loader.class_to_idx):
        classes_target.append(value)
    classes_samples = {}
    for class_id in classes_target:
        idx_tr = get_class_indices(data_loader, class_id)
        trainset = torch.utils.data.DataLoader(data_loader, batch_size=len(idx_tr),
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_tr))
        classes_samples[class_id] = trainset
    similarity = {}
    for class_id1 in range(len(classes_target)):
        for class_id2 in range(class_id1 + 1, len(classes_target)):
            class_1 = classes_samples[class_id1]
            class_2 = classes_samples[class_id2]
            class_1_batches = []
            class_2_batches = []
            # distance = []
            for data, label in class_1:
                class_1_batches.append(data)
            for data, label in class_2:
                class_2_batches.append(data)
            # Abandoned, for reshaping the tensor I made it complicated
            # class_1_tensor = class_1_batches[0].to(device)
            # class_2_tensor = class_2_batches[0].to(device)
            # for index in range(len(class_1_batches)):
            #     if index == 0:
            #         continue
            #     class_1_tensor = torch.cat((class_1_tensor,class_1_batches[index].to(device)),0)
            # for index in range(len(class_2_batches)):
            #     if index == 0:
            #         continue
            #     class_2_tensor = torch.cat((class_2_tensor,class_2_batches[index].to(device)),0)
            size = list(class_1_batches[0].shape)
            image_size = size[len(size) - 1] * size[len(size) - 2] * size[len(size) - 3]
            if len(class_1_batches) == 1:
                dist = cdist(class_1_batches[0].reshape(-1, image_size).cpu().numpy(),
                             class_2_batches[0].reshape(-1, image_size).cpu().numpy(), metric='cosine')
                similarity[str(class_id1) + ' ' + str(class_id2)] = {'class': class_id2, 'mean': dist.mean(),
                                                                     'max': dist.max(), 'min': dist.min(),
                                                                     'std': dist.std()}
    if dataset == 'CIFAR10':
        with open('similarity_cifar10_cosine.json', 'w') as file:
            file.write(json.dumps(similarity))
    else:
        with open('similarity_mnist_cosine.json', 'w') as file:
            file.write(json.dumps(similarity))
    return similarity


def similarity_matrix_transform(similarity, classes_target):
    matrix = np.zeros((classes_target, classes_target))
    matrix = matrix.tolist()
    for i in range(classes_target):
        for j in range(i + 1, classes_target):
            matrix[i][j] = similarity[str(i) + ' ' + str(j)]
    return matrix


def read_similarity_matrix(classes_target, file_name):
    with open(file_name, 'r') as file:
        similarity_dict = json.loads(file.read())
    similarity_matrix = similarity_matrix_transform(similarity_dict, classes_target)
    return similarity_matrix
