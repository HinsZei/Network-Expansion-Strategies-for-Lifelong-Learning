import random
import math
import torch
import torchvision
from torchvision import transforms, datasets
import collections
import pytorch_util

'''
check how many samples in each class in trainset
'''


def draw_random_tasks(numbers, numbers_per_task, random_shuffle_tasks):
    """

    :param numbers: list of classes, 0...n(provided by torch.dataset)
    :param numbers_per_task: as the name said
    :param random_shuffle_tasks:  boolean, indicates whether the classes are shuffled
    :return: task sequence
    """
    number_l = numbers.copy()
    if random_shuffle_tasks:
        random.shuffle(number_l)
    if len(numbers) > numbers_per_task:
        nr_of_full_tasks = math.floor(len(number_l) / numbers_per_task)
        tasks = [number_l[x:x + numbers_per_task] for x in
                 range(0, nr_of_full_tasks * numbers_per_task,
                       numbers_per_task)]
        if len(numbers) % nr_of_full_tasks != 0:
            # add rest of the tasks
            tasks.append(number_l[nr_of_full_tasks * numbers_per_task:])
    else:
        tasks = [number_l]

    return tasks


def get_MNIST_tasks(number_list, numbers_per_task, batch_size, val_split_perc,
                    random_shuffle_tasks=True):
    train = datasets.MNIST("../Data", train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST("../Data", train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))

    testset_all = torch.utils.data.DataLoader(test, batch_size=batch_size)
    task_list = draw_random_tasks(number_list, numbers_per_task, random_shuffle_tasks)
    # add the indices of old tasks' samples into memory, later the indices in memory is mixes with new task's indices
    # to generate each train/test set for the new tasks
    tasks = {}
    memory_tr = {}
    memory_te = {}
    # 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000
    memory_tr_size = 2000
    for task_id, task in enumerate(task_list):
        idx_tr = []
        idx_te = []
        number_new_samples = 0
        if len(memory_tr.keys()) != 0:
            number_old_samples = math.floor(memory_tr_size / len(memory_tr.keys()))
        else:
            number_old_samples = 0
        size_each_class_tr = math.floor(memory_tr_size / (len(memory_tr.keys()) + numbers_per_task))

        if len(memory_tr.keys()) != 0:
            # except for the first task, add stored indices into idx_tr, which represents the samples for each task
            for key in memory_tr.keys():
                idx_tr += memory_tr[key]
            for key in memory_te.keys():
                idx_te += memory_te[key]
        for classes in task:
            idx_r = pytorch_util.get_class_indices(train, classes)
            # pick some samples
            memory_tr[classes] = random.sample(idx_r, size_each_class_tr)
            idx_tr += idx_r
            number_new_samples += len(idx_r)
        for classes in task:
            idx_r = pytorch_util.get_class_indices(test, classes)
            # for test set all the samples should be included
            memory_te[classes] = idx_r
            idx_te += idx_r
        if len(memory_tr.keys()) > numbers_per_task:
            for key in memory_tr.keys():
                memory_tr[key] = random.sample(memory_tr[key], size_each_class_tr)
        # test set included all test samples in previous tasks, while train set only contains current task samples
        # and holdout data
        trainset = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                               sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_tr))
        testset = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                              sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_te))
        val_split_size = int(round(len(trainset) * val_split_perc, 0))

        trainset, valset = torch.utils.data.random_split(trainset,
                                                         [len(trainset) - val_split_size, val_split_size])

        tasks[task_id] = {'Class_ids': task, 'Train': trainset, 'Val': valset, 'Test': testset,
                          'OldSamples': number_old_samples, 'NewSamples': number_new_samples}
    return tasks, testset_all


def get_MNIST(number_list, batch_size, val_split_perc):
    """get the whole MNIST dataset"""
    train = datasets.MNIST("../Data", train=True, download=True,
                           transform=transforms.Compose([transforms.ToTensor()]))
    test = datasets.MNIST("../Data", train=False, download=True,
                          transform=transforms.Compose([transforms.ToTensor()]))

    idx_tr = pytorch_util.get_class_indices(train, number_list)
    idx_te = pytorch_util.get_class_indices(test, number_list)

    trainset = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                           sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_tr))
    testset = torch.utils.data.DataLoader(test, batch_size=batch_size,
                                          sampler=torch.utils.data.sampler.SubsetRandomSampler(idx_te))

    val_split_size = int(round(len(trainset) * val_split_perc, 0))

    train_set, val_set = torch.utils.data.random_split(trainset,
                                                       [len(trainset) - val_split_size, val_split_size])

    Data = {"Train": train_set, "Val": val_set, "Test": testset}

    return Data
