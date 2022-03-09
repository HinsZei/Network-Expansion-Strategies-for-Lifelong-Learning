import time
import numpy as np
import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR
import similarity_util
import pytorch_util
import pandas as pd


def create_proxy_outputs_from_onehot(model, task_pool, batch_y):
    """
    Labels need to be transformed for Pytorch to go from 0-X
    Since we shuffle the tasks our data is not provided with increasing labels
    """
    current_output_neurons = list(model.children())[-1].in_features

    # inititalize array with 0s 
    proxy_outputs = torch.zeros([batch_y.shape[0], current_output_neurons])
    # convert one hot to class
    batch_classes = np.where(batch_y == 1)[1]

    # get index of the class in pool
    pool_ind = [task_pool.index(c) for c in batch_y]

    for i, j in enumerate(pool_ind):
        proxy_outputs[i, j] = 1

    return proxy_outputs


def create_proxy_outputs(model, task_pool, batch_y):
    """
    Labels need to be transformed for Pytorch to go from 0-X
    Since we shuffle the tasks our data is not provided with increasing labels
    """
    current_output_neurons = list(model.children())[-1].in_features

    # inititalize array with 0s 
    proxy_outputs = torch.zeros(len(batch_y), dtype=torch.long)

    # get index of the class in pool
    pool_ind = [task_pool.index(c) for c in batch_y]

    for i, j in enumerate(pool_ind):
        proxy_outputs[i] = j

    return proxy_outputs


def batch_transform(batch, model, task_pool, device):
    """
    Transforms a batch to fit the model requirements
    """
    batch_x, batch_y = batch
    batch_y = create_proxy_outputs(model, task_pool, batch_y)

    # calculate model outputs and loss and backprop
    if model.type == "fc":
        shape_length = len(batch_x.shape)
        batch_x = batch_x.view(-1, batch_x.shape[shape_length - 3] * batch_x.shape[shape_length - 2] *
                               batch_x.shape[shape_length - 1])
    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
    return batch_x, batch_y


def train(model, trainset, task_pool, epoch, optimizer, device, amount_of_batches=-1, reg_loss=0):
    if amount_of_batches == -1:
        amount_of_batches = len(trainset)
    # batch_size = trainset.dataset.batch_size
    batch_count = 0
    # amount_of_batches *= len(trainsets)
    # for trainset in trainsets:
    for b_id, batch in enumerate(trainset.dataset):
        if batch_count > amount_of_batches:
            break
        if b_id not in trainset.indices:
            continue
        batch_x, batch_y = batch_transform(batch, model, task_pool, device)
        # at each epoch we have to zero the gradients 2 options:
        # optimizer.zero_grad() # if we have different optimizers
        model.zero_grad()  # zero all gradients for whole net

        output = model(batch_x)

        loss = F.cross_entropy(output, batch_y)
        # loss = F.cross_entropy(output, batch_y)+reg_loss(model)

        loss.backward()
        optimizer.step()

        if batch_count == amount_of_batches - 1 or batch_count == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_count + 1, amount_of_batches,
                       100. * batch_count / amount_of_batches, loss.item()))
        batch_count += 1


def validate(model, valset, task_pool, device, verbose=True):
    """
    Arguments:
        model -- (Pytorch Model)
        testset -- (Pytorch Dataset) 
        task_pool -- (List<integers>) Task labels the model has already seen
        device -- (Pytorch Device) Flag whether model is trained on CPU or GPU
        verbose -- (boolean) Flag whether feedback should be provided
    """
    model.eval()
    correct = 0
    val_loss = 0
    batch_size = valset.dataset.batch_size
    last_batch_size = 0
    with torch.no_grad():
        for b_id, batch in enumerate(valset.dataset):
            if b_id == len(valset) - 1:
                last_batch_size = len(val_batch_y)
            if b_id not in valset.indices:
                continue
            val_batch_x, val_batch_y = batch_transform(batch, model, task_pool, device)

            output = model(val_batch_x)
            val_loss += F.cross_entropy(output, val_batch_y, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability

            correct += pred.eq(val_batch_y.view_as(pred)).sum().item()

    val_loss /= len(valset)
    total_samples = (len(valset) - 1) * batch_size + last_batch_size
    if verbose:
        print(
            f"\nValidationset: Average loss: {round(val_loss, 4)}, Accuracy: {correct}/{total_samples} ({round(100. * correct / (total_samples), 4)}%)\n")
    return round(val_loss, 4), round(correct / (total_samples), 3)


def test(model, testset, task_pool, device, verbose=True):
    """
    Arguments:
        model -- (Pytorch Model)
        testset -- (Pytorch Dataset) 
        task_pool -- (List<integers>) Task labels the model has already seen
        device -- (Pytorch Device) Flag whether model is trained on CPU or GPU
    """
    model.eval()
    correct = 0
    test_loss = 0
    try:
        batch_size = testset.batch_size
    except:
        batch_size = testset.dataset.batch_size
    last_batch_size = 0
    with torch.no_grad():
        for b_id, batch in enumerate(testset):
            test_batch_x, test_batch_y = batch_transform(batch, model,
                                                         task_pool, device)
            if (b_id + 1) == len(testset):
                last_batch_size = len(test_batch_y)
            output = model(test_batch_x)
            test_loss += F.cross_entropy(output, test_batch_y, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(test_batch_y.view_as(pred)).sum().item()

    test_loss /= len(testset)
    total_samples = (len(testset) - 1) * batch_size + last_batch_size
    if verbose:
        print(
            f"\nTest set: Average loss: {round(test_loss, 4)}, Accuracy: {correct}/{total_samples} ({100. * correct / (total_samples)}%)\n\n")
    return round(test_loss, 4), round(correct / (total_samples), 3)


def training_process(MODEL, EPOCHS, TASKS, TESTSET, OPTIMIZER, LEARNING_RATE,
                     AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available", EARLY_STOP=False, strategy=0):
    """
    Drives the training process
    Arguments:
        MODEL -- (Pytorch Model Function)
        EPOCHS -- (Integer) how many epochs should be trained per task
        TASKS -- (Integer) How many tasks should the model learn
        TESTSET -- (Pytorch Dataset)
        OPTIMIZER -- (Pytorch Optimizizer Function)
        LEARNING_RATE -- (Float)
        AMOUNT_OF_TRAIN_BATCHES -- (Integer) How many batches should be trained
    """

    DEVICE = pytorch_util.define_runtime_environment(device_flag)
    torch.cuda.empty_cache()

    # create net with parameters of the first task
    first_key = list(TASKS.keys())[0]
    data_shape = TASKS[first_key]['Train'].dataset.dataset[0][0].shape
    initial_classes = len(TASKS[first_key]['Class_ids'])
    model = MODEL(data_shape, initial_classes).to(DEVICE)
    # reg_loss = Regularization(model, 0.1, p=2).to(DEVICE)
    done_keys = []
    task_pool = []
    try:
        df = pd.read_csv('log.csv')
    except:
        columns = ['number of classes learnt', 'task sequence', 'ratio of new/old samples', 'action',
                   'similarity_mean',
                   'similarity_tasks', 'previous val acc', 'current val acc', 'previous test acc',
                   'current test acc', 'previous_val_acc_min', 'previous_val_acc_max', 'previous_val_acc_mean',
                   'previous_test_acc_min', 'previous_test_acc_max', 'previous_test_acc_mean', 'train time',
                   'parameters']
        df = pd.DataFrame(columns=columns, index=None)
        df.to_csv('log.csv')
    start = time.time()
    for task_id, key in enumerate(TASKS.keys()):
        print("\nNEW TASK", task_id)
        trainset = TASKS[key]['Train']
        valset = TASKS[key]['Val']
        testset = TASKS[key]['Test']
        # similarity
        similarity_matrix = similarity_util.read_similarity_matrix(10, 'similarity_mnist_cosine.json')

        # reset optimizer for new task
        task_pool += TASKS[key]['Class_ids']
        print(f"With {len(trainset)} training and {len(valset)} validation batches.")
        optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        if EARLY_STOP:
            early_stopper = pytorch_util.Early_Stop(patience=5)

        for epoch in range(1, EPOCHS + 1):
            train(model, trainset, task_pool, epoch, optimizer,
                  DEVICE, AMOUNT_OF_TRAIN_BATCHES)
            # train(model, trainset, task_pool, epoch, optimizer,
            #       DEVICE, AMOUNT_OF_TRAIN_BATCHES,reg_loss)
            v_l, v_ac = validate(model, valset, task_pool, DEVICE)
            if EARLY_STOP:
                early_stopper(v_ac, v_l, model, epoch)
                if early_stopper.trigger:
                    break
            scheduler.step()

        log_dict = {}
        log_dict['number of classes learnt'] = len(task_pool)
        log_dict['task sequence'] = f'{task_pool}'
        log_dict['ratio of new/old samples'] = f"{TASKS[key]['NewSamples']}/{TASKS[key]['OldSamples']}"
        log_dict['action'] = strategy

        print("--------------------------------------------------")
        print(f"number of classes being learnt:{len(task_pool)},classes:{task_pool}")
        print("--------------------------------------------------")
        print(f"ratio of new/old classes'samples:{TASKS[key]['NewSamples']}/{TASKS[key]['OldSamples']}")
        print("--------------------------------------------------")
        if strategy == 1:
            print("action: expand the second last layer")
        elif strategy == 2:
            print("action: duplicate the second last layer")
        else:
            print("action: None")
        print("--------------------------------------------------")
        similarity_list = []
        similarity_tasks = str()
        if len(done_keys):
            count = 0
            for old_class in task_pool[:-2]:
                for new_class in TASKS[key]["Class_ids"]:
                    if old_class < new_class:
                        print(
                            # f"distance between class {old_class} and class {new_class} is: max:{similarity_matrix[old_class][new_class]['max']} , min:{similarity_matrix[old_class][new_class]['min']}")
                            f"distance between class {old_class} and class {new_class} is: mean:{similarity_matrix[old_class][new_class]['mean']}")
                        similarity_list.append(similarity_matrix[old_class][new_class]['mean'])
                    else:
                        print(
                            f"distance between class {old_class} and class {new_class} is: mean:{similarity_matrix[new_class][old_class]['mean']}")
                        similarity_list.append(similarity_matrix[new_class][old_class]['mean'])
                count += 2
                if count % 4 == 0:
                    similarity_tasks += f"TASK {(count / 4) - 1}: {np.mean(similarity_list[count - 4:count])} "
                log_dict['similarity_mean'] = np.mean(similarity_list)
                log_dict['similarity_tasks'] = similarity_tasks

        # print task accuracies
        print("--------------------------------------------------")

        pre_acc = str()
        if len(done_keys):
            pre_v = []
            print("Previous tasks:")
            for k in done_keys:
                prev_v_l, prev_v_ac = validate(model, TASKS[k]['Val'], task_pool,
                                               DEVICE, verbose=False)
                pre_v.append(prev_v_ac)
                pre_acc += f"Task: {k}: {prev_v_ac} "
                print(f"Task: {k}: Accuracy:{prev_v_ac}, Loss:{prev_v_l}")
            log_dict['previous_val_acc_min'] = np.min(pre_v)
            log_dict['previous_val_acc_max'] = np.max(pre_v)
            log_dict['previous_val_acc_mean'] = np.mean(pre_v)
        print("\nCurrent task:")
        curr_v_l, curr_v_ac = validate(model, TASKS[key]['Val'], task_pool,
                                       DEVICE, verbose=False)
        cur_acc = f"{curr_v_ac}"
        print(f"Task: {key}: Accuracy:{curr_v_ac}, Loss:{curr_v_l}")
        print("--------------------------------------------------\n")
        log_dict['previous val acc'] = pre_acc
        log_dict['current val acc'] = cur_acc
        pre_acc = str()
        if len(done_keys):
            print("Previous tasks:")
            pre_t = []
            for k in done_keys:
                prev_t_l, prev_t_ac = test(model, TASKS[k]['Test'], task_pool,
                                           DEVICE)
                pre_t.append(prev_t_ac)
                pre_acc += f"Task: {k}: {prev_t_ac} "
                # print(f"Task: {k}: Accuracy:{prev_t_ac}, Loss:{prev_t_l}")
            log_dict['previous_test_acc_min'] = np.min(pre_t)
            log_dict['previous_test_acc_max'] = np.max(pre_t)
            log_dict['previous_test_acc_mean'] = np.mean(pre_t)
        print("\nCurrent task:")
        curr_t_l, curr_t_ac = test(model, testset, task_pool, DEVICE)
        cur_acc = f"{curr_t_ac}"
        # print(f"Task: {key}: Accuracy:{curr_t_ac}, Loss:{curr_t_l}")
        print("--------------------------------------------------\n")
        log_dict['previous test acc'] = pre_acc
        log_dict['current test acc'] = cur_acc
        done_keys.append(key)
        if task_id == list(TASKS.keys())[-1]:
            total_parameters = sum(param.numel() for param in model.parameters())
            log_dict['parameters'] = total_parameters
            torch.cuda.synchronize(DEVICE)
            end = time.time()
            log_dict['train time'] = int(end) - int(start)
        df = df.append(log_dict, ignore_index=True)
        if task_id != list(TASKS.keys())[-1]:
            # reset learning rate for optimizer after decay
            for g in optimizer.param_groups:
                g['lr'] = LEARNING_RATE
            # update model
            # old_model = copy.deepcopy(model)
            # model = architecture_update.update_model(MODEL, model, data_shape,
            #                                          len(TASKS[key]["Class_ids"]), DEVICE)
            # layer = architecture_update.update_output_layer(model._modules['fc4'], 2)
            # model._modules['fc4'] = layer
            model.update_output_layer()
            if strategy == 1:
                model.wider()
            elif strategy == 2:
                # new_layer = architecture_update.extend_layer(model._modules['fc3'], torch.nn.ReLU)
                # model._modules['fc3'] = new_layer
                model.deeper(task_id)
            elif strategy == 3:
                if task_id % 2 == 0:
                    model.wider()
            elif strategy == 4:
                if task_id % 2 == 0:
                    model.deeper(task_id)
            elif strategy == 5:
                if task_id != 0:
                    if log_dict['similarity_mean'] <= 0.607:
                        model.wider()
            elif strategy == 6:
                if task_id != 0:
                    if log_dict['similarity_mean'] <= 0.607:
                        model.deeper(task_id)
            # del old_model
            model.to(DEVICE)
            # if not architecture_update.check_model_integrity(old_model, model):
            #     print("Model integrity broken after addition of output units.")
            #     break

    df.to_csv('log.csv', index=False)
    return model


def training_process_update_first(MODEL, EPOCHS, TASKS, TESTSET, OPTIMIZER, LEARNING_RATE,
                                  AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available", EARLY_STOP=False, strategy=0):
    """
    Drives the training process
    Arguments:
        MODEL -- (Pytorch Model Function)
        EPOCHS -- (Integer) how many epochs should be trained per task
        TASKS -- (Integer) How many tasks should the model learn
        TESTSET -- (Pytorch Dataset)
        OPTIMIZER -- (Pytorch Optimizizer Function)
        LEARNING_RATE -- (Float)
        AMOUNT_OF_TRAIN_BATCHES -- (Integer) How many batches should be trained
    """

    DEVICE = pytorch_util.define_runtime_environment(device_flag)
    torch.cuda.empty_cache()

    # create net with parameters of the first task
    first_key = list(TASKS.keys())[0]
    data_shape = TASKS[first_key]['Train'].dataset.dataset[0][0].shape
    initial_classes = len(TASKS[first_key]['Class_ids'])
    model = MODEL(data_shape, initial_classes).to(DEVICE)
    # reg_loss = Regularization(model, 0.1, p=2).to(DEVICE)
    done_keys = []
    task_pool = []
    try:
        df = pd.read_csv('log.csv')
    except:
        columns = ['number of classes learnt', 'task sequence', 'ratio of new/old samples', 'action',
                   'similarity_mean',
                   'similarity_tasks', 'previous val acc', 'current val acc', 'previous test acc',
                   'current test acc', 'previous_val_acc_min', 'previous_val_acc_max', 'previous_val_acc_mean',
                   'previous_test_acc_min', 'previous_test_acc_max', 'previous_test_acc_mean', 'train time',
                   'parameters']
        df = pd.DataFrame(columns=columns, index=None)
        df.to_csv('log.csv')
    start = time.time()
    # similarity
    similarity_matrix = similarity_util.read_similarity_matrix(10, 'similarity_mnist_cosine.json')
    for task_id, key in enumerate(TASKS.keys()):
        print("\nNEW TASK", task_id)
        trainset = TASKS[key]['Train']
        valset = TASKS[key]['Val']
        testset = TASKS[key]['Test']
        # reset optimizer for new task
        task_pool += TASKS[key]['Class_ids']
        print("--------------------------------------------------")
        similarity_list = []
        similarity_tasks = str()
        if len(done_keys):
            count = 0
            for old_class in task_pool[:-2]:
                for new_class in TASKS[key]["Class_ids"]:
                    if old_class < new_class:
                        print(
                            # f"distance between class {old_class} and class {new_class} is: max:{similarity_matrix[old_class][new_class]['max']} , min:{similarity_matrix[old_class][new_class]['min']}")
                            f"distance between class {old_class} and class {new_class} is: mean:{similarity_matrix[old_class][new_class]['mean']}")
                        similarity_list.append(similarity_matrix[old_class][new_class]['mean'])
                    else:
                        print(
                            f"distance between class {old_class} and class {new_class} is: mean:{similarity_matrix[new_class][old_class]['mean']}")
                        similarity_list.append(similarity_matrix[new_class][old_class]['mean'])
                count += 2
                if count % 4 == 0:
                    similarity_tasks += f"TASK {(count / 4) - 1}: {np.mean(similarity_list[count - 4:count])} "
                log_dict['similarity_mean'] = np.mean(similarity_list)
                log_dict['similarity_tasks'] = similarity_tasks
        print("--------------------------------------------------")
        if len(done_keys):
            # update model
            # old_model = copy.deepcopy(model)
            # model = architecture_update.update_model(MODEL, model, data_shape,
            #                                          len(TASKS[key]["Class_ids"]), DEVICE)
            # layer = architecture_update.update_output_layer(model._modules['fc4'], 2)
            # model._modules['fc4'] = layer
            model.update_output_layer()
            if strategy == 1:
                model.wider()
            elif strategy == 2:
                # new_layer = architecture_update.extend_layer(model._modules['fc3'], torch.nn.ReLU)
                # model._modules['fc3'] = new_layer
                model.deeper(task_id)
            elif strategy == 3:
                if task_id % 2 == 0:
                    model.wider()
            elif strategy == 4:
                if task_id % 2 == 0:
                    model.deeper(task_id)
            elif strategy == 5:
                if task_id != 0:
                    if log_dict['similarity_mean'] < 0.60:
                        model.wider()
            elif strategy == 6:
                if task_id != 0:
                    if log_dict['similarity_mean'] < 0.60:
                        model.deeper(task_id)
            # del old_model
            model.to(DEVICE)
        print(f"With {len(trainset)} training and {len(valset)} validation batches.")
        optimizer = OPTIMIZER(model.parameters(), lr=LEARNING_RATE)
        scheduler = StepLR(optimizer, step_size=1, gamma=0.7)
        if EARLY_STOP:
            early_stopper = pytorch_util.Early_Stop(patience=5)

        for epoch in range(1, EPOCHS + 1):
            train(model, trainset, task_pool, epoch, optimizer,
                  DEVICE, AMOUNT_OF_TRAIN_BATCHES)
            # train(model, trainset, task_pool, epoch, optimizer,
            #       DEVICE, AMOUNT_OF_TRAIN_BATCHES,reg_loss)
            v_l, v_ac = validate(model, valset, task_pool, DEVICE)
            if EARLY_STOP:
                early_stopper(v_ac, v_l, model, epoch)
                if early_stopper.trigger:
                    break
            scheduler.step()

        log_dict = {}
        log_dict['number of classes learnt'] = len(task_pool)
        log_dict['task sequence'] = f'{task_pool}'
        log_dict['ratio of new/old samples'] = f"{TASKS[key]['NewSamples']}/{TASKS[key]['OldSamples']}"
        log_dict['action'] = strategy

        print("--------------------------------------------------")
        print(f"number of classes being learnt:{len(task_pool)},classes:{task_pool}")
        print("--------------------------------------------------")
        print(f"ratio of new/old classes'samples:{TASKS[key]['NewSamples']}/{TASKS[key]['OldSamples']}")
        print("--------------------------------------------------")
        if strategy == 1:
            print("action: expand the second last layer")
        elif strategy == 2:
            print("action: duplicate the second last layer")
        else:
            print("action: None")

        pre_acc = str()
        if len(done_keys):
            pre_v = []
            print("Previous tasks:")
            for k in done_keys:
                prev_v_l, prev_v_ac = validate(model, TASKS[k]['Val'], task_pool,
                                               DEVICE, verbose=False)
                pre_v.append(prev_v_ac)
                pre_acc += f"Task: {k}: {prev_v_ac} "
                print(f"Task: {k}: Accuracy:{prev_v_ac}, Loss:{prev_v_l}")
            log_dict['previous_val_acc_min'] = np.min(pre_v)
            log_dict['previous_val_acc_max'] = np.max(pre_v)
            log_dict['previous_val_acc_mean'] = np.mean(pre_v)
        print("\nCurrent task:")
        curr_v_l, curr_v_ac = validate(model, TASKS[key]['Val'], task_pool,
                                       DEVICE, verbose=False)
        cur_acc = f"{curr_v_ac}"
        print(f"Task: {key}: Accuracy:{curr_v_ac}, Loss:{curr_v_l}")
        print("--------------------------------------------------\n")
        log_dict['previous val acc'] = pre_acc
        log_dict['current val acc'] = cur_acc
        pre_acc = str()
        if len(done_keys):
            print("Previous tasks:")
            pre_t = []
            for k in done_keys:
                prev_t_l, prev_t_ac = test(model, TASKS[k]['Test'], task_pool,
                                           DEVICE)
                pre_t.append(prev_t_ac)
                pre_acc += f"Task: {k}: {prev_t_ac} "
                # print(f"Task: {k}: Accuracy:{prev_t_ac}, Loss:{prev_t_l}")
            log_dict['previous_test_acc_min'] = np.min(pre_t)
            log_dict['previous_test_acc_max'] = np.max(pre_t)
            log_dict['previous_test_acc_mean'] = np.mean(pre_t)
        print("\nCurrent task:")
        curr_t_l, curr_t_ac = test(model, testset, task_pool, DEVICE)
        cur_acc = f"{curr_t_ac}"
        # print(f"Task: {key}: Accuracy:{curr_t_ac}, Loss:{curr_t_l}")
        print("--------------------------------------------------\n")
        log_dict['previous test acc'] = pre_acc
        log_dict['current test acc'] = cur_acc
        done_keys.append(key)
        if task_id == list(TASKS.keys())[-1]:
            total_parameters = sum(param.numel() for param in model.parameters())
            log_dict['parameters'] = total_parameters
            torch.cuda.synchronize(DEVICE)
            end = time.time()
            log_dict['train time'] = int(end) - int(start)
        df = df.append(log_dict, ignore_index=True)
        if task_id != list(TASKS.keys())[-1]:
            # reset learning rate for optimizer after decay
            for g in optimizer.param_groups:
                g['lr'] = LEARNING_RATE
            model.to(DEVICE)

    df.to_csv('log.csv', index=False)
    return model
