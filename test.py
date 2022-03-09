import torch.optim as optim
import model
import train_test_val
import MNIST_util

tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)

optimizer = optim.Adam
EPOCHS = 50
# model_function = model.ConvNet1
model_function = model.FCNet
# model = models.resnet18(pretrained=False)
# print(model)
# class_similarity_dict(dataset='MNIST', device='cuda')
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=1
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=1
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=1
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=1
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=1
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=2
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=2
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=2
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=2
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=2
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=3
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=3
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=3
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=3
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=3
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=4
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=4
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=4
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=4
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=4
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=5
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=5
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=5
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=5
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=5
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=6
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=6
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=6
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=6
                                      )
tasks, testset = MNIST_util.get_MNIST_tasks(number_list=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], numbers_per_task=2,
                                            batch_size=32, val_split_perc=0.3, random_shuffle_tasks=True)
net = train_test_val.training_process(model_function, EPOCHS, tasks, testset, optimizer,
                                      LEARNING_RATE=0.0001,
                                      AMOUNT_OF_TRAIN_BATCHES=-1, device_flag="available",
                                      EARLY_STOP=True, strategy=6
                                      )
