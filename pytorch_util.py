import torch


class Early_Stop():
    def __init__(self, patience):
        self.patience = patience
        self.increasing_loss_count = 0
        self.best_epoch = 0
        self.best_val_acc = 0
        self.best_val_loss = None
        self.trigger = False

    def __call__(self, val_acc, val_loss, net, epoch):
        if self.best_val_loss == None:
            self.best_val_loss = val_loss
            self.best_val_acc = val_acc
            torch.save(net.state_dict(), 'model_state_checkpoint.pt')
        else:
            if self.best_val_loss <= val_loss:
                self.increasing_loss_count += 1
                if self.patience <= self.increasing_loss_count:
                    self.trigger = True
                    # restore last best model
                    net.load_state_dict(torch.load('model_state_checkpoint.pt'))
                    print(f"\nEarly stopping at epoch {epoch}. Restoring network of epoch {self.best_epoch}\n")

            else:
                self.best_val_loss = val_loss
                self.best_val_acc = val_acc
                self.best_epoch = epoch
                self.increasing_loss_count = 0
                # save current model as best model
                torch.save(net.state_dict(), 'model_state_checkpoint.pt')


def define_runtime_environment(flag="available"):
    if flag == "available":
        if torch.cuda.is_available():
            print(f"Available GPUs: {torch.cuda.device_count()}")
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    elif flag == "cuda":
        try:
            device = torch.device("cuda:0")
        except:
            print("GPU not found -> assigning CPU")
            device = torch.device("cpu")
    else:
        device = torch.device("cpu")

    return device


def get_class_indices(dataset, classes):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == classes:
            # if dataset.targets[i] in classes:
            indices.append(i)
    return indices


'''
def get_class_indices(dataset, classes, flag='test'):
    indices = []
    for i in range(len(dataset.targets)):
        if dataset.targets[i] == classes:
            # if dataset.targets[i] in classes:
            indices.append(i)
    return indices
'''
