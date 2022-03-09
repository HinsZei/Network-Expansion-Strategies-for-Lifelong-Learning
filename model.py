import torch
import torch.nn as nn
import torch.nn.functional as F
import architecture_update
from torchvision.models import resnet18


class ConvNet(nn.Module):
    def __init__(self, data_shape, amnt_classes):
        super().__init__()
        inp_data_rows = data_shape[1]
        inp_data_cols = data_shape[2]
        channels = data_shape[0]
        classes = amnt_classes

        self.type = "conv"
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 32, 5)
        # self.conv3 = nn.Conv2d(32,64,5)
        # create random proxy data of same shape to run through the previous layer
        # to check the output shape ()

        # -1 denotes any shape of data
        x_to_define_linear = torch.randn(channels * inp_data_rows, inp_data_cols).view(-1, channels, inp_data_rows,
                                                                                       inp_data_cols)

        self._to_linear = None
        self.convs(x_to_define_linear)
        self.fc1 = nn.Linear(self._to_linear, 10)
        self.fc2 = nn.Linear(10, classes)

    def convs(self, x):
        """
        function to process convolutions and get the input shape for the fully connected layer
        """

        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # (2,2) is pooling shape
        x = F.max_pool2d(F.relu(self.conv2(x)), (1, 1))
        # x = F.max_pool2d(F.relu(self.conv3(x)), (1,1))

        # only fetch dims in first iteration
        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]

        return x

    def forward(self, x):
        x = self.convs(x)
        # print("after conv", x.shape)
        x = x.view(-1, self._to_linear)  # .view is reshape ... this flattens X before
        # x = torch.flatten(x,1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        # print("x after fc", x.shape)
        output = F.log_softmax(x, dim=1)

        return output


class FCNet(nn.Module):
    def __init__(self, data_shape, amnt_classes, amnt_neurons=16):
        super().__init__()
        self.type = "fc"
        self.fc1 = nn.Linear(data_shape[0] * data_shape[2] * data_shape[1], 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, amnt_neurons)
        self.fc4 = nn.Linear(amnt_neurons, amnt_classes)

    def wider(self):
        self.fc3, self.fc4 = architecture_update.expand_layer(self.fc3, self.fc4, round(self.fc3.out_features * 3 / 2))

    def deeper(self,count):
        self.fc3 = architecture_update.extend_layer(self.fc3, nn.ReLU,count)

    def update_output_layer(self):
        self.fc4 = architecture_update.update_output_layer(self.fc4, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output


class Regularisation(torch.nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model
        :param weight_decay:
        :param p:
        '''
        super(Regularisation, self).__init__()
        if weight_decay <= 0:
            print("param weight_decay can not <=0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)
        self.weight_info(self.weight_list)

    def to(self, device):
        '''
        :param device: cude or cpu
        :return:
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        :param model:
        :return:
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p=2):
        '''
        :param weight_list:
        :param p:
        :param weight_decay:
        :return:
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss


class Resnet_mod(nn.Module):
    def __init__(self, data_shape, amnt_classes, amnt_neurons=16):
        super().__init__()
        self.type = "conv"
        self.linear_neurons = None
        self.restnet = nn.Sequential(*list(resnet18(pretrained=True).children())[:-1])
        x_to_linear = torch.randn(data_shape[0] * data_shape[1], data_shape[2]).view(-1, data_shape[0], data_shape[1],
                                                                                     data_shape[2])
        self.cal_linear_neurons(x_to_linear)
        self.fc1 = nn.Linear(self.linear_neurons, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, amnt_classes)

    def cal_linear_neurons(self, x):
        x = F.relu(self.restnet(x))
        if self.linear_neurons is None:
            self.linear_neurons = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def wider(self):
        self.fc3, self.fc4 = architecture_update.expand_layer(self.fc3, self.fc4, round(self.fc3.out_features * 3 / 2))

    def deeper(self):
        self.fc3 = architecture_update.extend_layer(self.fc3, nn.ReLU)

    def update_output_layer(self):
        self.fc4 = architecture_update.update_output_layer(self.fc4, 2)

    def forward(self, x):
        x = F.relu(self.restnet(x))
        x = x.view(-1,self.linear_neurons)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        output = F.log_softmax(x, dim=1)
        return output
