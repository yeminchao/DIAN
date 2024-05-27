import os
import itertools
import numpy as np
import scipy.io as sio
from sklearn.metrics import classification_report, cohen_kappa_score
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from data_pre import data_pre
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
from fvcore.nn import FlopCountAnalysis, parameter_count_table


def plot_confusion_matrix(cm, savename, title="Confusion Matrix"):
    plt.figure(figsize=(12, 8), dpi=100)
    np.set_printoptions(precision=2)

    ind_array = np.arange(len(classes))
    x, y = np.meshgrid(ind_array, ind_array)
    for x_val, y_val in zip(x.flatten(), y.flatten()):
        c = cm[y_val][x_val]
        if c > 0.001:
            plt.text(
                x_val,
                y_val,
                "%0.2f" % (c,),
                color="red",
                fontsize=15,
                va="center",
                ha="center",
            )

    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.binary)
    plt.title(title)
    plt.colorbar()
    xlocations = np.array(range(len(classes)))
    plt.xticks(xlocations, classes, rotation=90)
    plt.yticks(xlocations, classes)
    plt.ylabel("Actual label")
    plt.xlabel("Predict label")

    tick_marks = np.array(range(len(classes))) + 0.5
    plt.gca().set_xticks(tick_marks, minor=True)
    plt.gca().set_yticks(tick_marks, minor=True)
    plt.gca().xaxis.set_ticks_position("none")
    plt.gca().yaxis.set_ticks_position("none")
    plt.grid(True, which="minor", linestyle="-")
    plt.gcf().subplots_adjust(bottom=0.15)


class HybridSN1(nn.Module):
    def __init__(self):
        super(HybridSN1, self).__init__()

        self.S_conv3d_1 = nn.Conv3d(1, 8, (7, 1, 1), stride=(2, 1, 1))

        self.S_conv3d_2 = nn.Conv3d(8, 16, (7, 1, 1), stride=(1, 1, 1))

        self.S_conv3d_3 = nn.Conv3d(16, 32, (7, 1, 1), stride=(2, 1, 1))

        self.S_conv3d_4 = nn.Conv3d(32, 64, (7, 1, 1), stride=(1, 1, 1))

        self.S_conv2d_1 = nn.Conv2d(832, 256, (2, 2))

        self.S_linear_1 = nn.Linear(4096, 2048)

        self.S_linear_2 = nn.Linear(2048, 512)

        self.relu = nn.ReLU()

    def forward(self, x1):
        out = self.S_conv3d_1(x1)

        out = F.relu(out)

        out = self.S_conv3d_2(out)

        out = F.relu(out)

        out = self.S_conv3d_3(out)

        out = F.relu(out)

        out = self.S_conv3d_4(out)

        out = F.relu(out)

        out = out.view(-1, out.shape[1] * out.shape[2], out.shape[3], out.shape[4])

        out = self.S_conv2d_1(out)

        out = F.relu(out)

        out = out.view(out.size(0), -1)

        out = F.relu(self.S_linear_1(out))

        out = self.S_linear_2(out)

        return out


class HybridSN2(nn.Module):
    def __init__(self):
        super(HybridSN2, self).__init__()

        self.T_conv3d_1 = nn.Conv3d(1, 8, (4, 1, 1), stride=(2, 1, 1))

        self.T_conv3d_2 = nn.Conv3d(8, 16, (4, 1, 1), stride=(1, 1, 1))

        self.T_conv3d_3 = nn.Conv3d(16, 32, (4, 1, 1), stride=(1, 1, 1))

        self.T_conv3d_4 = nn.Conv3d(32, 64, (4, 1, 1), stride=(1, 1, 1))

        self.T_conv2d_1 = nn.Conv2d(1664, 512, (2, 2))

        self.T_linear_1 = nn.Linear(8192, 2048)

        self.T_linear_2 = nn.Linear(2048, 512)

        self.relu = nn.ReLU()

    def forward(self, x2):
        out = self.T_conv3d_1(x2)

        out = F.relu(out)

        out = self.T_conv3d_2(out)

        out = F.relu(out)

        out = self.T_conv3d_3(out)

        out = F.relu(out)

        out = self.T_conv3d_4(out)

        out = F.relu(out)

        out = out.view(-1, out.shape[1] * out.shape[2], out.shape[3], out.shape[4])

        out = self.T_conv2d_1(out)

        out = F.relu(out)

        out = out.view(out.size(0), -1)

        out = F.relu(self.T_linear_1(out))

        out = self.T_linear_2(out)

        return out


class localattetion(nn.Module):
    def __init__(self):
        super(localattetion, self).__init__()
        self.linear_3 = nn.Linear(512, 256)
        self.linear_4 = nn.Linear(256, 512)

    def forward(self, out):
        out = F.relu(self.linear_3(out))
        out = self.linear_4(out)

        return out


class TwoStreamNet(nn.Module):
    def __init__(self, class_num):
        super(TwoStreamNet, self).__init__()
        self.linear_3 = nn.Linear(512, class_num)
        self.relu2 = nn.LogSoftmax(dim=1)

    def forward(self, out, out2):
        out = F.relu(out * out2)

        out = self.relu2(self.linear_3(out))

        return out


class Model(nn.Module):
    def __init__(self, class_num):
        super(Model, self).__init__()
        self.source_model = HybridSN1()
        self.target_model = HybridSN2()
        self.attention = localattetion()
        self.twostream_model = TwoStreamNet(class_num)

    def forward(self, out1, out2, flag):
        if flag == 0:
            source_out = self.source_model(out1)
            target_out = self.target_model(out2)
            source_attention = self.attention(source_out)
            target_attention = self.attention(target_out)
            f = EuclideanDistances(
                source_out * source_attention, target_out * target_attention
            )
            f2 = Distances(target_out * target_attention, target_out * target_attention)
            source_out = self.twostream_model(source_out, source_attention)
            target_out = self.twostream_model(target_out, target_attention)
        if flag == 1:
            target_out = self.target_model(out2)
            source_attention = torch.zeros(0)
            target_attention = self.attention(target_out)
            f = torch.zeros(0)
            f2 = torch.zeros(0)
            source_out = torch.zeros(0)
            target_out = self.twostream_model(target_out, target_attention)

        return source_out, target_out, source_attention, target_attention, f, f2


class My_loss(nn.Module):
    def __init__(self):
        super(My_loss, self).__init__()

    def forward(self, y1, y2, f, f2):
        same1 = torch.zeros([]).to(device)
        different1 = torch.zeros([]).to(device)
        same2 = torch.zeros([]).to(device)
        different2 = torch.zeros([]).to(device)
        m1 = torch.zeros([]).to(device)
        n1 = torch.zeros([]).to(device)
        m2 = torch.zeros([]).to(device)
        n2 = torch.zeros([]).to(device)
        for i in range(0, len(y2)):
            for label in torch.unique(y2):
                index = torch.where(y2 == label)[0]
                data2 = f2[i, index]
                if label == y2[i]:

                    m1 += torch.sum(data2)
                    same1 += len(index) - 1
                else:

                    n1 += torch.sum(data2)
                    different1 += len(index)
            for label in torch.unique(y1):
                index = torch.where(y1 == label)[0]
                data = f[index, i]

                if label == y2[i]:

                    m2 += torch.sum(data)
                    same2 += len(index)
                else:
                    n2 += torch.sum(data)
                    different2 += len(index)

        return ((m1 / same1) + (m2 / same2)) / ((n1 / different1) + (n2 / different2))


def EuclideanDistances(a, b):
    sq_a = a**2
    sum_sq_a = torch.sum(sq_a, dim=1).unsqueeze(1)
    sq_b = b**2
    sum_sq_b = torch.sum(sq_b, dim=1).unsqueeze(0)
    bt = b.t()
    return torch.sqrt(sum_sq_a + sum_sq_b - 2 * a.mm(bt))


def Distances(a, b):
    d = torch.zeros((a.shape[0], a.shape[0])).to(device)
    for i in range(0, a.shape[0]):
        for j in range(0, a.shape[0]):
            d[i][j] = torch.norm(a[i] - b[j], p=2)
    return d


class TrainDS1(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtrain1.shape[0]
        self.x_data = torch.FloatTensor(Xtrain1)
        self.y_data = torch.LongTensor(ytrain1)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len


class TrainDS2(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtrain2.shape[0]
        self.x_data = torch.FloatTensor(Xtrain2)
        self.y_data = torch.LongTensor(ytrain2)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len


class TestDS2(torch.utils.data.Dataset):
    def __init__(self):
        self.len = Xtest2.shape[0]
        self.x_data = torch.FloatTensor(Xtest2)
        self.y_data = torch.LongTensor(ytest2)

    def __getitem__(self, index):

        return self.x_data[index], self.y_data[index]

    def __len__(self):

        return self.len


def print_model_parm_nums(model):
    total = sum([param.nelement() for param in model.parameters()])
    print("  + Number of params: %.2fM" % (total / 1e6))


if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # load data
    time = 11
    classes = ["Trees", "Asphalt", "Bitumen", "Shadow", "Brick", "Meadow", "Soil"]
    for times in range(1, time):
        print(times)

        source_trainIdx = np.load("data1/model_source" + str(times) + ".npy")
        Xtrain1, ytrain1 = data_pre(
            X_source, y_source, source_trainIdx, require_test=False
        )

        target_trainIdx = np.load("data1/model_target" + str(times) + ".npy")
        Xtrain2, ytrain2, Xtest2, ytest2 = data_pre(X_target, y_target, target_trainIdx)

        patch_size = 5
        source_train_number = 400
        target_train_number = 5
        class_num = len(np.unique(y_target)) - 1

        trainset1 = TrainDS1()
        trainset2 = TrainDS2()
        testset2 = TestDS2()

        train_loader1 = torch.utils.data.DataLoader(
            dataset=trainset1, batch_size=2800, shuffle=True, num_workers=0
        )
        train_loader2 = torch.utils.data.DataLoader(
            dataset=trainset2, batch_size=70, shuffle=True, num_workers=0
        )
        test_loader2 = torch.utils.data.DataLoader(
            dataset=testset2, batch_size=256, shuffle=False, num_workers=0
        )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        net = Model(class_num).to(device)

        print(print_model_parm_nums(net))

        criterion1 = nn.CrossEntropyLoss()
        criterion2 = My_loss()
        optimizer1 = optim.Adam(net.parameters(), lr=0.001, weight_decay=1e-3)
        a = 0.7

        total_loss = 0
        cun = []
        for epoch in range(100):
            c = 0
            for i, data in enumerate(zip(train_loader1, train_loader2)):
                inputs1 = data[0][0].to(device)

                labels1 = data[0][1].to(device)
                inputs2 = data[1][0].to(device)
                labels2 = data[1][1].to(device)
                optimizer1.zero_grad()

                source_out, target_out, source_attention, target_attention, f, f2 = net(
                    inputs1, inputs2, 0
                )

                loss1 = criterion1(source_out, labels1)
                loss2 = criterion1(target_out, labels2)
                loss3 = a * criterion2(labels1, labels2, f, f2)
                loss = loss1 + loss2 + loss3

                loss.backward()

                optimizer1.step()
                if epoch % 50 == 0:
                    print("[Epoch: %d]  [current loss: %f]" % (epoch + 1, loss))
                c = loss.item()
            cun.append(c)

        torch.save(
            net.state_dict(), "./model1/target_classifier_net" + str(times) + ".pkl"
        )

        count = 0
        for inputs, _ in test_loader2:
            inputs1 = torch.zeros(
                1,
                inputs1.shape[1],
                inputs1.shape[2],
                inputs1.shape[3],
                inputs1.shape[4],
            ).to(device)
            inputs = inputs.to(device)
            outputs1, outputs, source_attention, target_attention, f, f2 = net(
                inputs1, inputs, 1
            )
            outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
            if count == 0:
                y_pred_test = outputs
                count = 1
            else:
                y_pred_test = np.concatenate((y_pred_test, outputs))
        classification = classification_report(ytest2, y_pred_test, digits=4)
        print(classification)
        cm = confusion_matrix(ytest2, y_pred_test)
        cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        plot_confusion_matrix(
            cm_normalized, "confusion_matrix.png", title="confusion matrix"
        )

        foo_fig = plt.gcf()
        foo_fig.savefig(
            "./data1/matrix-converted-to" + str(times) + ".pdf",
            dpi=1200,
            bbox_inches="tight",
            pad_inches=0.0,
        )
        a = cohen_kappa_score(ytest2, y_pred_test)
        print("Kappa", a)
