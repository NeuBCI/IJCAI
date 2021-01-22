'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch as torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
# from visdom import Visdom
import time


class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5,padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5,padding=2)
        self.fc1   = nn.Linear(16*7*7, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out

class MyLeNet(nn.Module):
    def __init__(self):
        super(MyLeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 5,padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 5, padding=2)
        self.bn3 = nn.BatchNorm2d(16)
        self.conv_1x1 = nn.Conv2d(16, 10, 1)
        self.conv_1x1_1 = nn.Conv2d(16, 10, 1)
        np.random.seed(1)
        self.h1 = np.random.randint(-6, 7, 16)
        print(self.h1)
        np.random.seed(2)
        self.w1 = np.random.randint(-6, 7, 16)
        print(self.w1)
        self.conv_1x1_2 = nn.Conv2d(16, 10, 1)
        self.h2 = np.random.randint(-3, 4, 16)
        np.random.seed(3)
        self.w2 = np.random.randint(-3, 4, 16)
        np.random.seed(4)
        # self.h[0:16] = self.h[0:16] - 3
        # self.h[16:16 * 2] = self.h[16:16 * 2] - 3
        # self.h[16 * 2:16 * 3] = self.h[16 * 2:16 * 3] + 3
        # self.h[16 * 3:16 * 4] = self.h[16 * 3:16 * 4] + 3
        # self.w[0:16] = self.w[0:16] - 3
        # self.w[16:16 * 2] = self.w[16:16 * 2] + 3
        # self.w[16 * 2:16 * 3] = self.w[16 * 2:16 * 3] - 3
        # self.w[16 * 3:16 * 4] = self.w[16 * 3:16 * 4] + 3
        # self.fc = nn.Linear(32, 10)
        # self.fc1 = nn.Linear(32 * 2, 20)
        # self.fcbn = nn.BatchNorm2d(20)
        self.viz = Visdom(server='http://127.0.0.1', port=8097)
        assert self.viz.check_connection()

    def spacialInterleave(self, out, s, h, w, conv, a=[]):
        # out1 = torch.zeros((out.size(0), out.size(1), out.size(2)+s*2, out.size(3)+s*2)).cuda()
        out1 = F.pad(out, (s,s,s,s))
        out1[:,:,:,:] = 0
        for i in range(0, out.size(1)):
            out1[:, i, s + h[i]:s + out.size(2) + h[i], s +
                 w[i]:s + out.size(3) +
                 w[i]] = out[:, i % out.size(1), :, :]
        out1 = out1[:,:,s:-s,s:-s]
        out1 = conv(out1)

        for j in range(1):
            b = out1[j].cpu().numpy()
            for i in range(out1.size(1)):
                b[i] = b[i] / np.max(b[i])
                self.viz.image(b[i])
            time.sleep(1)
        locate = out1[0].view(out1.size(1), -1)
        loc = torch.max(locate, 1)[1].cpu().numpy()
        print(loc)
        h = out1.size(2)
        w = out1.size(3)

        out1 = F.max_pool2d(out1, out1.size(2))

        ll = torch.max(out1, 1)[1].cpu().numpy()[0]
        lh = loc[ll] // h
        lw = loc[ll] % h
        print('ll,lh,lw')
        print(ll)
        print(lh)
        print(lw)
        a = np.concatenate((a, a, a), axis=0)
        for i in range(out.size(1)):
            a[1, lh - self.h1[i], lw - self.w1[i]] = 0.5
        self.viz.image(a)
        time.sleep(3)
        return out1

    def getInformative(self, x):
        for j in range(1):
            a = x[j].cpu().numpy()
            a = (a - np.min(a[:])) / (np.max(a[:]) - np.min(a[:]))
            self.viz.image(a)
        k = np.zeros((3,1,3,3))
        k[0,0,:,2] = 1                  #1th kernel
        k[1,0,0,0] = 1                #2th kernel
        k[1,0,1:2,1] = 1
        k[2,0,0,0] = 1                #3th kernel
        k[2,0,1,1] = 1
        k[2,0,2,2] = 1
        k_hor=np.concatenate((k[1:0:-1,:,:,::-1],k),0)
        k_ver=k_hor.transpose((0,1,3,2))

        kernel_hor = torch.tensor(k_hor, dtype=torch.float).cuda()
        kernel_ver = torch.tensor(k_ver, dtype=torch.float).cuda()
        x1 = F.conv2d(x, kernel_hor, padding=1)
        x1 = F.pad(x1, (0, 1, 0, 0)) - F.pad(x1, (1, 0, 0, 0))
        x1 = x1[:, :, :, :-1]
        # s = x1.size()
        # s[1] = 1
        # for i in range(x1.size(1)):
        #     x1[:, i, :, :] = F.conv2d(x1[:,i,:,:].view(s), kernel_hor[i,:,:].view((1,3,3)), padding=1)
        print(x1.size())
        x1 = torch.cat((x1,-x1),1)
        x1 = F.relu(x1)
        x11 = torch.max_pool2d(x1,(3,1),stride=1,padding=(1,0))
        x11 = x11.eq(x1).float()
        x1 = x11 * x1
        x2 = F.conv2d(x, kernel_ver, padding=1)
        x2 = F.pad(x2, (0, 0, 0, 1)) - F.pad(x2, (0, 0, 1, 0))
        x2 = x2[:, :, :-1, :]
        x2 = torch.cat((x2, -x2), 1)
        x2 = F.relu(x2)
        x21 = torch.max_pool2d(x2, (3, 1), stride=1, padding=(1, 0))
        x21 = x21.eq(x2).float()
        x2 = x21 * x2
        x3 = torch.cat((x1,x2),1)
        print(x3.size())
        x4 = torch.max(x3,1,keepdim=True)[0]
        for j in range(1):
            b = x3[j].cpu().numpy()
            for i in range(x3.size(1)):
                b[i] = b[i] / np.max(b[i])
                self.viz.image(b[i])
            time.sleep(1)
        for j in range(1):
            b = x4[j].cpu().numpy()
            for i in range(x4.size(1)):
                b[i] = b[i] / np.max(b[i])
                self.viz.image(b[i])
            time.sleep(5)
        # mix =

    def forward(self, x):
        # self.viz.image(
        #     x.cpu().numpy
        # )
        # self.viz.image(np.random.rand(3, 512, 256),
        #           opts={
        #               'title': 'Random',
        #               'showlegend': True
        #           })
        # print(x.cpu().numpy())
        # a = x.cpu().numpy()[0]
        # b = np.concatenate((a,a,a),axis=0)
        # # print(a.shape)
        # self.viz.images(a*100, opts={
        #     'title': 'multi-images',
        # })
        # self.viz.bar(X=np.random.rand(20))
        # time.sleep(100)
        self.getInformative(x)
        return 0
        out = F.relu(self.bn1(self.conv1(x)))
        a=0
        for j in range(1):
            a = x[j].cpu().numpy()
            a = (a - np.min(a[:])) / (np.max(a[:]) - np.min(a[:]))
            self.viz.image(a)
            b = out[j].cpu().numpy()
            for i in range(out.size(1)):
                b[i] = b[i] / np.max(b[i])
                self.viz.image(b[i])

            b = self.conv1.weight.cpu().numpy()
            for i in range(out.size(1)):
                b[i] = (b[i] - np.min(b[i])) / (np.max(b[i]) - np.min(b[i]))
                self.viz.image(b[i])
            time.sleep(1)

        out1 = self.spacialInterleave(out, 6, self.h1, self.w1,
                                      self.conv_1x1_1, a=x[0].cpu().numpy())
        # out1 = self.conv_1x1_1(out)
        # out1 = F.max_pool2d(out1, out1.size(2))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn2(self.conv2(out)))
        out2 = self.spacialInterleave(out, 6, self.h2, self.w2,
                                      self.conv_1x1_2)
        # out2 = self.conv_1x1_2(out)
        # out2 = F.max_pool2d(out2, out2.size(2))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.bn3(self.conv3(out)))
        out = self.conv_1x1(out)


        # out = out1.cuda()
        # out = F.max_pool2d(out, 3, 1, 1)
        # locate = out.view(out.size(0), out.size(1), -1)
        # print(locate.size())
        # a = torch.max(locate, 2)[1].cpu().numpy()
        # h = a // out.size(3)
        # w = a % out.size(3)
        # height = torch.tensor(h, dtype=torch.float).cuda()
        # width = torch.tensor(w, dtype=torch.float).cuda()

        # a = a[1].item()
        # a = np.array(a)
        # h = a[1].div(out.size(3))
        # print(a.size())
        # print(a // out.size(3))
        # print(a % out.size(3))
        # print(a)
        # print(h)
        # print(w)
        out = F.avg_pool2d(out, out.size(2))



        # print(out.size())
        out = out.view(out.size(0), -1)
        out1 = out1.view(out1.size(0), -1)
        out2 = out2.view(out2.size(0), -1)
        # out = out1 + out2 + out
        out = out1
        # out1 = torch.cat((height, width), 1)
        # out1 = self.fc1(out1)
        # out1 = torch.unsqueeze(out1, -1)
        # out1 = torch.unsqueeze(out1, -1)
        # out1 = self.fcbn(out1)
        # # print(out1.size())
        # out1 = torch.squeeze(out1, -1)
        # out1 = torch.squeeze(out1, -1)
        # # print(out1.size())
        # out = torch.cat((out, out1), 1)
        # # print(out.size())
        # out = self.fc(out)
        return out


class MyLeNetSpaceShuffle(nn.Module):
    def __init__(self):
        super(MyLeNetSpaceShuffle, self).__init__()
        self.conv11 = nn.Conv2d(1, 2, (3, 1), padding=2)
        self.conv12 = nn.Conv2d(2, 2, (3, 1), padding=2, groups=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv21 = nn.Conv2d(16, 2, 3, padding=2)
        self.conv22 = nn.Conv2d(16, 2, 3, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv31 = nn.Conv2d(16, 10, (3, 1), padding=2)
        self.conv32 = nn.Conv2d(16, 10, (3, 1), padding=2)
        # self.fc = nn.Linear(32, 10)
        # self.fc1 = nn.Linear(32 * 2, 20)
        # self.fcbn = nn.BatchNorm2d(20)

    def rotateConv(self,x,c1,c2):

        l = [[(),(),()] for i in range(8)]
        # l[0] = [[1, -1], [0, 0], [-1, 1]]
        l[0] = [(0,2,2,0),(1,1,1,1),(2,0,0,2)]
        # l[1] = [[0, -1], [0, 0], [0, 0]]
        l[1] = [(1,1,2,0),(1,1,1,1),(1,1,1,1)]
        # l[2] = [[-1, -1], [0, 0], [1, 1]]
        l[2] = [(2,0,2,0),(1,1,1,1),(0,2,0,2)]
        # l[3] = [[-1, 0], [0, 0], [1, 0]]
        l[3] = [(2,0,1,1),(1,1,1,1),(0,2,1,1)]
        # l[4] = [[-1, 1], [0, 0], [1, -1]]
        l[4] = [(2,0,0,2),(1,1,1,1),(0,2,2,0)]
        # l[5] = [[0, 1], [0, 0], [0, -1]]
        l[5] = [(1,1,0,2),(1,1,1,1),(1,1,2,0)]
        # l[6] = [[1, 1], [0, 0], [-1, -1]]
        l[6] = [(0,2,0,2),(1,1,1,1),(2,0,2,0)]
        # l[7] = [[1, 0], [0, 0], [-1, 0]]
        l[7] = [(0,2,1,1),(1,1,1,1),(2,0,1,1)]
        print(c1.weight.size())
        print(c1.weight.size(2))

        w1 = torch.split(c1.weight, 1, dim=2)

        x0 = F.conv2d(x, w1[0])
        x1 = F.conv2d(x, w1[1])
        x2 = F.conv2d(x, w1[2])
        rotate = [0 for i  in range(8)]

        for i in range(8):
            rotate[i] = F.pad(x0, l[i][0]) + F.pad(x1, l[i][1]) + F.pad(
                x2, l[i][2])
            print(rotate[i].size())


        print(c2.weight.size())
        w1 = torch.split(c2.weight, 1, dim=2)

        for i in range(8):
            x0 = F.conv2d(rotate[i], w1[0], groups=2)
            x1 = F.conv2d(rotate[i], w1[1], groups=2)
            x2 = F.conv2d(rotate[i], w1[2], groups=2)
            j = (i+4)%8
            rotate[i] = F.pad(x0, l[j][0]) + F.pad(x1, l[j][1]) + F.pad(
                x2, l[j][2])
            print(rotate[i].size())

    def rotateConv_all(self, x, c1, c2):
        def rotate(x,r=8):
            a = x.size(0)
            x = torch.split(x, a/r, dim=0)
            y = torch.cat(x[1:],dim=0)
            y = torch.cat((y, x[0]), dim=0)
            return y

        l = [[(), (), ()] for i in range(8)]
        # l[0] = [[1, -1], [0, 0], [-1, 1]]
        l[0] = [(0, 2, 2, 0), (1, 1, 1, 1), (2, 0, 0, 2)]
        # l[1] = [[0, -1], [0, 0], [0, 0]]
        l[1] = [(1, 1, 2, 0), (1, 1, 1, 1), (1, 1, 1, 1)]
        # l[2] = [[-1, -1], [0, 0], [1, 1]]
        l[2] = [(2, 0, 2, 0), (1, 1, 1, 1), (0, 2, 0, 2)]
        # l[3] = [[-1, 0], [0, 0], [1, 0]]
        l[3] = [(2, 0, 1, 1), (1, 1, 1, 1), (0, 2, 1, 1)]
        # l[4] = [[-1, 1], [0, 0], [1, -1]]
        l[4] = [(2, 0, 0, 2), (1, 1, 1, 1), (0, 2, 2, 0)]
        # l[5] = [[0, 1], [0, 0], [0, -1]]
        l[5] = [(1, 1, 0, 2), (1, 1, 1, 1), (1, 1, 2, 0)]
        # l[6] = [[1, 1], [0, 0], [-1, -1]]
        l[6] = [(0, 2, 0, 2), (1, 1, 1, 1), (2, 0, 2, 0)]
        # l[7] = [[1, 0], [0, 0], [-1, 0]]
        l[7] = [(0, 2, 1, 1), (1, 1, 1, 1), (2, 0, 1, 1)]
        print(c1.weight.size())
        print(c1.weight.size(2))

        rotate = [0 for i in range(8)]

        for i in range(8):
            if i != 0:
                w1 = rotate(w1)
            w = torch.split(c2.weight, 1, dim=2)
            x0 = F.conv2d(x, w[0])
            x1 = F.conv2d(x, w[1])
            x2 = F.conv2d(x, w[2])
            rotate[i] = F.pad(x0, l[i][0]) + F.pad(x1, l[i][1]) + F.pad(
                x2, l[i][2])
            print(rotate[i].size())

        print(c2.weight.size())
        w1 = torch.split(c2.weight, 1, dim=2)

        for i in range(8):
            if i != 0:
                w1 = rotate(w1)
            w1 = torch.split(c2.weight, 1, dim=2)
            x0 = F.conv2d(rotate[i], w1[0], groups=2)
            x1 = F.conv2d(rotate[i], w1[1], groups=2)
            x2 = F.conv2d(rotate[i], w1[2], groups=2)
            j = (i + 4) % 8
            rotate[i] = F.pad(x0, l[j][0]) + F.pad(x1, l[j][1]) + F.pad(
                x2, l[j][2])
            print(rotate[i].size())


    def forward(self, x):
        self.rotateConv(x,self.conv11,self.conv12)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        # out1 = torch.zeros((out.size(0), out.size(1), out.size(2) + 6,
        #                     out.size(3) + 6)).cuda()
        # for i in range(0, out.size(1)):
        #     out1[:, i, 3 + self.h1[i]:3 + out.size(2) + self.h1[i], 3 +
        #          self.w1[i]:3 + out.size(3) + self.w1[i]] = out[:, i, :, :]
        # out = out1[:,:,3:-3,3:-3]
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        # out1 = torch.zeros((out.size(0), out.size(1), out.size(2) + 6,
        #                     out.size(3) + 6)).cuda()
        # for i in range(0, out.size(1)):
        #     out1[:, i, 3 + self.h2[i]:3 + out.size(2) + self.h2[i], 3 +
        #          self.w2[i]:3 + out.size(3) + self.w2[i]] = out[:, i, :, :]
        # out = out1[:, :, 3:-3, 3:-3]
        out = F.relu(self.bn3(self.conv3(out)))
        # out1 = torch.zeros((out.size(0), out.size(1), out.size(2) + 6,
        #                     out.size(3) + 6)).cuda()
        # for i in range(0, out.size(1)):
        #     out1[:, i, 3 + self.h3[i]:3 + out.size(2) + self.h3[i], 3 +
        #          self.w3[i]:3 + out.size(3) + self.w3[i]] = out[:, i, :, :]
        # out = out1[:, :, 3:-3, 3:-3]
        out = F.max_pool2d(out, 3, 1, 1)
        out = self.conv_1x1(out)
        out = F.max_pool2d(out, out.size(2))

        # print(out.size())
        out = out.view(out.size(0), -1)
        return out


class MyLeNetRotateInvariant(nn.Module):
    def __init__(self):
        super(MyLeNetRotateInvariant, self).__init__()
        self.r = 8
        self.conv1 = nn.Conv2d(1, 6, 3, padding=2)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6 * self.r, 16, 3, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16 * self.r, 10, 3, padding=2)
        # self.fc = nn.Linear(32, 10)
        # self.fc1 = nn.Linear(32 * 2, 20)
        # self.fcbn = nn.BatchNorm2d(20)

    def BN(self,x,BN):
        # print('BN')
        a = x.size(1)
        # print(a)
        x = torch.split(x,int(a/self.r),dim=1)
        x = torch.cat(x,dim=2)
        # print(x.size())
        x = BN(x)
        a = x.size(2)
        x = torch.split(x,int(a/self.r),dim=2)
        x = torch.cat(x, dim=1)
        # print(x.size())
        return x

    def rotateMax(self,x):
        # print('max')
        s0,s1,s2,s3 = x.size()
        # print(s1)
        x = x.view(s0,int(s1/self.r),self.r,s2,s3)
        x = torch.max(x,dim=2,keepdim=False)[0]
        return x

    def rotateConv(self, x, c, seprate=False):
        def rotateChannel(x, r=self.r):
            # print('rotate')
            a = x.size(1)
            # print(x.size())
            x = torch.split(x, int(a / r), dim=1)
            y = torch.cat(x[0:-1], dim=1)
            y = torch.cat((x[-1],y), dim=1)
            return y

        l = [[(), (), ()] for i in range(8)]
        # l[0] = [[1, -1], [0, 0], [-1, 1]]
        l[0] = [(0, 2, 2, 0), (1, 1, 1, 1), (2, 0, 0, 2)]
        # l[1] = [[0, -1], [0, 0], [0, 0]]
        l[1] = [(1, 1, 2, 0), (1, 1, 1, 1), (1, 1, 1, 1)]
        # l[2] = [[-1, -1], [0, 0], [1, 1]]
        l[2] = [(2, 0, 2, 0), (1, 1, 1, 1), (0, 2, 0, 2)]
        # l[3] = [[-1, 0], [0, 0], [1, 0]]
        l[3] = [(2, 0, 1, 1), (1, 1, 1, 1), (0, 2, 1, 1)]
        # l[4] = [[-1, 1], [0, 0], [1, -1]]
        l[4] = [(2, 0, 0, 2), (1, 1, 1, 1), (0, 2, 2, 0)]
        # l[5] = [[0, 1], [0, 0], [0, -1]]
        l[5] = [(1, 1, 0, 2), (1, 1, 1, 1), (1, 1, 2, 0)]
        # l[6] = [[1, 1], [0, 0], [-1, -1]]
        l[6] = [(0, 2, 0, 2), (1, 1, 1, 1), (2, 0, 2, 0)]
        # l[7] = [[1, 0], [0, 0], [-1, 0]]
        l[7] = [(0, 2, 1, 1), (1, 1, 1, 1), (2, 0, 1, 1)]
        # print('rotateConv')
        rotate = [0 for i in range(self.r)]
        wr = c.weight
        # print(wr.size())
        s1 = wr.size(2)
        s2 = wr.size(3)
        if seprate:
            w = torch.split(wr, 1, dim=2)
            # print(len(w))
            w = [torch.split(w[i], 1, dim=3) for i in range(s1)]
            o = [[F.conv2d(x, w[j][i]) for i in range(s2)] for j in range(s1)]
            for i in range(self.r):
                o1 = [[0 for j in range(s2)] for k in range(s1)]
                for j in range(s2):
                    for k in range(s1):
                        o1[j][k] = F.pad(F.pad(o[j][k],l[i][j]),l[int((i+self.r/2)%self.r)][k])
                rotate[i] = o1[0][0] + o1[0][1] + o1[0][2] + o1[1][0] + o1[1][1] + o1[1][2] + o1[2][0] + o1[2][1] + o1[2][2]
        else:
            for i in range(self.r):
                if i != 0:
                    wr = rotateChannel(wr)

                w = torch.split(wr, 1, dim=2)
                # print(len(w))
                w = [torch.split(w[i], 1, dim=3) for i in range(s1)]
                o = [[F.conv2d(x, w[j][i]) for i in range(s2)] for j in range(s1)]
                o1 = [[0 for j in range(s2)] for k in range(s1)]
                for j in range(s2):
                    for k in range(s1):
                        o1[j][k] = F.pad(F.pad(o[j][k],l[i][j]),l[int((i+self.r/2)%self.r)][k])
                rotate[i] = o1[0][0] + o1[0][1] + o1[0][2] + o1[1][0] + o1[1][1] + o1[1][2] + o1[2][0] + o1[2][1] + o1[2][2]
        rotate = torch.cat(rotate, dim=1)
        # print(rotate.size())
        rotate = rotate[:, :, 2:-2, 2:-2]
        # print(rotate.size())

        return rotate


    def forward(self, x):
        out = self.rotateConv(x, self.conv1, seprate=True)
        # print(out.size())
        out = F.relu(self.BN(out,self.bn1))
        # print(out.size())
        out = F.max_pool2d(out, 2)
        # print(out.size())
        out = self.rotateConv(out, self.conv2)
        # print(out.size())
        out = F.relu(self.BN(out, self.bn2))
        # print(out.size())
        out = F.max_pool2d(out, 2)
        # print(out.size())
        out = self.rotateConv(out, self.conv3)
        # print(out.size())
        out = self.rotateMax(out)
        # print(out.size())

        out = F.max_pool2d(out, out.size(2))
        # print(out.size())

        # print(out.size())
        out = out.view(out.size(0), -1)
        return out


class MyLeNetSpaceShuffle(nn.Module):
    def __init__(self):
        super(MyLeNetSpaceShuffle, self).__init__()
        self.conv11 = nn.Conv2d(1, 2, (3, 1), padding=2)
        self.conv12 = nn.Conv2d(2, 2, (3, 1), padding=2, groups=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv21 = nn.Conv2d(16, 2, 3, padding=2)
        self.conv22 = nn.Conv2d(16, 2, 3, padding=2)
        self.bn2 = nn.BatchNorm2d(16)
        self.conv31 = nn.Conv2d(16, 10, (3, 1), padding=2)
        self.conv32 = nn.Conv2d(16, 10, (3, 1), padding=2)
        # self.fc = nn.Linear(32, 10)
        # self.fc1 = nn.Linear(32 * 2, 20)
        # self.fcbn = nn.BatchNorm2d(20)

    def rotateConv(self, x, c1, c2):

        l = [[(), (), ()] for i in range(8)]
        # l[0] = [[1, -1], [0, 0], [-1, 1]]
        l[0] = [(0, 2, 2, 0), (1, 1, 1, 1), (2, 0, 0, 2)]
        # l[1] = [[0, -1], [0, 0], [0, 0]]
        l[1] = [(1, 1, 2, 0), (1, 1, 1, 1), (1, 1, 1, 1)]
        # l[2] = [[-1, -1], [0, 0], [1, 1]]
        l[2] = [(2, 0, 2, 0), (1, 1, 1, 1), (0, 2, 0, 2)]
        # l[3] = [[-1, 0], [0, 0], [1, 0]]
        l[3] = [(2, 0, 1, 1), (1, 1, 1, 1), (0, 2, 1, 1)]
        # l[4] = [[-1, 1], [0, 0], [1, -1]]
        l[4] = [(2, 0, 0, 2), (1, 1, 1, 1), (0, 2, 2, 0)]
        # l[5] = [[0, 1], [0, 0], [0, -1]]
        l[5] = [(1, 1, 0, 2), (1, 1, 1, 1), (1, 1, 2, 0)]
        # l[6] = [[1, 1], [0, 0], [-1, -1]]
        l[6] = [(0, 2, 0, 2), (1, 1, 1, 1), (2, 0, 2, 0)]
        # l[7] = [[1, 0], [0, 0], [-1, 0]]
        l[7] = [(0, 2, 1, 1), (1, 1, 1, 1), (2, 0, 1, 1)]
        print(c1.weight.size())
        print(c1.weight.size(2))

        w1 = torch.split(c1.weight, 1, dim=2)

        x0 = F.conv2d(x, w1[0])
        x1 = F.conv2d(x, w1[1])
        x2 = F.conv2d(x, w1[2])
        rotate = [0 for i in range(8)]

        for i in range(8):
            rotate[i] = F.pad(x0, l[i][0]) + F.pad(x1, l[i][1]) + F.pad(
                x2, l[i][2])
            print(rotate[i].size())

        print(c2.weight.size())
        w1 = torch.split(c2.weight, 1, dim=2)

        for i in range(8):
            x0 = F.conv2d(rotate[i], w1[0], groups=2)
            x1 = F.conv2d(rotate[i], w1[1], groups=2)
            x2 = F.conv2d(rotate[i], w1[2], groups=2)
            j = (i + 4) % 8
            rotate[i] = F.pad(x0, l[j][0]) + F.pad(x1, l[j][1]) + F.pad(
                x2, l[j][2])
            print(rotate[i].size())

    def rotateConv_all(self, x, c1, c2):
        def rotate(x, r=8):
            a = x.size(0)
            x = torch.split(x, a / r, dim=0)
            y = torch.cat(x[1:], dim=0)
            y = torch.cat((y, x[0]), dim=0)
            return y

        l = [[(), (), ()] for i in range(8)]
        # l[0] = [[1, -1], [0, 0], [-1, 1]]
        l[0] = [(0, 2, 2, 0), (1, 1, 1, 1), (2, 0, 0, 2)]
        # l[1] = [[0, -1], [0, 0], [0, 0]]
        l[1] = [(1, 1, 2, 0), (1, 1, 1, 1), (1, 1, 1, 1)]
        # l[2] = [[-1, -1], [0, 0], [1, 1]]
        l[2] = [(2, 0, 2, 0), (1, 1, 1, 1), (0, 2, 0, 2)]
        # l[3] = [[-1, 0], [0, 0], [1, 0]]
        l[3] = [(2, 0, 1, 1), (1, 1, 1, 1), (0, 2, 1, 1)]
        # l[4] = [[-1, 1], [0, 0], [1, -1]]
        l[4] = [(2, 0, 0, 2), (1, 1, 1, 1), (0, 2, 2, 0)]
        # l[5] = [[0, 1], [0, 0], [0, -1]]
        l[5] = [(1, 1, 0, 2), (1, 1, 1, 1), (1, 1, 2, 0)]
        # l[6] = [[1, 1], [0, 0], [-1, -1]]
        l[6] = [(0, 2, 0, 2), (1, 1, 1, 1), (2, 0, 2, 0)]
        # l[7] = [[1, 0], [0, 0], [-1, 0]]
        l[7] = [(0, 2, 1, 1), (1, 1, 1, 1), (2, 0, 1, 1)]
        print(c1.weight.size())
        print(c1.weight.size(2))

        rotate = [0 for i in range(8)]

        for i in range(8):
            if i != 0:
                w1 = rotate(w1)
            w = torch.split(c2.weight, 1, dim=2)
            x0 = F.conv2d(x, w[0])
            x1 = F.conv2d(x, w[1])
            x2 = F.conv2d(x, w[2])
            rotate[i] = F.pad(x0, l[i][0]) + F.pad(x1, l[i][1]) + F.pad(
                x2, l[i][2])
            print(rotate[i].size())

        print(c2.weight.size())
        w1 = torch.split(c2.weight, 1, dim=2)

        for i in range(8):
            if i != 0:
                w1 = rotate(w1)
            w1 = torch.split(c2.weight, 1, dim=2)
            x0 = F.conv2d(rotate[i], w1[0], groups=2)
            x1 = F.conv2d(rotate[i], w1[1], groups=2)
            x2 = F.conv2d(rotate[i], w1[2], groups=2)
            j = (i + 4) % 8
            rotate[i] = F.pad(x0, l[j][0]) + F.pad(x1, l[j][1]) + F.pad(
                x2, l[j][2])
            print(rotate[i].size())

    def forward(self, x):
        self.rotateConv(x, self.conv11, self.conv12)
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        # out1 = torch.zeros((out.size(0), out.size(1), out.size(2) + 6,
        #                     out.size(3) + 6)).cuda()
        # for i in range(0, out.size(1)):
        #     out1[:, i, 3 + self.h1[i]:3 + out.size(2) + self.h1[i], 3 +
        #          self.w1[i]:3 + out.size(3) + self.w1[i]] = out[:, i, :, :]
        # out = out1[:,:,3:-3,3:-3]
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        # out1 = torch.zeros((out.size(0), out.size(1), out.size(2) + 6,
        #                     out.size(3) + 6)).cuda()
        # for i in range(0, out.size(1)):
        #     out1[:, i, 3 + self.h2[i]:3 + out.size(2) + self.h2[i], 3 +
        #          self.w2[i]:3 + out.size(3) + self.w2[i]] = out[:, i, :, :]
        # out = out1[:, :, 3:-3, 3:-3]
        out = F.relu(self.bn3(self.conv3(out)))
        # out1 = torch.zeros((out.size(0), out.size(1), out.size(2) + 6,
        #                     out.size(3) + 6)).cuda()
        # for i in range(0, out.size(1)):
        #     out1[:, i, 3 + self.h3[i]:3 + out.size(2) + self.h3[i], 3 +
        #          self.w3[i]:3 + out.size(3) + self.w3[i]] = out[:, i, :, :]
        # out = out1[:, :, 3:-3, 3:-3]
        out = F.max_pool2d(out, 3, 1, 1)
        out = self.conv_1x1(out)
        out = F.max_pool2d(out, out.size(2))

        # print(out.size())
        out = out.view(out.size(0), -1)
        return out


class MyLeNetRotateInvariantNew(nn.Module):
    def __init__(self):
        super(MyLeNetRotateInvariantNew, self).__init__()
        self.r = 8
        self.conv1 = nn.Conv2d(1, 16, 3, padding=2)
        self.bn1 = nn.BatchNorm2d(16, momentum=0.1 / self.r)
        self.conv2 = nn.Conv2d(2*self.r, 16, 3, padding=2)
        self.bn2 = nn.BatchNorm2d(16, momentum=0.1 / self.r)
        self.conv3 = nn.Conv2d(16, 10, 3, padding=2)
        # self.fc = nn.Linear(32, 10)
        # self.fc1 = nn.Linear(32 * 2, 20)
        # self.fcbn = nn.BatchNorm2d(20)
        self.viz = Visdom(server='http://127.0.0.1', port=8097)
        assert self.viz.check_connection()

    def BN(self, x, BN):
        # print('BN')
        a = x.size(1)
        # print(a)
        x = torch.split(x, int(a / self.r), dim=1)
        x = [BN(x[i]) for i in range(self.r)]
        x = torch.cat(x, dim=1)
        # print(x.size())
        return x

    def rotateMax(self, x):
        # print('max')
        s0, s1, s2, s3 = x.size()
        # print(s1)
        x = x.view(s0, int(s1 / self.r), self.r, s2, s3)
        x = torch.max(x, dim=2, keepdim=False)[0]
        return x

    def rotatesplit(self, x, n):
        s0,s1,s2,s3 = x.size()
        x = x.view(s0, int(s1 / self.r), self.r, s2, s3)
        x = torch.split(x, n, dim=1)
        return [a.view(s0,-1,s2,s3) for a in x]

    def nms(self, x):
        s0, s1, s2, s3 = x.size()
        x = x.view(s0, int(s1 / self.r), self.r, s2, s3)
        y = torch.max(x, dim=2, keepdim=True)[0]
        y = torch.cat([y for i in range(self.r)], dim=2)
        y = y.ne(x).float()
        x = x*y
        return x.sum(dim=0).sum(dim=0).mean()

    def ds(self, x):
        s0, s1, s2, s3 = x.size()
        x = x.view(s0, int(s1 / self.r), self.r, s2, s3)
        y = torch.mean(x, dim=2, keepdim=True)
        y = torch.cat([y for i in range(self.r)], dim=2)
        x = (x - y).abs()
        # y = y.ne(x).float()
        # x = x*y
        return x.sum(dim=0).sum(dim=0).mean()

    def nms_ds(self, x, n):
        a = self.rotatesplit(x, n)
        return self.nms(a[0])+self.ds(a[1])

    def rotateConv(self, x, c, scale, seprate=False):
        def rotateChannel(x, r=self.r):
            # print('rotate')
            a = x.size(1)
            # print(x.size())
            x = torch.split(x, int(a / r), dim=1)
            y = torch.cat(x[0:-1], dim=1)
            y = torch.cat((x[-1], y), dim=1)
            return y

        def rotateMatrix(theta):
            return np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])

        wr = c.weight
        s1 = wr.size(2)
        s2 = wr.size(3)
        locx = np.array([[j-(s2-1)/2 for j in range(s2)] for i in range(s1)])
        locy = np.array([[i-(s1-1)/2 for j in range(s2)] for i in range(s1)])
        loc = np.concatenate((locx.reshape(-1,1),locy.reshape(-1,1)),axis=1)
        loc = loc * scale
        # print('rotateConv')
        rotate = [0 for i in range(self.r)]
        # print(wr.size())
        if seprate:
            w = torch.split(wr, 1, dim=2)
            w = [torch.split(w[i], 1, dim=3) for i in range(s1)]
            o = [F.conv2d(x, w[j][i]) for i in range(s2) for j in range(s1)]
            for i in range(self.r):
                l = np.dot(loc,rotateMatrix(2*np.pi/self.r*i))
                l = np.around(l)
                m = int(np.max(l))
                p = [(int(l[j, 0] + m), int(m-l[j, 0]), int(l[j, 1] + m),
                      int(m-l[j, 1])) for j in range(s1 * s2)]
                o1 = [
                    F.pad(o[j], p[j]).unsqueeze(0)
                    for j in range(s1 * s2)
                ]
                o1 = torch.cat(o1,0)
                rotate[i] = torch.sum(o1, dim=0,
                                      keepdim=False)[:, :, m:-m, m:-m]
        else:
            for i in range(self.r):
                if i != 0:
                    wr = rotateChannel(wr)

                w = torch.split(wr, 1, dim=2)
                w = [torch.split(w[i], 1, dim=3) for i in range(s1)]
                o = [F.conv2d(x, w[j][i]) for i in range(s2) for j in range(s1)]
                l = np.dot(loc, rotateMatrix(2 * np.pi / self.r * i))
                l = np.around(l)
                m = int(np.max(l))
                p = [(int(l[j, 0] + m), int(m - l[j, 0]), int(l[j, 1] + m),
                      int(m - l[j, 1])) for j in range(s1 * s2)]
                o1 = [
                    F.pad(o[j], p[j]).unsqueeze(0)
                    for j in range(s1 * s2)
                ]
                o1 = torch.cat(o1, 0)
                rotate[i] = torch.sum(o1, dim=0,
                                      keepdim=False)[:, :, m:-m, m:-m]

        rotate = torch.cat(rotate, dim=1)
        # print(rotate.size())
        # rotate = rotate[:, :, 2:-2, 2:-2]
        # print(rotate.size())

        return rotate

    def forward(self, x):
        # b = x.cpu().numpy()
        # bmin = np.min(b)
        # bmax = np.max(b)
        # for i in range(1):
        #     b[i] = (b[i] - np.min(b[i])) / (np.max(b[i]) - np.min(b[i]))
        #     self.viz.image(b[i])
        # out = self.rotateConv(x, self.conv1, 1, seprate=True)
        # out = F.relu(self.BN(out, self.bn1))
        out = self.conv1(x)
        out = F.relu(self.bn1(out))
        n = self.nms(out)
        # # out = self.rotateMax(out)

        # print(out.size())
        # n = self.nms_ds(out,1)
        # n = self.nms(out)

        # b = out[0].cpu().numpy()
        # print(len(b))
        # print(len(b[0]))
        # c = int(out.size(1)/self.r)
        # print(c)
        # for i in range(c):
        #     for j in range(self.r):
        #         b[i+j*c] = b[i+j*c] / np.max(b[i+j*c])
        #         self.viz.image(b[i+j*c])

        # print(self.conv1.weight.size())
        # print(self.conv1.weight)
        # b = self.conv1.weight.cpu().numpy()
        # bmin = np.min(b)
        # bmax = np.max(b)
        # for i in range(self.conv1.weight.size(0)):
        #     print(b[i])
        #     b[i] = (b[i] - bmin) / (bmax - bmin)
        #     self.viz.image(b[i])
        # time.sleep(20)
        # # out = F.relu(self.bn1(out))
        # # # print(out.size())
        # # out = F.max_pool2d(out, 2, stride=(1, 1))
        out = F.max_pool2d(out, 2)
        # # # print(out.size())
        # # self.r = 16
        # out = self.rotateConv(out, self.conv2, 1)
        # # print(out.size())
        # out = F.relu(self.BN(out, self.bn2))
        # n = self.nms(out) +n
        out = self.conv2(out)
        out = F.relu(self.bn2(out))



        # out = self.rotateMax(out)
        # out = F.relu(self.bn2(out))
        # # print(out.size())
        # out = F.max_pool2d(out, 4, stride=(1, 1))
        out = F.max_pool2d(out, 2)
        # # # print(out.size())
        # out = self.rotateConv(out, self.conv3, 1)
        # n = self.nms(out) +n
        # # print(out.size())
        # out = self.rotateMax(out)
        # # # print(out.size())
        out = self.conv3(out)

        out = F.max_pool2d(out, out.size(2))
        # # print(out.size())

        # # print(out.size())
        out = out.view(out.size(0), -1)
        return out,n

class Rotate_Conv(nn.Module):
    def __init__(self, in_channel, out_channel,ks, seperate=False):
        super(Rotate_Conv, self).__init__()
        self.r = 8
        self.seperate = seperate
        self.conv = nn.Conv2d(in_channel, int(out_channel/self.r),ks)
        self.bn = nn.BatchNorm2d(int(out_channel / self.r),
                                 momentum=0.1 / self.r)
        self.get_rot_pad()

    def get_rot_pad(self):
        self.a = self.cal_rot_pad(self.conv, 1, seprate=self.seperate)
        return 0

    def get_weight(self):
        self.w = self.cal_weight(self.conv, 1, seprate=self.seperate)
        return 0

    def BN(self, x, bn):
        # print('BN')
        a = x.size(1)
        # print(a)
        x = torch.split(x, int(a / self.r), dim=1)
        x = [bn(x[i]) for i in range(self.r)]
        x = torch.cat(x, dim=1)
        # print(x.size())
        return x

    def cal_rot_pad(self, c, scale, seprate=False):
        def rotateMatrix(theta):
            return np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])

        wr = c.weight
        s1 = wr.size(2)
        s2 = wr.size(3)
        locx = np.array([[j - (s2 - 1) / 2 for j in range(s2)]
                         for i in range(s1)])
        locy = np.array([[i - (s1 - 1) / 2 for j in range(s2)]
                         for i in range(s1)])
        loc = np.concatenate((locx.reshape(-1, 1), locy.reshape(-1, 1)),
                             axis=1)
        loc = loc * scale
        rotate = [0 for i in range(self.r)]
        if seprate:
            for i in range(self.r):
                l = np.dot(loc, rotateMatrix(2 * np.pi / self.r * i))
                l = np.around(l)
                rotate[i] = l

        else:
            for i in range(self.r):
                l = np.dot(loc, rotateMatrix(2 * np.pi / self.r * i))
                l = np.around(l)
                rotate[i] = l
        m = 0
        for l in rotate:
            a = int(np.max(l))
            if a > m:
                m = a

        p = [[(int(l[j, 0] + m), int(m - l[j, 0]), int(l[j, 1] + m),
               int(m - l[j, 1])) for j in range(s1 * s2)] for l in rotate]
        return [p, m]

    def cal_weight(self, c, scale, seprate=False):
        def rotateChannel(x, r=self.r):
            # print('rotate')
            a = x.size(1)
            # print(x.size())
            x = torch.split(x, int(a / r), dim=1)
            y = torch.cat(x[0:-1], dim=1)
            y = torch.cat((x[-1], y), dim=1)
            return y

        wr = c.weight
        s1 = wr.size(2)
        s2 = wr.size(3)
        rotate = [0 for i in range(self.r)]
        if seprate:
            w = wr.permute(0, 2, 3, 1).contiguous()
            # print(w.size())
            w = w.view(w.size(0), -1, 1, 1)
            # print(w.size())
            w = [w for i in range(self.r)]
            # W = torch.cat(w, 0)
            W = w
            # print(W.size())
        else:
            for i in range(self.r):
                if i != 0:
                    wr = rotateChannel(wr)

                w = wr.permute(0, 2, 3, 1).contiguous()
                # print(w.size())
                # print(wr.size())
                w = w.view(w.size(0), -1, 1, 1)
                # print(w.size())
                rotate[i] = w
            # W = torch.cat(rotate,0)
            W = rotate
        return W

    def rotateConv_eff(self, x, a, w):
        o = [[F.pad(x, p) for p in pp] for pp in a[0]]
        o = [torch.cat(oo, dim=1) for oo in o]
        o = [F.conv2d(o[i], w[i]) for i in range(self.r)]
        o = torch.cat(o, dim=1)[:, :, a[1]:-a[1], a[1]:-a[1]]
        return o

    def forward(self, x):
        self.get_weight()
        out = self.rotateConv_eff(x, self.a, self.w)
        out = F.relu(self.BN(out, self.bn))
        return out


class MyLeNetRotateInvariantNew_nms(nn.Module):
    def __init__(self):
        super(MyLeNetRotateInvariantNew_nms, self).__init__()
        self.r = 8
        base = 16

        # self.conv1 = nn.Conv2d(1, int(base / self.r), 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(int(base / self.r), momentum=0.1 / self.r)
        self.rconv1 = Rotate_Conv(1,base,3,seperate=True)
        # self.conv1 = nn.Conv2d(1, int(base), 3, padding=1)
        # self.bn1 = nn.BatchNorm2d(int(base))

        # self.conv2 = nn.Conv2d(base, int(base / self.r), 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(int(base / self.r), momentum=0.1 / self.r)
        self.rconv2 = Rotate_Conv(base,base,3)
        # self.conv2 = nn.Conv2d(base, int(base), 3, padding=1)
        # self.bn2 = nn.BatchNorm2d(int(base))

        # self.conv3 = nn.Conv2d(base, int(base * 2 / self.r), 3, padding=1)
        # self.bn3 = nn.BatchNorm2d(int(base * 2 / self.r),
        #                           momentum=0.1 / self.r)
        # self.rconv3 = Rotate_Conv(base,base*2,3)
        self.conv3 = nn.Conv2d(base, int(base * 2), 3, padding=1)
        self.bn3 = nn.BatchNorm2d(int(base * 2))

        # self.conv4 = nn.Conv2d(base*2, int(base*2 / self.r), 3, padding=1)
        # self.bn4 = nn.BatchNorm2d(int(base * 2 / self.r))
        self.conv4 = nn.Conv2d(base*2, int(base * 2), 3, padding=1)
        self.bn4 = nn.BatchNorm2d(int(base * 2))

        # self.conv5 = nn.Conv2d(base * 2, int(base * 4 / self.r), 3, padding=1)
        # self.bn5 = nn.BatchNorm2d(int(base * 4 / self.r),
        #                           momentum=0.1 / self.r)
        self.conv5 = nn.Conv2d(base * 2, int(base * 4), 3, padding=1)
        self.bn5 = nn.BatchNorm2d(int(base * 4 ))

        self.conv6 = nn.Conv2d(base * 4,
                               10,
                               3,
                               padding=1)
        # self.get_rot_pad()
        # self.viz = Visdom(server='http://127.0.0.1', port=8097)
        # assert self.viz.check_connection()

    def get_rot_pad(self):
        self.a1 = self.cal_rot_pad(self.conv1, 1, seprate=True)
        self.a2 = self.cal_rot_pad(self.conv2, 1)
        self.a3 = self.cal_rot_pad(self.conv3, 1)
        # self.a4 = self.cal_rot_pad(self.conv4, 1)
        # self.a5 = self.cal_rot_pad(self.conv5, 1)
        # self.a6 = self.cal_rot_pad(self.conv6, 1)
        return 0

    def get_weight(self):
        self.w1 = self.cal_weight(self.conv1, 1, seprate=True)
        self.w2 = self.cal_weight(self.conv2, 1)
        self.w3 = self.cal_weight(self.conv3, 1)
        # self.w4 = self.cal_weight(self.conv4, 1)
        # self.w5 = self.cal_weight(self.conv5, 1)
        # self.w6 = self.cal_weight(self.conv6, 1)
        return 0

    def BN(self, x, bn):
        # print('BN')
        a = x.size(1)
        # print(a)
        x = torch.split(x, int(a / self.r), dim=1)
        x = [bn(x[i]) for i in range(self.r)]
        x = torch.cat(x, dim=1)
        # print(x.size())
        return x

    def rotateMax(self, x):
        # print('max')
        s0, s1, s2, s3 = x.size()
        # print(s1)
        x = x.view(s0, int(s1 / self.r), self.r, s2, s3)
        x = torch.max(x, dim=2, keepdim=False)[0]
        return x

    def rotatesplit(self, x, n):
        s0, s1, s2, s3 = x.size()
        x = x.view(s0, int(s1 / self.r), self.r, s2, s3)
        x = torch.split(x, n, dim=1)
        return [a.view(s0, -1, s2, s3) for a in x]

    def nms(self, x):
        s0, s1, s2, s3 = x.size()
        x = x.view(s0, int(s1 / self.r), self.r, s2, s3)
        y = torch.max(x, dim=2, keepdim=True)[0]
        y = torch.cat([y for i in range(self.r)], dim=2)
        y = y.ne(x).float()
        x = x * y
        return x.mean()

    def ds(self, x):
        s0, s1, s2, s3 = x.size()
        x = x.view(s0, int(s1 / self.r), self.r, s2, s3)
        y = torch.mean(x, dim=2, keepdim=True)
        y = torch.cat([y for i in range(self.r)], dim=2)
        x = (x - y).abs()
        # y = y.ne(x).float()
        # x = x*y
        return x.sum(dim=0).sum(dim=0).mean()

    def nms_ds(self, x, n):
        a = self.rotatesplit(x, n)
        return self.nms(a[0]) + self.ds(a[1])

    def cal_rot_pad(self,c,scale,seprate=False):

        def rotateMatrix(theta):
            return np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])

        wr = c.weight
        s1 = wr.size(2)
        s2 = wr.size(3)
        locx = np.array([[j - (s2 - 1) / 2 for j in range(s2)]
                         for i in range(s1)])
        locy = np.array([[i - (s1 - 1) / 2 for j in range(s2)]
                         for i in range(s1)])
        loc = np.concatenate((locx.reshape(-1, 1), locy.reshape(-1, 1)),
                             axis=1)
        loc = loc * scale
        # print('rotateConv')
        rotate = [0 for i in range(self.r)]
        if seprate:
            # w = wr.view(wr.size(0), -1, 1, 1)
            # w = [w for i in range(self.r)]
            # W = torch.cat(w, 0)
            for i in range(self.r):
                l = np.dot(loc, rotateMatrix(2 * np.pi / self.r * i))
                l = np.around(l)
                rotate[i] = l

        else:
            for i in range(self.r):
                # if i != 0:
                #     wr = rotateChannel(wr)

                # w = wr.view(wr.size(0), -1, 1, 1)
                # if i==0:
                #     W = w
                # else:
                #     W = torch.cat((W,w),0)
                l = np.dot(loc, rotateMatrix(2 * np.pi / self.r * i))
                l = np.around(l)
                rotate[i] = l
        m=0
        for l in rotate:
            a = int(np.max(l))
            if a>m:
                m=a

        # p = [(int(l[j, 0] + m), int(m - l[j, 0]), int(l[j, 1] + m),
        #        int(m - l[j, 1])) for j in range(s1 * s2) for l in rotate]
        p = [[(int(l[j, 0] + m), int(m - l[j, 0]), int(l[j, 1] + m),
                      int(m - l[j, 1])) for j in range(s1 * s2)] for l in rotate]
        return [p,m]

    def cal_weight(self, c, scale, seprate=False):
        def rotateChannel(x, r=self.r):
            # print('rotate')
            a = x.size(1)
            # print(x.size())
            x = torch.split(x, int(a / r), dim=1)
            y = torch.cat(x[0:-1], dim=1)
            y = torch.cat((x[-1], y), dim=1)
            return y

        wr = c.weight
        s1 = wr.size(2)
        s2 = wr.size(3)
        rotate = [0 for i in range(self.r)]
        if seprate:
            w = wr.permute(0, 2, 3, 1).contiguous()
            # print(w.size())
            w = w.view(w.size(0),-1,1,1)
            # print(w.size())
            w = [w for i in range(self.r)]
            # W = torch.cat(w, 0)
            W=w
            # print(W.size())
        else:
            for i in range(self.r):
                if i != 0:
                    wr = rotateChannel(wr)

                w = wr.permute(0, 2, 3, 1).contiguous()
                # print(w.size())
                # print(wr.size())
                w = w.view(w.size(0),-1,1,1)
                # print(w.size())
                rotate[i] = w
            # W = torch.cat(rotate,0)
            W = rotate
        return W

    def rotateConv_eff(self, x, a, w):
        # o = [F.pad(x,p) for p in a[0]]
        # o = torch.cat(o,dim=1)
        # o = F.conv2d(o,w,groups=self.r)[:,:,a[1]:-a[1],a[1]:-a[1]]
        o = [[F.pad(x,p) for p in pp] for pp in a[0]]
        o = [torch.cat(oo,dim=1) for oo in o]
        o = [F.conv2d(o[i],w[i]) for i in range(self.r)]
        o = torch.cat(o, dim=1)[:, :, a[1]:-a[1], a[1]:-a[1]]
        return o

    def rotateConv(self, x, c, scale, seprate=False):
        def rotateChannel(x, r=self.r):
            # print('rotate')
            a = x.size(1)
            # print(x.size())
            x = torch.split(x, int(a / r), dim=1)
            y = torch.cat(x[0:-1], dim=1)
            y = torch.cat((x[-1], y), dim=1)
            return y

        def rotateMatrix(theta):
            return np.array([[np.cos(theta), -np.sin(theta)],
                             [np.sin(theta), np.cos(theta)]])

        wr = c.weight
        s1 = wr.size(2)
        s2 = wr.size(3)
        locx = np.array([[j - (s2 - 1) / 2 for j in range(s2)]
                         for i in range(s1)])
        locy = np.array([[i - (s1 - 1) / 2 for j in range(s2)]
                         for i in range(s1)])
        loc = np.concatenate((locx.reshape(-1, 1), locy.reshape(-1, 1)),
                             axis=1)
        loc = loc * scale
        # print('rotateConv')
        rotate = [0 for i in range(self.r)]
        # print(wr.size())
        if seprate:
            w = torch.split(wr, 1, dim=2)
            w = [torch.split(w[i], 1, dim=3) for i in range(s1)]
            o = [F.conv2d(x, w[j][i]) for i in range(s2) for j in range(s1)]
            for i in range(self.r):
                l = np.dot(loc, rotateMatrix(2 * np.pi / self.r * i))
                l = np.around(l)
                m = int(np.max(l))
                p = [(int(l[j, 0] + m), int(m - l[j, 0]), int(l[j, 1] + m),
                      int(m - l[j, 1])) for j in range(s1 * s2)]
                o1 = [F.pad(o[j], p[j]).unsqueeze(0) for j in range(s1 * s2)]
                o1 = torch.cat(o1, 0)
                rotate[i] = torch.sum(o1, dim=0,
                                      keepdim=False)[:, :, m:-m, m:-m]

        else:
            for i in range(self.r):
                if i != 0:
                    wr = rotateChannel(wr)

                # w = torch.split(wr, 1, dim=2)
                # w = [torch.split(w[i], 1, dim=3) for i in range(s1)]
                w = wr.permute(0, 2, 3, 1).contiguous()
                w = w.view(w.size(0),-1,1,1)
                # if i==0:
                #     W = w
                # else:
                #     W = torch.cat((W,w),0)
                # o = [
                #     F.conv2d(x, w[j][i]) for i in range(s2) for j in range(s1)
                # ]
                o = [
                    x for i in range(s2) for j in range(s1)
                ]
                l = np.dot(loc, rotateMatrix(2 * np.pi / self.r * i))
                l = np.around(l)
                m = int(np.max(l))
                p = [(int(l[j, 0] + m), int(m - l[j, 0]), int(l[j, 1] + m),
                      int(m - l[j, 1])) for j in range(s1 * s2)]
                # o1 = [F.pad(o[j], p[j]).unsqueeze(0) for j in range(s1 * s2)]
                o1 = [F.pad(o[j], p[j]) for j in range(s1 * s2)]
                o1 = torch.cat(o1, 1)[:, :, m:-m, m:-m]
                # if i==0:
                #     O1 = o1
                # else:
                #     O1 = torch.cat((O1,o1),1)
                # o1 = F.conv2d(o1,w)
                rotate[i]= F.conv2d(o1,w)

        rotate = torch.cat(rotate, dim=1)

        # print(rotate.size())
        # rotate = rotate[:, :, 2:-2, 2:-2]
        # print(rotate.size())

        return rotate

    def forward(self, x):
        # b = x.cpu().numpy()
        # bmin = np.min(b)
        # bmax = np.max(b)
        # for i in range(1):
        #     b[i] = (b[i] - np.min(b[i])) / (np.max(b[i]) - np.min(b[i]))
        #     self.viz.image(b[i])
        # out = self.rotateConv(x, self.conv1, 1, seprate=True)
        # self.get_weight()
        # out = self.rotateConv_eff(x, self.a1, self.w1)
        # out = F.relu(self.BN(out, self.bn1))
        out = self.rconv1(x)
        # n = self.nms(out)
        # b = out[0].cpu().numpy()
        # print(len(b))
        # print(len(b[0]))
        # c = int(out.size(1) / self.r)
        # print(c)
        # for i in range(c):
        #     for j in range(self.r):
        #         b[i + j * c] = b[i + j * c] / np.max(b[i + j * c])
        #         self.viz.image(b[i + j * c])

        # print(self.conv1.weight.size())
        # print(self.conv1.weight)
        # b = self.conv1.weight.cpu().numpy()
        # bmin = np.min(b)
        # bmax = np.max(b)
        # for i in range(self.conv1.weight.size(0)):
        #     print(b[i])
        #     b[i] = (b[i] - bmin) / (bmax - bmin)
        #     self.viz.image(b[i])
        # time.sleep(20)

        # out = self.conv1(x)
        # out = F.relu(self.bn1(out))

        # out = self.conv2(out)
        # out = F.relu(self.bn2(out))
        # out = self.rotateConv(out, self.conv2, 1)
        # out = self.rotateConv_eff(out, self.a2, self.w2)
        # out = F.relu(self.BN(out, self.bn2))
        out = self.rconv2(out)
        n = self.nms(out)

        # n = n + self.nms(out)

        out = F.max_pool2d(out, 2)
        out = self.conv3(out)
        out = F.relu(self.bn3(out))
        # out = self.rotateConv(out, self.conv3, 1)
        # out = self.rotateConv_eff(out, self.a3, self.w3)
        # out = F.relu(self.BN(out, self.bn3))
        # out = self.rconv3(out)
        # n = n + self.nms(out)
        out = self.conv4(out)
        out = F.relu(self.bn4(out))
        # out = self.rotateConv(out, self.conv4, 1)
        # out = self.rotateConv_eff(out, self.a4, self.w4)
        # out = F.relu(self.BN(out, self.bn4))
        # n = n + self.nms(out)
        out = F.max_pool2d(out, 2)
        out = self.conv5(out)
        out = F.relu(self.bn5(out))
        # out = self.rotateConv_eff(out, self.a5, self.w5)
        # out = F.relu(self.BN(out, self.bn5))
        out = self.conv6(out)
        # out = self.rotateConv_eff(out, self.a6, self.w6)
        # out = self.rotateMax(out)
        out = F.max_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        return out, n
