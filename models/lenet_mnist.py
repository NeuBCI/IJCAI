'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F
import torch as torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
from visdom import Visdom
import time

rot = 8

class Rotate_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, ks, s=1, seperate=False,ifbn=True):
        super(Rotate_Conv, self).__init__()
        self.r = rot
        self.s = s
        self.ifbn=ifbn
        self.seperate = seperate
        self.conv = nn.Conv2d(in_channel, int(out_channel / self.r), ks)
        if self.ifbn:
            self.bn = nn.BatchNorm2d(int(out_channel / self.r),momentum=0.1/self.r)
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

    def rotateConv_eff(self, x, a, w, s=1):
        o = [[F.pad(x, p) for p in pp] for pp in a[0]]
        o = [torch.cat(oo, dim=1) for oo in o]
        o = [F.conv2d(o[i], w[i], stride=s) for i in range(self.r)]
        o = torch.cat(o, dim=1)[:, :, a[1]:-a[1], a[1]:-a[1]]
        return o

    def forward(self, x):
        self.get_weight()
        out = self.rotateConv_eff(x, self.a, self.w, s=self.s)
        if self.ifbn:
            out = self.BN(out, self.bn)
        return out

class Rotate_Conv_noshare(nn.Module):
    def __init__(self, in_channel, out_channel, ks, s=1, seperate=False,ifbn=True):
        super(Rotate_Conv_noshare, self).__init__()
        self.r = rot
        self.s = s
        self.ifbn=ifbn
        self.seperate = seperate
        self.conv = nn.Conv2d(int(in_channel / self.r), int(out_channel / self.r), ks)
        if self.ifbn:
            self.bn = nn.BatchNorm2d(int(out_channel / self.r),momentum=0.1/self.r)
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
        seprate = True
        if self.r==4:
            for i in range(self.r):
                if not seprate and i!=0:
                    wr = rotateChannel(wr)
                wr=wr[:,:,range(wr.size(2))[::-1],:]
                wr = wr.permute(0, 1, 3, 2).contiguous()
                rotate[i] = wr
            W = rotate
        elif seprate:
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

    def rotateConv_eff(self, x, a, w, s=1):
        if self.r==4:
            o = [F.conv2d(x, w[i], padding=int((self.ks-1)/2), stride=s) for i in range(self.r)]
            o = torch.cat(o, dim=1)
        else:
            b = x.size(1)
            x = torch.split(x, int(b / self.r), dim=1)
            o = [[F.pad(x[i], p) for p in pp] for i,pp in enumerate(a[0])]
            # o = [[F.pad(x1, p) for p in pp] for x1,pp in x,a[0]]
            o = [torch.cat(oo, dim=1) for oo in o]
            o = [F.conv2d(o[i], w[i], stride=s) for i in range(self.r)]
            o = torch.cat(o, dim=1)[:, :, a[1]:-a[1], a[1]:-a[1]]
        return o

    def forward(self, x):
        self.get_weight()
        out = self.rotateConv_eff(x, self.a, self.w, s=self.s)
        if self.ifbn:
            out = self.BN(out, self.bn)
        return out

class Repeat_Conv(nn.Module):
    def __init__(self, in_planes,planes, r):
        super(Repeat_Conv, self).__init__()
        self.r = r
        self.conv = nn.Conv2d(in_planes, int(planes/self.r), kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(int(planes/self.r))

    def forward(self, x):
        out = self.bn(self.conv(x))
        out = [out for i in range(self.r)]
        out = torch.cat(out, dim=1)
        return out

class Norm_Conv(nn.Module):
    def __init__(self, in_planes, planes):
        super(Norm_Conv, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        return self.bn(self.conv(x))

class MyLeNetRotateInvariantNew_nms(nn.Module):
    def __init__(self):
        super(MyLeNetRotateInvariantNew_nms, self).__init__()
        self.r = rot
        base = 16

        ## tiny network 1
        # self.rconv1 = Rotate_Conv(1,base,3,seperate=True)
        # self.rconv2 = Rotate_Conv(base,base,3)
        # # self.rconv3 = Rotate_Conv(base,base,3)

        # # self.rconv1 = Norm_Conv(1,base)
        # # self.rconv2 = Norm_Conv(base,base)
        # self.rconv3 = Norm_Conv(base,base)

        # self.conv4 = nn.Conv2d(base,10,3)

        ##tiny network 2 
        # base = 8
        # self.rconv1 = Rotate_Conv(1,base,3,seperate=True)
        # self.rconv2 = Rotate_Conv(base,base*2,3)
        # self.rconv3 = Rotate_Conv(base*2,base*4,3)

        # # self.rconv1 = Norm_Conv(1,base)
        # # self.rconv2 = Norm_Conv(base,base*2)
        # # self.rconv3 = Norm_Conv(base*2,base*4)

        # self.conv4 = nn.Conv2d(base*4,10,3)

        ## ordinary network
        # # self.dp = nn.Dropout(p=0.1)
        self.rconv1 = Rotate_Conv(1,base,3,seperate=True)
        self.rconv2 = Rotate_Conv(base,base,3)
        self.rconv3 = Rotate_Conv(base,base*2,3)
        # self.rconv4 = Rotate_Conv(base*2,base*2,3)
        # self.rconv5 = Rotate_Conv(base*2,base*4,3)
        # self.rconv6 = Rotate_Conv(base*4,base*4,3)

        # self.rconv1 = Norm_Conv(1,base)
        # self.rconv2 = Norm_Conv(base,base)
        # self.rconv3 = Norm_Conv(base,base*2)
        self.rconv4 = Norm_Conv(base*2,base*2)
        self.rconv5 = Norm_Conv(base*2,base*4)
        self.rconv6 = Norm_Conv(base*4,base*4)

        self.conv7 = nn.Conv2d(base*4, 10, 4, padding=2)


        # self.conv6 = nn.Conv2d(20,
        #                        10,
        #                        3,
        #                        padding=1)
        # self.get_rot_pad()
        self.viz = Visdom(server='http://127.0.0.1', port=8097)
        assert self.viz.check_connection()


    def rotateMax(self, x):
        # print('max')
        s0, s1, s2, s3 = x.size()
        # print(s1)
        x = x.view(s0, int(s1 / self.r), self.r, s2, s3)
        x = torch.max(x, dim=2, keepdim=False)[0]
        # x = torch.mean(x, dim=2, keepdim=False)
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

    def forward(self, x):
        # b = x.cpu().numpy()
        # bmin = np.min(b)
        # bmax = np.max(b)
        # for i in range(1):
        #     b[i] = (b[i] - np.min(b[i])) / (np.max(b[i]) - np.min(b[i]))
        #     self.viz.image(b[i])
        
        ## tiny networks
        # out = self.rconv1(x)
        # out = F.relu(out)
        # out = F.max_pool2d(out, 2)
        # out = self.rconv2(out)
        # out = F.relu(out)
        # out = F.max_pool2d(out, 2)
        # out = self.rconv3(out)
        # out = F.relu(out)
        # n = self.nms(out)

        # out = self.conv4(out)

        ## ordinary networks
        out = self.rconv1(x)
        out = F.relu(out)
        out = self.rconv2(out)
        out = F.relu(out)
        # b = out[0].cpu().numpy()
        # print(len(b))
        # print(len(b[0]))
        # c = int(out.size(1) / self.r)
        # print(c)
        # for i in range(c):
        #     d = np.zeros((out.size(2),out.size(3)))
        #     for j in range(self.r):
        #         b[i + j * c] = b[i + j * c] / np.max(b[i + j * c])
        #         self.viz.image(b[i + j * c])
        #         d = d + b[i+j*c]
        #         if j==self.r-1:
        #             d = d / np.max(d)
        #             self.viz.image(d)
        #             d[:,:] = 0


        # print(self.rconv1.conv.weight.size())
        # print(self.rconv1.conv.weight)
        # b = self.rconv1.conv.weight.cpu().numpy()
        # bmin = np.min(b)
        # bmax = np.max(b)
        # for i in range(self.rconv1.conv.weight.size(0)):
        #     print(b[i])
        #     b[i] = (b[i] - bmin) / (bmax - bmin)
        #     self.viz.image(b[i])
        # time.sleep(20)
        out = F.max_pool2d(out, 2)
        out = self.rconv3(out)
        out = F.relu(out)
        n = self.nms(out)
        out = self.rconv4(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2)
        out = self.rconv5(out)
        out = F.relu(out)
        out = self.rconv6(out)
        out = F.relu(out)
        out = F.max_pool2d(out, 2, padding=(1,1))
        # out = self.dp(out)
        out = self.conv7(out)
        # out = self.rotateMax(out)

        # out = self.conv1(x)
        # # print(x.size())
        # out = F.relu(self.bn1(out))
        # # out = self.dp(out)
        # out = self.conv2(out)
        # out = F.relu(self.bn2(out))
        # # out = self.dp(out)
        # out = F.max_pool2d(out, 2)
        # out = self.conv3(out)
        # out = F.relu(self.bn3(out))
        # # out = self.dp(out)
        # out = self.conv4(out)
        # out = F.relu(self.bn4(out))
        # # out = self.dp(out)
        # out = F.max_pool2d(out, 2)
        # out = self.conv5(out)
        # out = F.relu(self.bn5(out))
        # # out = self.dp(out)
        # out = self.conv6(out)
        # out = F.relu(self.bn6(out))
        # # out = self.dp(out)
        # out = F.max_pool2d(out, 2, padding=(1,1))
        # out = self.dp(out)
        # out = self.conv7(out)
        # n=out.mean()

        out = F.max_pool2d(out, out.size(2))
        out = out.view(out.size(0), -1)
        return out, n
