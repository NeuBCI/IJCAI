'''DenseNet in PyTorch.'''
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
rot = 4
ifinterpolation = True

class Rotate_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, ks, s=1, seperate=False, ifbn=True, stride=1):
        super(Rotate_Conv, self).__init__()
        self.r = rot
        self.s = s
        self.ifbn = ifbn
        self.seperate = seperate
        self.stride = stride
        self.ks = ks
        self.conv = nn.Conv2d(in_channel, int(out_channel / self.r), ks)
        self.bias = self.conv.bias
        if self.ifbn:
            self.bn = nn.BatchNorm2d(int(in_channel / self.r),momentum=0.1/self.r)
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

    def cal_rot_pad_(self, c, scale, seprate=False):
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
        interpolation = [0 for i in range(self.r)]
        if seprate:
            for i in range(self.r):
                l = np.dot(loc, rotateMatrix(2 * np.pi / self.r * i))
                l_f = np.floor(l)
                rotate[i] = l_f
                a = l - l_f
                b = 1. - a
                interpolation[i] = [np.array([[a[j,0]*a[j,1],b[j,0]*a[j,1]],[a[j,0]*b[j,1],b[j,0]*b[j,1]]]) for j in range(s1*s2)]

        else:
            for i in range(self.r):
                l = np.dot(loc, rotateMatrix(2 * np.pi / self.r * i))
                l_f = np.floor(l)
                rotate[i] = l_f
                a = l - l_f
                b = 1. - a
                interpolation[i] = [np.array([[a[j,0]*a[j,1],b[j,0]*a[j,1]],[a[j,0]*b[j,1],b[j,0]*b[j,1]]]) for j in range(s1*s2)]
        m = 0
        for l in rotate:
            a = int(np.max(l))
            if a > m:
                m = a

        p = [[((int(l[j, 0] + m), int(m - l[j, 0]), int(l[j, 1] + m),
               int(m - l[j, 1])),interpolation[i,j]) for j in range(s1 * s2)] for i,l in enumerate(rotate)]
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
            a[1] = int(a[1]/s)
            o = [[F.pad(x, p) for p, interp in pp] for pp in a[0]]
            o = [torch.cat(oo, dim=1) for oo in o]
            o = [F.conv2d(o[i], w[i], stride=s) for i in range(self.r)]
            o = torch.cat(o, dim=1)[:, :, a[1]:-a[1], a[1]:-a[1]]
        return o

    def rotateConv_eff_(self, x, a, w, s=1):
        def interpo(x,interp):
            s0,s1,s2,s3=x.size()
            interp = torch.from_numpy(interp.reshape(1,1,2,2)).type(torch.FloatTensor).cuda()
            x = x.view(-1,1,s2,s3)
            return F.conv2d(x,interp, padding=(1,0,1,0)).view(s0,s1,s2,s3)
        if self.r==4:
            o = [F.conv2d(x, w[i], padding=int((self.ks-1)/2), stride=s) for i in range(self.r)]
            o = torch.cat(o, dim=1)
        else:
            a[1] = int(a[1]/s)
            o = [[F.pad(interpo(x, interp), p) for p, interp in pp] for pp in a[0]]
            o = [torch.cat(oo, dim=1) for oo in o]
            o = [F.conv2d(o[i], w[i], stride=s) for i in range(self.r)]
            o = torch.cat(o, dim=1)[:, :, a[1]:-a[1], a[1]:-a[1]]
        return o

    def forward(self, x):
        self.get_weight()
        if self.ifbn:
            x = F.relu(self.BN(x, self.bn))
        out = self.rotateConv_eff(x, self.a, self.w, s=self.stride)
        return out

class Rotate_Conv_noshare(nn.Module):
    def __init__(self, in_channel, out_channel, ks, s=1, seperate=False,ifbn=True):
        super(Rotate_Conv_noshare, self).__init__()
        self.r = rot
        self.s = s
        self.ifbn=ifbn
        self.seperate = seperate
        self.conv = nn.Conv2d(int(in_channel / self.r), int(out_channel / self.r), ks)
        self.bias = self.conv.bias
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

    def cal_rot_pad_(self, c, scale, seprate=False):
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
        interpolation = [0 for i in range(self.r)]
        if seprate:
            for i in range(self.r):
                l = np.dot(loc, rotateMatrix(2 * np.pi / self.r * i))
                l_f = np.floor(l)
                rotate[i] = l_f
                a = l - l_f
                b = 1. - a
                interpolation[i] = [np.array([[a[j,0]*a[j,1],b[j,0]*a[j,1]],[a[j,0]*b[j,1],b[j,0]*b[j,1]]]) for j in range(s1*s2)]

        else:
            for i in range(self.r):
                l = np.dot(loc, rotateMatrix(2 * np.pi / self.r * i))
                l_f = np.floor(l)
                rotate[i] = l_f
                a = l - l_f
                b = 1. - a
                interpolation[i] = [np.array([[a[j,0]*a[j,1],b[j,0]*a[j,1]],[a[j,0]*b[j,1],b[j,0]*b[j,1]]]) for j in range(s1*s2)]
        m = 0
        for l in rotate:
            a = int(np.max(l))
            if a > m:
                m = a

        p = [[(int(l[j, 0] + m), int(m - l[j, 0]), int(l[j, 1] + m),
               int(m - l[j, 1])) for j in range(s1 * s2)] for l in rotate]
        return [p, m, interpolation]

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

class Bottleneck(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, growth_rate*4, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(growth_rate*4)
        self.conv2 = nn.Conv2d(growth_rate*4, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = self.conv2(F.relu(self.bn2(out)))
        out = torch.cat([out,x], 1)
        return out


class Bottleneck_rot(nn.Module):
    def __init__(self, in_planes, growth_rate):
        super(Bottleneck_rot, self).__init__()
        self.r = rot
        self.conv1 = Rotate_Conv(in_planes, growth_rate*4, ks=1)
        self.conv2 = Rotate_Conv(growth_rate*4, growth_rate, ks=3)

    def cat_rot(self, x, y):
        x = x.view(x.size(0), int(x.size(1)/self.r), 4, x.size(2), x.size(3))
        y = y.view(y.size(0), int(y.size(1)/self.r), 4, y.size(2), y.size(3))
        x = torch.cat([x, y], dim=1)
        return x.view(x.size(0), -1, x.size(3), x.size(4))

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.cat_rot(out, x)
        return out


class Transition(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_planes)
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        out = self.conv(F.relu(self.bn(x)))
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet, self).__init__()
        self.growth_rate = growth_rate
        self.r = rot

        num_planes = 2*growth_rate
        self.conv1 = nn.Conv2d(3, num_planes, kernel_size=3, padding=1, bias=False)

        self.dense1 = self._make_dense_layers(block, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        # out_planes = int(math.floor(num_planes*reduction))
        # self.trans3 = Transition(num_planes, out_planes)
        # num_planes = out_planes

        # self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        # num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    
    def nms(self, x):
        s0, s1, s2, s3 = x.size()
        x = x.view(s0, int(s1 / self.r), self.r, s2, s3)
        y = torch.max(x, dim=2, keepdim=True)[0]
        y = torch.cat([y for i in range(self.r)], dim=2)
        y = y.ne(x).float()
        x = x * y
        return x.mean()

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        n = self.nms(out)
        out = self.trans2(self.dense2(out))
        # out = self.trans3(self.dense3(out))
        out = self.dense3(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, n

class DenseNet_rot(nn.Module):
    def __init__(self, block, nblocks, growth_rate=12, reduction=0.5, num_classes=10):
        super(DenseNet_rot, self).__init__()
        self.growth_rate = growth_rate
        self.r = rot

        num_planes = 2*growth_rate
        self.conv1 = Rotate_Conv(3, num_planes, ks=3, seperate=True, ifbn=False)

        self.dense1 = self._make_dense_layers(Bottleneck_rot, num_planes, nblocks[0])
        num_planes += nblocks[0]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans1 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense2 = self._make_dense_layers(block, num_planes, nblocks[1])
        num_planes += nblocks[1]*growth_rate
        out_planes = int(math.floor(num_planes*reduction))
        self.trans2 = Transition(num_planes, out_planes)
        num_planes = out_planes

        self.dense3 = self._make_dense_layers(block, num_planes, nblocks[2])
        num_planes += nblocks[2]*growth_rate
        # out_planes = int(math.floor(num_planes*reduction))
        # self.trans3 = Transition(num_planes, out_planes)
        # num_planes = out_planes

        # self.dense4 = self._make_dense_layers(block, num_planes, nblocks[3])
        # num_planes += nblocks[3]*growth_rate

        self.bn = nn.BatchNorm2d(num_planes)
        self.linear = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_planes, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_planes, self.growth_rate))
            in_planes += self.growth_rate
        return nn.Sequential(*layers)

    def nms(self, x):
        s0, s1, s2, s3 = x.size()
        x = x.view(s0, int(s1 / self.r), self.r, s2, s3)
        y = torch.max(x, dim=2, keepdim=True)[0]
        y = torch.cat([y for i in range(self.r)], dim=2)
        y = y.ne(x).float()
        x = x * y
        return x.mean()

    def forward(self, x):
        out = self.conv1(x)
        out = self.dense1(out)
        n = self.nms(out)
        out = self.trans1(out)
        out = self.trans2(self.dense2(out))
        # out = self.trans3(self.dense3(out))
        out = self.dense3(out)
        out = F.avg_pool2d(F.relu(self.bn(out)), 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, n

def DenseNet121():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=32)

def DenseNet169():
    return DenseNet(Bottleneck, [6,12,32,32], growth_rate=32)

def DenseNet201():
    return DenseNet(Bottleneck, [6,12,48,32], growth_rate=32)

def DenseNet161():
    return DenseNet(Bottleneck, [6,12,36,24], growth_rate=48)

def densenet_cifar():
    return DenseNet(Bottleneck, [6,12,24,16], growth_rate=12)

def test():
    net = densenet_cifar()
    x = torch.randn(1,3,32,32)
    y = net(x)
    print(y)

# test()
