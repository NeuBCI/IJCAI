'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from visdom import Visdom

rot = 4

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


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
            o = [[F.pad(x, p) for p in pp] for pp in a[0]]
            o = [torch.cat(oo, dim=1) for oo in o]
            o = [F.conv2d(o[i], w[i], stride=s) for i in range(self.r)]
            o = torch.cat(o, dim=1)[:, :, a[1]:-a[1], a[1]:-a[1]]
        return o

    def forward(self, x):
        self.get_weight()
        out = self.rotateConv_eff(x, self.a, self.w, s=self.stride)
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
    def __init__(self, in_planes, planes, ks, stride=1):
        super(Norm_Conv, self).__init__()
        self.conv = nn.Conv2d(in_planes, planes, kernel_size=ks, stride=stride, padding=int((ks-1)/2), bias=False)
        self.bn = nn.BatchNorm2d(planes)

    def forward(self, x):
        return self.bn(self.conv(x))

class BasicBlock_Rotate(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock_Rotate, self).__init__()
        self.conv1 = Rotate_Conv(in_planes, planes, 3)
        self.conv2 = Rotate_Conv(planes, planes, 3)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = Rotate_Conv(in_planes, self.expansion * planes, 1, s=stride)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.conv2(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 16
        self.r = rot

        # self.conv1 = Norm_Conv(3,16,3)
        # self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        # self.conv1 = Repeat_Conv(3,16,self.r)
        self.conv1 = Rotate_Conv(3,16,3,s=1,seperate=True)
        self.layer1 = self._make_layer(BasicBlock_Rotate, 16, num_blocks[0], stride=1)
        self.conv2 = Norm_Conv(16,16,1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
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
        out = F.relu(self.conv1(x))
        # n = out.mean()
        # n = self.nms(out)
        out = self.layer1(out)
        n = self.nms(out)
        out = F.relu(self.conv2(out))
        # out = self.layer11(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, n

class ResNet_img(nn.Module):
    def __init__(self, block, num_blocks, num_classes=100):
        super(ResNet_img, self).__init__()
        self.in_planes = 64
        self.r = rot

        # self.conv1 = Norm_Conv(3,64,7,stride=2)
        # self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        # self.conv1 = Repeat_Conv(3,16,self.r)
        self.conv1 = Rotate_Conv(3,64,7,s=1,stride=2,seperate=True)
        self.layer1 = self._make_layer(BasicBlock_Rotate, 64, num_blocks[0], stride=1)
        self.conv2 = Norm_Conv(64,64,1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
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
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        # n = out.mean()
        # n = self.nms(out)
        out = self.layer1(out)
        n = self.nms(out)
        out = F.relu(self.conv2(out))
        # out = self.layer11(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 7)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out, n


def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])


def test():
    net = ResNet18()
    y = net(torch.randn(1,3,32,32))
    print(y.size())

# test()
if __name__ == "__main__":
    test()
