'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# from visdom import Visdom


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
    'VGGslim': [32, 64, 'M', 128, 128, 'M', 128, 256, 64, 32, 'M', 32, 32, 32, 32, 'M', 32, 32, 32, 32, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(512, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)


class myVGG(nn.Module):
    def __init__(self,vgg_name,num_classes=10):
        super(myVGG,self).__init__()
        self.n_rotate = 5
        self.r = 4
        self.features, self.features1 = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(cfg[vgg_name][-2], num_classes)

    def forward(self,x):
        out = self.features(x)
        n = self.nms(out)
        out = self.features1(out)
        out = out.view(out.size(0),-1)
        out = self.classifier(out)
        return out, n

    def _make_layers(self,cfg):
        layers = []
        layers1 = []
        cnt = 0
        in_channels = 3
        for x in cfg:
            if x == 'M':
                if cnt < self.n_rotate:
                    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                else:
                    layers1 += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if cnt < self.n_rotate:
                    layers += [Rotate_Conv(in_channels,x,3,seperate=True if cnt==0 else False),
                                nn.ReLU(inplace=True)]
                    # layers += [Norm_Conv(in_channels,x),
                    #             nn.ReLU(inplace=True)]
                    # layers += [Repeat_Conv(in_channels,x,self.r),
                    #             nn.ReLU(inplace=True)]
                else:
                    layers1 += [Norm_Conv(in_channels,x),
                                nn.ReLU(inplace=True)]
                cnt = cnt + 1
                in_channels = x
        
        return nn.Sequential(*layers), nn.Sequential(*layers1)

    
    def nms(self, x):
        s0, s1, s2, s3 = x.size()
        x = x.view(s0, int(s1 / self.r), self.r, s2, s3)
        y = torch.max(x, dim=2, keepdim=True)[0]
        y = torch.cat([y for i in range(self.r)], dim=2)
        y = y.ne(x).float()
        x = x * y
        return x.mean()


class Rotate_Conv(nn.Module):
    def __init__(self, in_channel, out_channel, ks, s=1, seperate=False, ifbn=True):
        super(Rotate_Conv, self).__init__()
        self.r = 4
        self.s = s
        self.ifbn = ifbn
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


def test():
    net = VGG('VGG11')
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
