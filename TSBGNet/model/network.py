from .base_function import *
from .external_function import SpectralNorm
import torch.nn.functional as F
import torch
from torchvision import models
import functools
import os
import torchvision.transforms as transforms


##############################################################################################################
# Network function
##############################################################################################################
# def define_structure(init_type='normal', gpu_ids=[]):
#
#     net = structure()
#
#     return init_net(net, init_type, gpu_ids)

def define_inpainting(init_type='normal', gpu_ids=[]):

    net = inpainting()

    return init_net(net, init_type, gpu_ids)

def de(init_type='orthogonal', gpu_ids=[]):

    net = deconder()

    return init_net(net, init_type, gpu_ids)

def rh(init_type='orthogonal', gpu_ids=[]):

    net = ronghe()

    return init_net(net, init_type, gpu_ids)


def define_Discriminator_2(init_type='orthogonal', gpu_ids=[]):

    net = Discriminator_2()

    return init_net(net, init_type, gpu_ids)



#############################################################################################################
# Network structure
#############################################################################################################
class ConvUp(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.gate_nonlinearity = nn.Sigmoid()
        self.nonlinearity = nn.GELU()
        self.pool_fix = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        kwargs_gate = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}

        self.conv = SpectralNorm(nn.Conv2d(512, 256, **kwargs))
        self.conv17 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.shortcut11 = SpectralNorm(nn.Conv2d(512, 128, **kwargs_short))
        self.model11 = nn.Sequential(self.conv, self.norm(256),self.nonlinearity, self.conv17,
                                     self.norm(128), self.nonlinearity)
        self.gateconv11 = SpectralNorm(nn.Conv2d(512, 128, **kwargs))
        self.gate11 = nn.Sequential(self.gateconv11, self.norm(128), self.gate_nonlinearity)

    def forward(self, input, size):
        out = F.interpolate(input=input, size=size, mode='bilinear')
        out = self.nonlinearity(self.model11(out) + self.shortcut11(out)) * self.gate11(out)
        return out

class res(nn.Module):
    def __init__(self):
        super(res, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_a = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1}
        kwargs_b = {'kernel_size': 3, 'stride': 1, 'padding': 2, 'dilation': 2}
        kwargs_c = {'kernel_size': 3, 'stride': 1, 'padding': 4, 'dilation': 4}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        kwargs_gate = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_out = {'kernel_size': 3, 'padding': 0, 'bias': True}
        # self.nonlinearity = nn.LeakyReLU(0.1)
        self.nonlinearity = nn.GELU()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
        self.pool_fix = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.gate_nonlinearity = nn.Sigmoid()
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)

        # decoder3 3*3
        self.conv22 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv23 = SpectralNorm(nn.Conv2d(64, 64, **kwargs_a))
        self.shortcut14 = SpectralNorm(nn.Conv2d(64, 64, **kwargs_a))
        self.model14 = nn.Sequential(self.norm(64), self.nonlinearity, self.conv22, self.norm(64), self.nonlinearity,
                                     self.conv23)
        self.gateconv14 = SpectralNorm(nn.Conv2d(64, 64, **kwargs_a))
        self.gate14 = nn.Sequential(self.gateconv14, self.norm(64), self.gate_nonlinearity)

        # out1
        self.conv_out1 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv_out11 = SpectralNorm(nn.Conv2d(64, 3, **kwargs))
        self.model_out1 = nn.Sequential(self.nonlinearity, self.conv_out1, self.norm(64),
                                        self.nonlinearity, self.conv_out11, nn.Tanh())

        # decoder3 5*5
        self.conv24 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv25 = SpectralNorm(nn.Conv2d(64, 64, **kwargs_b))
        self.shortcut15 = SpectralNorm(nn.Conv2d(64, 64, **kwargs_b))
        self.model15 = nn.Sequential(self.norm(64), self.nonlinearity, self.conv24, self.norm(64), self.nonlinearity,
                                     self.conv25)
        self.gateconv15 = SpectralNorm(nn.Conv2d(64, 64, **kwargs_b))
        self.gate15 = nn.Sequential(self.gateconv15, self.norm(64), self.gate_nonlinearity)

        # out2
        self.conv_out2 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv_out21 = SpectralNorm(nn.Conv2d(64, 3, **kwargs))
        self.model_out2 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out2, self.norm(64),
                                        self.nonlinearity, self.conv_out21, nn.Tanh())

        # decoder3 7*7
        self.conv26 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv27 = SpectralNorm(nn.Conv2d(64, 64, **kwargs_c))
        self.shortcut16 = SpectralNorm(nn.Conv2d(64, 64, **kwargs_c))
        self.model16 = nn.Sequential(self.norm(64), self.nonlinearity, self.conv26, self.norm(64), self.nonlinearity,
                                     self.conv27)
        self.gateconv16 = SpectralNorm(nn.Conv2d(64, 64, **kwargs_c))
        self.gate16 = nn.Sequential(self.gateconv16, self.norm(64), self.gate_nonlinearity)

        # out3
        self.conv_out3 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv_out31 = SpectralNorm(nn.Conv2d(64, 3, **kwargs))
        self.model_out3 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out3, self.norm(64),
                                        self.nonlinearity, self.conv_out31, nn.Tanh())
    def forward(self, x):

        a = self.nonlinearity(self.model14(x) + self.shortcut14(x)) * self.gate14(x)
        out = self.model_out1(a)

        #result.append(out)

        b = self.nonlinearity(self.model15(x) + self.shortcut15(x)) * self.gate15(x)
        # out = self.model_out2(b)
        #
        # result.append(out)

        c = self.nonlinearity(self.model16(x) + self.shortcut16(x)) * self.gate16(x)
        # out = self.model_out3(c)
        #
        # result.append(out)
        return a, b, c, out
# 特征融合
class ronghe(nn.Module):
    def __init__(self):
        super(ronghe, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_a = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        kwargs_b = {'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 1}
        kwargs_c = {'kernel_size': 7, 'stride': 2, 'padding': 3, 'output_padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        kwargs_gate = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_out = {'kernel_size': 3, 'padding': 0, 'bias': True}
        # self.nonlinearity = nn.LeakyReLU(0.1)
        self.nonlinearity = nn.GELU()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
        self.pool_fix = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.gate_nonlinearity = nn.Sigmoid()
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)

# 特征融合
        self.down_128 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_gate))
        self.conv13 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.shortcut7 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_short))
        self.model7 = nn.Sequential(self.nonlinearity, self.down_128, self.norm(128), self.nonlinearity)
        self.gateconv7 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_gate))
        self.gate7 = nn.Sequential(self.gateconv7, self.norm(128), self.gate_nonlinearity)
        # 2
        self.down_129 = SpectralNorm(nn.Conv2d(128, 256, **kwargs_gate))
        self.conv14 = SpectralNorm(nn.Conv2d(256, 256, **kwargs))
        self.shortcut8 = SpectralNorm(nn.Conv2d(128, 256, **kwargs_short))
        self.model8 = nn.Sequential(self.nonlinearity, self.down_129, self.norm(256), self.nonlinearity)
        self.gateconv8 = SpectralNorm(nn.Conv2d(128, 256, **kwargs_gate))
        self.gate8 = nn.Sequential(self.gateconv8, self.norm(256), self.gate_nonlinearity)

        # 第二层
        self.down_64 = SpectralNorm(nn.Conv2d(128, 256,**kwargs_gate))
        self.conv15 = SpectralNorm(nn.Conv2d(256, 256, **kwargs))
        self.shortcut9 = SpectralNorm(nn.Conv2d(128, 256, **kwargs_short))
        self.model9 = nn.Sequential(self.nonlinearity, self.down_64, self.norm(256), self.nonlinearity)
        self.gateconv9 = SpectralNorm(nn.Conv2d(128, 256, **kwargs_gate))
        self.gate9 = nn.Sequential(self.gateconv9, self.norm(256), self.gate_nonlinearity)

        # 第三层
        self.down_32 = SpectralNorm(nn.Conv2d(256, 256, 1, 1))
        self.conv16 = SpectralNorm(nn.Conv2d(256, 256, **kwargs))
        self.shortcut10 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_short))
        self.model10 = nn.Sequential(self.nonlinearity, self.down_32, self.norm(256), self.nonlinearity)
        self.gateconv10 = SpectralNorm(nn.Conv2d(256, 256, **kwargs))
        self.gate10 = nn.Sequential(self.gateconv10, self.norm(256), self.gate_nonlinearity)

        # 第四层
        self.up = ConvUp()

        # 融合
        self.down = SpectralNorm(nn.Conv2d(768, 256, 1, 1))
        self.conv17 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.shortcut11 = SpectralNorm(nn.Conv2d(768, 128, **kwargs_short))
        self.model11 = nn.Sequential(self.nonlinearity, self.down, self.norm(256), self.nonlinearity, self.conv17,
                                     self.norm(128), self.pool_fix)
        self.gateconv11 = SpectralNorm(nn.Conv2d(768, 128, **kwargs))
        self.gate11 = nn.Sequential(self.gateconv11, self.norm(128), self.gate_nonlinearity)
    def forward(self, x1, x2, x3, x4, x5, x6):
        x1 = self.model7(x1) * self.gate7(x1)
        x1 = self.model8(x1) * self.gate8(x1)
        x2 = self.model9(x2) * self.gate9(x2)
        x3 = self.model10(x3) * self.gate10(x3)
        x4 = self.up(x4, (32, 32))
        x5 = self.up(x5, (32, 32))
        x6 = self.up(x6, (32, 32))
        x_DE = torch.cat([x1, x2, x3], 1)
        x_ST = torch.cat([x4, x5, x6], 1)

        x_DE = self.nonlinearity(self.pool_fix(self.model11(x_DE)) + self.shortcut11(x_DE)) * self.gate11(
            x_DE)
        x_ST = self.nonlinearity(self.pool_fix(self.model11(x_ST)) + self.shortcut11(x_ST)) * self.gate11(
            x_ST)
        return x_DE, x_ST

class deconder(nn.Module):
    def __init__(self):
        super(deconder, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_a = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        kwargs_b = {'kernel_size': 5, 'stride': 2, 'padding': 2, 'output_padding': 1}
        kwargs_c = {'kernel_size': 7, 'stride': 2, 'padding': 3, 'output_padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        kwargs_gate = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_out = {'kernel_size': 3, 'padding': 0, 'bias': True}
        # self.nonlinearity = nn.LeakyReLU(0.1)
        self.nonlinearity = nn.GELU()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
        self.pool_fix = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.gate_nonlinearity = nn.Sigmoid()
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)
        # decoder1
        self.conv18 = SpectralNorm(nn.Conv2d(256, 512, **kwargs))
        self.conv19 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.shortcut12 = SpectralNorm(nn.Conv2d(256, 512, **kwargs_short))
        self.model12 = nn.Sequential(self.nonlinearity, self.conv18, self.norm(512), self.nonlinearity,
                                     self.conv19)
        self.gateconv12 = SpectralNorm(nn.Conv2d(256, 512, **kwargs))
        self.gate12 = nn.Sequential(self.gateconv12, self.norm(512), self.gate_nonlinearity)

        # decoder2
        self.conv20 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.conv21 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.shortcut13 = SpectralNorm(nn.Conv2d(512, 512, **kwargs_short))
        self.model13 = nn.Sequential(self.norm(512), self.nonlinearity, self.conv20, self.norm(512), self.nonlinearity,
                                     self.conv21)
        self.gateconv13 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.gate13 = nn.Sequential(self.gateconv13, self.norm(512), self.gate_nonlinearity)

        # decoder3 3*3
        self.conv22 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.conv23 = SpectralNorm(nn.ConvTranspose2d(512, 512, **kwargs_a))
        self.shortcut14 = SpectralNorm(nn.ConvTranspose2d(512, 512, **kwargs_a))
        self.model14 = nn.Sequential(self.norm(512), self.nonlinearity, self.conv22, self.norm(512), self.nonlinearity,
                                     self.conv23)
        self.gateconv14 = SpectralNorm(nn.ConvTranspose2d(512, 512, **kwargs_a))
        self.gate14 = nn.Sequential(self.gateconv14, self.norm(512), self.gate_nonlinearity)

        # out1
        self.conv_out1 = SpectralNorm(nn.Conv2d(512, 128, **kwargs))
        self.conv_out11 = SpectralNorm(nn.Conv2d(128, 3, **kwargs))
        self.model_out1 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out1, self.norm(128),
                                        self.nonlinearity, self.conv_out11, nn.Tanh())

        # decoder3 5*5
        self.conv24 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.conv25 = SpectralNorm(nn.ConvTranspose2d(512, 512, **kwargs_b))
        self.shortcut15 = SpectralNorm(nn.ConvTranspose2d(512, 512, **kwargs_b))
        self.model15 = nn.Sequential(self.norm(512), self.nonlinearity, self.conv24, self.norm(512), self.nonlinearity,
                                     self.conv25)
        self.gateconv15 = SpectralNorm(nn.ConvTranspose2d(512, 512, **kwargs_b))
        self.gate15 = nn.Sequential(self.gateconv15, self.norm(512), self.gate_nonlinearity)

        # out2
        self.conv_out2 = SpectralNorm(nn.Conv2d(512, 128, **kwargs))
        self.conv_out21 = SpectralNorm(nn.Conv2d(128, 3, **kwargs))
        self.model_out2 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out2, self.norm(128),
                                        self.nonlinearity, self.conv_out21, nn.Tanh())

        # decoder3 7*7
        self.conv26 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.conv27 = SpectralNorm(nn.ConvTranspose2d(512, 512, **kwargs_c))
        self.shortcut16 = SpectralNorm(nn.ConvTranspose2d(512, 512, **kwargs_c))
        self.model16 = nn.Sequential(self.norm(512), self.nonlinearity, self.conv26, self.norm(512), self.nonlinearity,
                                     self.conv27)
        self.gateconv16 = SpectralNorm(nn.ConvTranspose2d(512, 512, **kwargs_c))
        self.gate16 = nn.Sequential(self.gateconv16, self.norm(512), self.gate_nonlinearity)

        # out3
        self.conv_out3 = SpectralNorm(nn.Conv2d(512, 128, **kwargs))
        self.conv_out31 = SpectralNorm(nn.Conv2d(128, 3, **kwargs))
        self.model_out3 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out3, self.norm(128),
                                        self.nonlinearity, self.conv_out31, nn.Tanh())

        # decoder4
        self.conv28 = SpectralNorm(nn.Conv2d(512, 256, **kwargs))
        self.conv29 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_a))
        self.shortcut17 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_a))
        self.model17 = nn.Sequential(self.nonlinearity, self.conv28, self.norm(256), self.nonlinearity,
                                     self.conv29)
        self.gateconv17 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_a))
        self.gate17 = nn.Sequential(self.gateconv17, self.norm(256), self.gate_nonlinearity)

        # decoder5
        self.conv30 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.conv31 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.shortcut18 = SpectralNorm(nn.Conv2d(256, 128, **kwargs_short))
        self.model18 = nn.Sequential(self.nonlinearity, self.conv30, self.norm(128), self.nonlinearity,
                                     self.conv31)
        self.gateconv18 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.gate18 = nn.Sequential(self.gateconv18, self.norm(128), self.gate_nonlinearity)

        # decoder6
        self.conv32 = SpectralNorm(nn.Conv2d(128, 64, **kwargs))
        self.conv33 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs_a))
        self.shortcut19 = SpectralNorm(nn.ConvTranspose2d(128, 64, **kwargs_a))
        self.model19 = nn.Sequential(self.nonlinearity, self.conv32, self.norm(64), self.nonlinearity,
                                     self.conv33)
        self.gateconv19 = SpectralNorm(nn.ConvTranspose2d(128, 64, **kwargs_a))
        self.gate19 = nn.Sequential(self.gateconv19, self.norm(64), self.gate_nonlinearity)

        # decoder7
        self.conv34 = SpectralNorm(nn.Conv2d(64, 3, **kwargs))
        self.conv35 = SpectralNorm(nn.Conv2d(3, 3, **kwargs))
        self.shortcut20 = SpectralNorm(nn.Conv2d(64, 3, **kwargs_short))
        self.model20 = nn.Sequential(self.nonlinearity, self.conv34, self.norm(3), self.nonlinearity,
                                     self.conv35)
        self.gateconv20 = SpectralNorm(nn.Conv2d(64, 3, **kwargs))
        self.gate20 = nn.Sequential(self.gateconv20, self.norm(3), self.gate_nonlinearity)

        # out4
        self.conv_out4 = SpectralNorm(nn.Conv2d(3, 3, **kwargs_short))
        self.model_out4 = nn.Sequential(self.nonlinearity, self.conv_out4, nn.Tanh())
        self.transformer = Transformer()
        self.atten = AttnAware()

    def forward(self, x_DE, x_ST):
        result = []
        x = self.transformer(x_DE, x_ST)

        x = self.nonlinearity(self.model12(x) + self.shortcut12(x)) * self.gate12(x)

        x = self.nonlinearity(self.model13(x) + self.shortcut13(x)) * self.gate13(x)

        a = self.nonlinearity(self.model14(x) + self.shortcut14(x)) * self.gate14(x)
        out = self.model_out1(a)

        result.append(out)

        b = self.nonlinearity(self.model15(x) + self.shortcut15(x)) * self.gate15(x)
        out = self.model_out2(b)

        result.append(out)

        c = self.nonlinearity(self.model16(x) + self.shortcut16(x)) * self.gate16(x)
        out = self.model_out3(c)

        result.append(out)
        x = c

        # x = self.atten(a, a, a)
        # x = self.atten(b, b, x)
        # x = self.atten(c, c, x)

        x = self.nonlinearity(self.model17(x) + self.shortcut17(x)) * self.gate17(x)
        x = self.nonlinearity(self.model18(x) + self.shortcut18(x)) * self.gate18(x)
        x = self.nonlinearity(self.model19(x) + self.shortcut19(x)) * self.gate19(x)
        x = self.nonlinearity(self.model20(x) + self.shortcut20(x)) * self.gate20(x)
        out = self.model_out4(x)
        result.append(out)
        return result






class inpainting(nn.Module):
    def __init__(self):
        super(inpainting, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_a = {'kernel_size': 3, 'stride': 2, 'padding': 1, 'output_padding': 1}
        kwargs_t = {'kernel_size': 3, 'stride': 1, 'padding': 1, 'dilation': 1}
        kwargs_b = {'kernel_size': 3, 'stride': 1, 'padding': 2, 'dilation': 2}
        kwargs_c = {'kernel_size': 3, 'stride': 1, 'padding': 4, 'dilation': 4}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        kwargs_gate = {'kernel_size': 4, 'stride': 2, 'padding': 1}
        kwargs_out = {'kernel_size': 3, 'padding': 0, 'bias': True}
        # self.nonlinearity = nn.LeakyReLU(0.1)
        self.nonlinearity = nn.GELU()
        self.pool = nn.AvgPool2d(kernel_size=4, stride=2, padding=1)
        self.pool_fix = nn.AvgPool2d(kernel_size=1, stride=1, padding=0)
        self.gate_nonlinearity = nn.Sigmoid()
        self.norm = functools.partial(nn.InstanceNorm2d, affine=True)

        # encoder1
        self.conv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs))
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.bypass1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs_short))
        self.model1 = nn.Sequential(self.conv1, self.norm(32), self.nonlinearity, self.conv2, self.norm(32), self.pool)
        self.shortcut1 = nn.Sequential(self.pool, self.bypass1)
        self.gateconv1 = SpectralNorm(nn.Conv2d(6, 32, **kwargs_gate))
        self.gate1 = nn.Sequential(self.gateconv1, self.norm(32), self.gate_nonlinearity)

        # encoder2
        self.conv3 = SpectralNorm(nn.Conv2d(32, 64, **kwargs))
        self.conv4 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.shortcut2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_short))
        self.model2 = nn.Sequential(self.conv3, self.norm(64), self.nonlinearity, self.conv4,
                                    self.norm(64), self.pool)
        self.gateconv2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_gate))
        self.gate2 = nn.Sequential(self.gateconv2, self.norm(64), self.gate_nonlinearity)

        # encoder3
        self.conv5 = SpectralNorm(nn.Conv2d(64, 128, **kwargs))
        self.conv6 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.shortcut3 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_short))
        self.model3 = nn.Sequential(self.conv5, self.norm(128), self.nonlinearity, self.conv6,
                                    self.norm(128), self.pool)
        self.gateconv3 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_gate))
        self.gate3 = nn.Sequential(self.gateconv3, self.norm(128), self.gate_nonlinearity)

        # encoder4
        self.conv7 = SpectralNorm(nn.Conv2d(128, 256, **kwargs))
        self.conv8 = SpectralNorm(nn.Conv2d(256, 256, **kwargs))
        self.shortcut4 = SpectralNorm(nn.Conv2d(128, 256, **kwargs_short))
        self.model4 = nn.Sequential(self.conv7, self.norm(256), self.nonlinearity, self.conv8,
                                    self.norm(256), self.pool)
        self.gateconv4 = SpectralNorm(nn.Conv2d(128, 256, **kwargs_gate))
        self.gate4 = nn.Sequential(self.gateconv4, self.norm(256), self.gate_nonlinearity)

        # encoder5
        self.conv9 = SpectralNorm(nn.Conv2d(256, 512, **kwargs))
        self.conv10 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.shortcut5 = SpectralNorm(nn.Conv2d(256, 512, **kwargs_short))
        self.model5 = nn.Sequential(self.conv9, self.norm(512), self.nonlinearity, self.conv10,
                                    self.norm(512), self.pool)
        self.gateconv5 = SpectralNorm(nn.Conv2d(256, 512, **kwargs_gate))
        self.gate5 = nn.Sequential(self.gateconv5, self.norm(512), self.gate_nonlinearity)

        # encoder6
        self.conv11 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.conv12 = SpectralNorm(nn.Conv2d(512, 512, **kwargs))
        self.shortcut6 = SpectralNorm(nn.Conv2d(512, 512, **kwargs_short))
        self.model6 = nn.Sequential(self.conv11, self.norm(512), self.nonlinearity, self.conv12,
                                    self.norm(512), self.pool)
        self.gateconv6 = SpectralNorm(nn.Conv2d(512, 512, **kwargs_gate))
        self.gate6 = nn.Sequential(self.gateconv6, self.norm(512), self.gate_nonlinearity)

        # 特征融合
        self.down_128 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_gate))
        self.conv13 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.shortcut7 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_short))
        self.model7 = nn.Sequential(self.down_128, self.norm(64), self.nonlinearity, self.conv13,self.nonlinearity)
        self.gateconv7 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_gate))
        self.gate7 = nn.Sequential(self.gateconv7, self.gate_nonlinearity)
        # 2
        self.down_129 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_gate))
        self.conv14 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.shortcut8 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_short))
        self.model8 = nn.Sequential(self.down_129, self.norm(128), self.nonlinearity,self.conv14,self.nonlinearity)
        self.gateconv8 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_gate))
        self.gate8 = nn.Sequential(self.gateconv8, self.gate_nonlinearity)

        # 第二层
        self.down_64 = SpectralNorm(nn.Conv2d(64, 128,**kwargs_gate))
        self.conv15 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.shortcut9 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_short))
        self.model9 = nn.Sequential(self.down_64, self.norm(128), self.nonlinearity,self.conv15,self.nonlinearity)
        self.gateconv9 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_gate))
        self.gate9 = nn.Sequential(self.gateconv9, self.gate_nonlinearity)

        # 第三层
        self.down_32 = SpectralNorm(nn.Conv2d(128, 128, 1, 1))
        self.conv16 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.shortcut10 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model10 = nn.Sequential(self.down_32, self.norm(128), self.nonlinearity, self.conv16,self.nonlinearity)
        self.gateconv10 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.gate10 = nn.Sequential(self.gateconv10, self.gate_nonlinearity)

        # 第四层
        self.conv_4 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.conv_5 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs_a))
        self.shortcut_4 = SpectralNorm(nn.ConvTranspose2d(256, 128, **kwargs_a))
        self.model_4 = nn.Sequential(self.conv_4, self.norm(128), self.nonlinearity, self.conv_5, self.nonlinearity)
        self.gateconv_4 = SpectralNorm(nn.ConvTranspose2d(256, 128, **kwargs_a))
        self.gate_4 = nn.Sequential(self.gateconv_4, self.gate_nonlinearity)

        # 第5，6层
        self.up = ConvUp()

        # 融合
        self.down = SpectralNorm(nn.Conv2d(384, 128, 1, 1))
        self.conv17 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.shortcut11 = SpectralNorm(nn.Conv2d(384, 128, **kwargs_short))
        self.model11 = nn.Sequential(self.down, self.norm(128), self.nonlinearity, self.conv17,
                                     self.norm(128), self.pool_fix)
        self.gateconv11 = SpectralNorm(nn.Conv2d(384, 128, **kwargs))
        self.gate11 = nn.Sequential(self.gateconv11, self.norm(128), self.gate_nonlinearity)

        #######
        self.convA1 = SpectralNorm(nn.Conv2d(256, 256, **kwargs))
        self.convA2 = SpectralNorm(nn.Conv2d(256, 256, **kwargs))
        self.shortcutA3 = SpectralNorm(nn.Conv2d(256, 256, **kwargs_short))
        self.modelA = nn.Sequential(self.nonlinearity, self.convA1, self.norm(256), self.nonlinearity,
                                    self.convA2)
        self.gateconvA4 = SpectralNorm(nn.Conv2d(256, 256, **kwargs))
        self.gateA5 = nn.Sequential(self.gateconvA4, self.norm(256), self.gate_nonlinearity)
        #
        self.convB1 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.convB2 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.shortcutB3 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.modelB = nn.Sequential(self.nonlinearity, self.convB1, self.norm(128), self.nonlinearity,
                                    self.convB2)
        self.gateconvB4 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.gateB5 = nn.Sequential(self.gateconvB4, self.norm(128), self.gate_nonlinearity)
        #
        self.convC1 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.convC2 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.shortcutC3 = SpectralNorm(nn.Conv2d(64, 64, **kwargs_short))
        self.modelC = nn.Sequential(self.nonlinearity, self.convC1, self.norm(64), self.nonlinearity,
                                    self.convC2)
        self.gateconvC4 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.gateC5 = nn.Sequential(self.gateconvC4, self.norm(64), self.gate_nonlinearity)
        # 1
        self.convD1 = SpectralNorm(nn.Conv2d(512, 256, **kwargs))
        self.convD2 = SpectralNorm(nn.ConvTranspose2d(256, 256, **kwargs_a))
        self.shortcutD3 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_a))
        self.modelD = nn.Sequential(self.nonlinearity, self.convD1, self.norm(256), self.nonlinearity,
                                    self.convD2)
        self.gateconvD4 = SpectralNorm(nn.ConvTranspose2d(512, 256, **kwargs_a))
        self.gateD5 = nn.Sequential(self.gateconvD4, self.norm(256), self.gate_nonlinearity)
        # 2
        self.convE1 = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.convE2 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs_a))
        self.shortcutE3 = SpectralNorm(nn.ConvTranspose2d(256, 128, **kwargs_a))
        self.modelE = nn.Sequential(self.nonlinearity, self.convE1, self.norm(128), self.nonlinearity,
                                    self.convE2)
        self.gateconvE4 = SpectralNorm(nn.ConvTranspose2d(256, 128, **kwargs_a))
        self.gateE5 = nn.Sequential(self.gateconvE4, self.norm(128), self.gate_nonlinearity)
        # 3
        self.convF1 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.convF2 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs_a))
        self.shortcutF3 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs_a))
        self.modelF = nn.Sequential(self.nonlinearity, self.convF1, self.norm(128), self.nonlinearity,
                                    self.convF2)
        self.gateconvF4 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs_a))
        self.gateF5 = nn.Sequential(self.gateconvF4, self.norm(128), self.gate_nonlinearity)

        # decoder1
        self.conv18 = SpectralNorm(nn.Conv2d(384, 256, **kwargs))
        self.conv19 = SpectralNorm(nn.Conv2d(256, 256, **kwargs))
        self.shortcut12 = SpectralNorm(nn.Conv2d(384, 256, **kwargs_short))
        self.model12 = nn.Sequential(self.conv18, self.norm(256), self.nonlinearity,
                                     self.conv19)
        self.gateconv12 = SpectralNorm(nn.Conv2d(384, 256, **kwargs))
        self.gate12 = nn.Sequential(self.gateconv12, self.norm(256), self.gate_nonlinearity)

        # decoder1-2
        self.convr = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.convr1 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.shortcutr = SpectralNorm(nn.Conv2d(256, 128, **kwargs_short))
        self.modelr = nn.Sequential(self.norm(256), self.nonlinearity, self.convr, self.norm(128), self.nonlinearity,
                                     self.convr1)
        self.gateconvr = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.gater = nn.Sequential(self.gateconvr, self.norm(128), self.gate_nonlinearity)

        # decoder2
        self.conv20 = SpectralNorm(nn.Conv2d(384, 128, **kwargs))
        self.conv21 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.shortcut13 = SpectralNorm(nn.Conv2d(384, 128, **kwargs_short))
        self.model13 = nn.Sequential(self.nonlinearity, self.conv20, self.norm(128), self.nonlinearity,
                                     self.conv21)
        self.gateconv13 = SpectralNorm(nn.Conv2d(384, 128, **kwargs))
        self.gate13 = nn.Sequential(self.gateconv13, self.norm(128), self.gate_nonlinearity)

        # decoder3 3*3
        self.conv22 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv23 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_t))
        self.shortcut14 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_t))
        self.model14 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv22, self.norm(128), self.nonlinearity,
                                     self.conv23)
        self.gateconv14 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_t))
        self.gate14 = nn.Sequential(self.gateconv14, self.norm(128), self.gate_nonlinearity)

        # out1
        self.conv_out1 = SpectralNorm(nn.Conv2d(128, 64, **kwargs))
        self.conv_out11 = SpectralNorm(nn.Conv2d(64, 3, **kwargs))
        self.model_out1 = nn.Sequential(self.nonlinearity, self.conv_out1, self.norm(64),
                                        self.nonlinearity, self.conv_out11, nn.Tanh())

        # decoder3 5*5
        self.conv24 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv25 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_b))
        self.shortcut15 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_b))
        self.model15 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv24, self.norm(128), self.nonlinearity,
                                     self.conv25)
        self.gateconv15 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_b))
        self.gate15 = nn.Sequential(self.gateconv15, self.norm(128), self.gate_nonlinearity)

        # out2
        self.conv_out2 = SpectralNorm(nn.Conv2d(64, 32, **kwargs))
        self.conv_out21 = SpectralNorm(nn.Conv2d(32, 3, **kwargs))
        self.model_out2 = nn.Sequential(self.nonlinearity,  self.conv_out2, self.norm(32),
                                        self.nonlinearity, self.conv_out21, nn.Tanh())

        # decoder3 7*7
        self.conv26 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv27 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_c))
        self.shortcut16 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_c))
        self.model16 = nn.Sequential(self.norm(128), self.nonlinearity, self.conv26, self.norm(128), self.nonlinearity,
                                     self.conv27)
        self.gateconv16 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_c))
        self.gate16 = nn.Sequential(self.gateconv16, self.norm(128), self.gate_nonlinearity)

        # out3
        self.conv_out3 = SpectralNorm(nn.Conv2d(128, 64, **kwargs))
        self.conv_out31 = SpectralNorm(nn.Conv2d(64, 3, **kwargs))
        self.model_out3 = nn.Sequential(self.nonlinearity, nn.ReflectionPad2d(1), self.conv_out3, self.norm(64),
                                        self.nonlinearity, self.conv_out31, nn.Tanh())

        # decoder4
        self.convz = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.convz1 = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs_a))
        self.shortcutz = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs_a))
        self.modelz = nn.Sequential(self.norm(128), self.nonlinearity, self.convz, self.norm(128), self.nonlinearity,
                                     self.convz1)
        self.gateconvz = SpectralNorm(nn.ConvTranspose2d(128, 128, **kwargs_a))
        self.gatez = nn.Sequential(self.gateconvz, self.norm(128), self.gate_nonlinearity)


        # decoder5
        self.conv28 = SpectralNorm(nn.Conv2d(128, 64, **kwargs))
        self.conv29 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.shortcut17 = SpectralNorm(nn.Conv2d(128, 64, **kwargs))
        self.model17 = nn.Sequential(self.norm(128),self.nonlinearity, self.conv28, self.norm(64), self.nonlinearity,
                                     self.conv29)
        self.gateconv17 = SpectralNorm(nn.Conv2d(128, 64, **kwargs))
        self.gate17 = nn.Sequential(self.gateconv17, self.norm(64), self.gate_nonlinearity)

        # decoder6
        self.conv30 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv31 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs_a))
        self.shortcut18 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs_a))
        self.model18 = nn.Sequential(self.norm(64), self.nonlinearity, self.conv30, self.norm(64), self.nonlinearity,
                                     self.conv31)
        self.gateconv18 = SpectralNorm(nn.ConvTranspose2d(64, 64, **kwargs_a))
        self.gate18 = nn.Sequential(self.gateconv18, self.norm(64), self.gate_nonlinearity)

        # decoder7
        self.conv32 = SpectralNorm(nn.Conv2d(64, 32, **kwargs))
        self.conv33 = SpectralNorm(nn.ConvTranspose2d(32, 32, **kwargs_a))
        self.shortcut19 = SpectralNorm(nn.ConvTranspose2d(64, 32, **kwargs_a))
        self.model19 = nn.Sequential(self.norm(64), self.nonlinearity, self.conv32, self.norm(32), self.nonlinearity,
                                     self.conv33)
        self.gateconv19 = SpectralNorm(nn.ConvTranspose2d(64, 32, **kwargs_a))
        self.gate19 = nn.Sequential(self.gateconv19, self.norm(32), self.gate_nonlinearity)

        # decoder8
        self.conv34 = SpectralNorm(nn.Conv2d(32, 3, **kwargs))
        self.conv35 = SpectralNorm(nn.Conv2d(3, 3, **kwargs))
        self.shortcut20 = SpectralNorm(nn.Conv2d(32, 3, **kwargs_short))
        self.model20 = nn.Sequential(self.norm(32), self.nonlinearity, self.conv34, self.norm(3), self.nonlinearity,
                                     self.conv35)
        self.gateconv20 = SpectralNorm(nn.Conv2d(32, 3, **kwargs))
        self.gate20 = nn.Sequential(self.gateconv20, self.norm(3), self.gate_nonlinearity)

        # out4
        self.conv_out4 = SpectralNorm(nn.Conv2d(3, 3, **kwargs_short))
        self.model_out4 = nn.Sequential(self.nonlinearity,  self.conv_out4, nn.Tanh())

        self.transformer = Transformer()
        self.atten1 = Auto_Attn(128)
        self.atten2 = Auto_Attn(128)
        self.atten3 = Auto_Attn(128)
        self.conv_1 = SpectralNorm(nn.Conv2d(384, 128, **kwargs_short))
        self.conv_2 = SpectralNorm(nn.Conv2d(192, 64, **kwargs_short))
        self.conv_3 = SpectralNorm(nn.Conv2d(128, 64, **kwargs_short))
        self.res = res()
        self.atten4 = Auto_Attn(64)
        self.atten5 = Auto_Attn(64)
        self.atten6 = Auto_Attn(64)
        self.ad_pool = nn.AdaptiveAvgPool2d(output_size=(1, 1))
        self.liner = nn.Linear(256, 1024)
        self.liner1 = nn.Linear(1024, 256)

        self.convzz = SpectralNorm(nn.Conv2d(256, 128, **kwargs))
        self.modelzz = nn.Sequential(self.convzz, self.gate_nonlinearity)

    def forward(self, x):
        feature = []
        result = []

        x1 = self.nonlinearity(self.model1(x) + self.shortcut1(x)) * self.gate1(x)
        x2 = self.nonlinearity(self.model2(x1) + self.pool(self.shortcut2(x1))) * self.gate2(x1)
        x3 = self.nonlinearity(self.model3(x2) + self.pool(self.shortcut3(x2))) * self.gate3(x2)
        x4 = self.nonlinearity(self.model4(x3) + self.pool(self.shortcut4(x3))) * self.gate4(x3)
        x5 = self.nonlinearity(self.model5(x4) + self.pool(self.shortcut5(x4))) * self.gate5(x4)
        x6 = self.nonlinearity(self.model6(x5) + self.pool(self.shortcut6(x5))) * self.gate6(x5)

        ht = self.nonlinearity(self.modelD(x6) + self.shortcutD3(x6)) * self.gateD5(x6)
        ht = self.nonlinearity(self.modelE(ht) + self.shortcutE3(ht)) * self.gateE5(ht)
        ht = self.nonlinearity(self.modelF(ht) + self.shortcutF3(ht)) * self.gateF5(ht)


        feature.append(x6)

        x1 = self.nonlinearity(self.model7(x1) + self.pool(self.shortcut7(x1))) * self.gate7(x1)
        x1 = self.nonlinearity(self.model8(x1) + self.pool(self.shortcut8(x1))) * self.gate8(x1)
        x2 = self.nonlinearity(self.model9(x2) + self.pool(self.shortcut9(x2))) * self.gate9(x2)
        x3 = self.nonlinearity(self.model10(x3) + self.shortcut10(x3)) * self.gate10(x3)
        x4 = self.nonlinearity(self.model_4(x4) + self.shortcut_4(x4)) * self.gate_4(x4)


        x5 = self.up(x5, (32, 32))
        x6 = self.up(x6, (32, 32))
        x_DE = torch.cat([x1, x2, x3], 1)
        x_ST = torch.cat([x4, x5, x6], 1)


        x_DE = self.nonlinearity(self.pool_fix(self.model11(x_DE)) + self.shortcut11(x_DE)) * self.gate11(x_DE)
        #x_ST = self.nonlinearity(self.pool_fix(self.model11(x_ST)) + self.shortcut11(x_ST)) * self.gate11(x_ST)
        x_ST = self.nonlinearity(self.model13(x_ST) + self.shortcut13(x_ST)) * self.gate13(x_ST)

        fengshu = torch.cat([x_DE, x_ST], dim=1)
        fengshu = self.modelzz(fengshu)
        x_t = fengshu * x_ST
        x_s = fengshu * x_DE
     
        x_t = self.ad_pool(x_t)
        x_s = self.ad_pool(x_s)

        fengshu = torch.cat([x_t, x_s], dim=1)
        fengshu = rearrange(fengshu, 'b n h d->b (n h d)')
        fengshu = self.liner(fengshu)
        fengshu = self.liner1(fengshu)
        fengshu = F.softmax(fengshu,dim=-1)
        fengshu = rearrange(fengshu, 'b (n h d)->b n h d',h=1,d=1)
        x_t, x_s = fengshu.chunk(2,dim=1)
        x_ST = x_ST * x_s + x_ST
        x_DE = x_DE * x_t + x_DE

        st = x_ST
        de = x_DE
        x = self.transformer(x_DE, x_ST, ht)

        x = self.nonlinearity(self.model12(x) + self.shortcut12(x)) * self.gate12(x)
        x = self.nonlinearity(self.modelA(x) + self.shortcutA3(x)) * self.gateA5(x)
        x = self.nonlinearity(self.modelr(x) + self.shortcutr(x)) * self.gater(x)


        a = self.nonlinearity(self.model14(x) + self.shortcut14(x)) * self.gate14(x)
        out = self.model_out1(a)
        result.append(out)


        b = self.nonlinearity(self.model15(x) + self.shortcut15(x)) * self.gate15(x)

        c = self.nonlinearity(self.model16(x) + self.shortcut16(x)) * self.gate16(x)

        xa = self.atten1(a, a, a)
        xb = self.atten2(b, b, xa)
        xc = self.atten3(c, c, xb)
        cat = torch.cat([xa, xb, xc], dim=1)
        cat = self.nonlinearity(self.conv_1(cat))
        cats = nn.Sigmoid()(cat)
        sts = cats * st
        x = x + sts * 0.1 + cat

        x = self.nonlinearity(self.modelz(x) + self.shortcutz(x)) * self.gatez(x)
        x = self.nonlinearity(self.modelB(x) + self.shortcutB3(x)) * self.gateB5(x)
        x = self.nonlinearity(self.model17(x) + self.shortcut17(x)) * self.gate17(x)

        a, b, c, result1 = self.res(x)
        result.append(result1)



        xa = self.atten4(a, a, a)
        xb = self.atten5(b, b, xa)
        xc = self.atten6(c, c, xb)
        cat = torch.cat([xa, xb, xc], dim=1)

        cat = self.nonlinearity(self.conv_2(cat))
        cats = nn.Sigmoid()(cat)
        de = F.interpolate(input=de, size=(64, 64), mode='bilinear')
        de = self.nonlinearity(self.conv_3(de))
        #x = F.interpolate(input=x, size=(64, 64), mode='bilinear')
        sts = cats * de
        x = x + sts * 0.1 + cat

        x = self.nonlinearity(self.modelC(x) + self.shortcutC3(x)) * self.gateC5(x)
        x = self.nonlinearity(self.model18(x) + self.shortcut18(x)) * self.gate18(x)
        out = self.model_out2(x)
        result.append(out)

        x = self.nonlinearity(self.model19(x) + self.shortcut19(x)) * self.gate19(x)
        x = self.nonlinearity(self.model20(x) + self.shortcut20(x)) * self.gate20(x)

        out = self.model_out4(x)

        result.append(out)
        return feature, result

class Discriminator_1(nn.Module):
    def __init__(self):
        super(Discriminator_1, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # encoder1
        self.conv1 = SpectralNorm(nn.Conv2d(3, 32, **kwargs))
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.bypass1 = SpectralNorm(nn.Conv2d(3, 32, **kwargs_short))
        self.model1 = nn.Sequential(self.conv1, self.nonlinearity, self.conv2, self.pool)
        self.shortcut1 = nn.Sequential(self.pool, self.bypass1)

        # encoder2
        self.conv3 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.conv4 = SpectralNorm(nn.Conv2d(32, 64, **kwargs))
        self.bypass2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_short))
        self.model2 = nn.Sequential(self.nonlinearity, self.conv3, self.nonlinearity, self.conv4)
        self.shortcut2 = nn.Sequential(self.bypass2)

        # encoder3
        self.conv5 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv6 = SpectralNorm(nn.Conv2d(64, 128, **kwargs))
        self.bypass3 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_short))
        self.model3 = nn.Sequential(self.nonlinearity, self.conv5, self.nonlinearity, self.conv6)
        self.shortcut3 = nn.Sequential(self.bypass3)

        # encoder4
        self.conv7 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv8 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass4 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model4 = nn.Sequential(self.nonlinearity, self.conv7, self.nonlinearity, self.conv8)
        self.shortcut4 = nn.Sequential(self.bypass4)

        # encoder5
        self.conv9 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv10 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model5 = nn.Sequential(self.nonlinearity, self.conv9, self.nonlinearity, self.conv10)
        self.shortcut5 = nn.Sequential(self.bypass5)

        # encoder6
        self.conv11 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv12 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass6 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model6 = nn.Sequential(self.nonlinearity, self.conv11, self.nonlinearity, self.conv12)
        self.shortcut6 = nn.Sequential(self.bypass6)

        # concat
        self.concat = SpectralNorm(nn.Conv2d(128, 1, 3))

    def forward(self, x):
        x = self.model1(x) + self.shortcut1(x)
        x = self.pool(self.model2(x)) + self.pool(self.shortcut2(x))
        x = self.pool(self.model3(x)) + self.pool(self.shortcut3(x))
        out = self.pool(self.model4(x)) + self.pool(self.shortcut4(x))
        out = self.pool(self.model5(out)) + self.pool(self.shortcut5(out))
        out = self.pool(self.model6(out)) + self.pool(self.shortcut6(out))
        out = self.concat(self.nonlinearity(out))

        return out


class Discriminator_2(nn.Module):
    def __init__(self):
        super(Discriminator_2, self).__init__()
        kwargs = {'kernel_size': 3, 'stride': 1, 'padding': 1}
        kwargs_short = {'kernel_size': 1, 'stride': 1, 'padding': 0}
        self.nonlinearity = nn.LeakyReLU(0.1)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

        # encoder0
        self.conv1 = SpectralNorm(nn.Conv2d(3, 32, **kwargs))
        self.conv2 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.bypass1 = SpectralNorm(nn.Conv2d(3, 32, **kwargs_short))
        self.model1 = nn.Sequential(self.conv1, self.nonlinearity, self.conv2, self.pool)
        self.shortcut1 = nn.Sequential(self.pool, self.bypass1)

        # encoder1
        self.conv3 = SpectralNorm(nn.Conv2d(32, 32, **kwargs))
        self.conv4 = SpectralNorm(nn.Conv2d(32, 64, **kwargs))
        self.bypass2 = SpectralNorm(nn.Conv2d(32, 64, **kwargs_short))
        self.model2 = nn.Sequential(self.nonlinearity, self.conv3, self.nonlinearity, self.conv4)
        self.shortcut2 = nn.Sequential(self.bypass2)

        # encoder2
        self.conv5 = SpectralNorm(nn.Conv2d(64, 64, **kwargs))
        self.conv6 = SpectralNorm(nn.Conv2d(64, 128, **kwargs))
        self.bypass3 = SpectralNorm(nn.Conv2d(64, 128, **kwargs_short))
        self.model3 = nn.Sequential(self.nonlinearity, self.conv5, self.nonlinearity, self.conv6)
        self.shortcut3 = nn.Sequential(self.bypass3)

        # encoder3
        self.conv7 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv8 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass4 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model4 = nn.Sequential(self.nonlinearity, self.conv7, self.nonlinearity, self.conv8)
        self.shortcut4 = nn.Sequential(self.bypass4)

        # encoder4
        self.conv9 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv10 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass5 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model5 = nn.Sequential(self.nonlinearity, self.conv9, self.nonlinearity, self.conv10)
        self.shortcut5 = nn.Sequential(self.bypass5)

        # encoder5
        self.conv11 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.conv12 = SpectralNorm(nn.Conv2d(128, 128, **kwargs))
        self.bypass6 = SpectralNorm(nn.Conv2d(128, 128, **kwargs_short))
        self.model6 = nn.Sequential(self.nonlinearity, self.conv11, self.nonlinearity, self.conv12)
        self.shortcut6 = nn.Sequential(self.bypass6)

        # concat
        self.concat = SpectralNorm(nn.Conv2d(128, 1, 3))

    def forward(self, x):
        x = self.model1(x) + self.shortcut1(x)
        x = self.pool(self.model2(x)) + self.pool(self.shortcut2(x))
        x = self.pool(self.model3(x)) + self.pool(self.shortcut3(x))
        out = self.pool(self.model4(x)) + self.pool(self.shortcut4(x))
        out = self.pool(self.model5(out)) + self.pool(self.shortcut5(out))
        out = self.concat(self.nonlinearity(out))

        return out



class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.enc_1 = nn.Sequential(*vgg16.features[:5])
        self.enc_2 = nn.Sequential(*vgg16.features[5:10])
        self.enc_3 = nn.Sequential(*vgg16.features[10:17])

        # fix the encoder
        for i in range(3):
            for param in getattr(self, 'enc_{:d}'.format(i + 1)).parameters():
                param.requires_grad = False

    def forward(self, image):
        results = [image]
        for i in range(3):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

