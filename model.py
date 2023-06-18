from torch import nn
import torch
import math
import torch.nn.functional as F
import numpy as np
import pywt
# from utils import dc
from torch.autograd import Function
from torchvision import transforms
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class DWTFunction_2D(Function):
    @staticmethod
    def forward(ctx, input, matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1):
        ctx.save_for_backward(matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1)
        L = torch.matmul(matrix_Low_0, input)
        H = torch.matmul(matrix_High_0, input)
        LL = torch.matmul(L, matrix_Low_1)
        LH = torch.matmul(L, matrix_High_1)
        HL = torch.matmul(H, matrix_Low_1)
        HH = torch.matmul(H, matrix_High_1)
        return LL, LH, HL, HH
    @staticmethod
    def backward(ctx, grad_LL, grad_LH, grad_HL, grad_HH):
        matrix_Low_0, matrix_Low_1, matrix_High_0, matrix_High_1 = ctx.saved_variables
        grad_L = torch.add(torch.matmul(grad_LL, matrix_Low_1.t()), torch.matmul(grad_LH, matrix_High_1.t()))
        grad_H = torch.add(torch.matmul(grad_HL, matrix_Low_1.t()), torch.matmul(grad_HH, matrix_High_1.t()))
        grad_input = torch.add(torch.matmul(matrix_Low_0.t(), grad_L), torch.matmul(matrix_High_0.t(), grad_H))
        return grad_input, None, None, None, None


class DWT_2D(nn.Module):
    """
    input: the 2D data to be decomposed -- (N, C, H, W)
    output -- lfc: (N, C, H/2, W/2)
              hfc_lh: (N, C, H/2, W/2)
              hfc_hl: (N, C, H/2, W/2)
              hfc_hh: (N, C, H/2, W/2)
    """
    def __init__(self, wavename):
        """
        2D discrete wavelet transform (DWT) for 2D image decomposition
        :param wavename: pywt.wavelist(); in the paper, 'chx.y' denotes 'biorx.y'.
        """
        super(DWT_2D, self).__init__()
        wavelet = pywt.Wavelet(wavename)
        self.band_low = wavelet.rec_lo
        self.band_high = wavelet.rec_hi
        assert len(self.band_low) == len(self.band_high)
        self.band_length = len(self.band_low)
        assert self.band_length % 2 == 0
        self.band_length_half = math.floor(self.band_length / 2)

    def get_matrix(self):
        """
        生成变换矩阵
        generating the matrices: \mathcal{L}, \mathcal{H}
        :return: self.matrix_low = \mathcal{L}, self.matrix_high = \mathcal{H}
        """
        L1 = np.max((self.input_height, self.input_width))
        L = math.floor(L1 / 2)
        matrix_h = np.zeros( ( L,      L1 + self.band_length - 2 ) )
        matrix_g = np.zeros( ( L1 - L, L1 + self.band_length - 2 ) )
        end = None if self.band_length_half == 1 else (-self.band_length_half+1)

        index = 0
        for i in range(L):
            for j in range(self.band_length):
                matrix_h[i, index+j] = self.band_low[j]
            index += 2
        matrix_h_0 = matrix_h[0:(math.floor(self.input_height / 2)), 0:(self.input_height + self.band_length - 2)]
        matrix_h_1 = matrix_h[0:(math.floor(self.input_width / 2)), 0:(self.input_width + self.band_length - 2)]

        index = 0
        for i in range(L1 - L):
            for j in range(self.band_length):
                matrix_g[i, index+j] = self.band_high[j]
            index += 2
        matrix_g_0 = matrix_g[0:(self.input_height - math.floor(self.input_height / 2)),0:(self.input_height + self.band_length - 2)]
        matrix_g_1 = matrix_g[0:(self.input_width - math.floor(self.input_width / 2)),0:(self.input_width + self.band_length - 2)]

        matrix_h_0 = matrix_h_0[:,(self.band_length_half-1):end]
        matrix_h_1 = matrix_h_1[:,(self.band_length_half-1):end]
        matrix_h_1 = np.transpose(matrix_h_1)
        matrix_g_0 = matrix_g_0[:,(self.band_length_half-1):end]
        matrix_g_1 = matrix_g_1[:,(self.band_length_half-1):end]
        matrix_g_1 = np.transpose(matrix_g_1)

        if torch.cuda.is_available():
            self.matrix_low_0 = torch.Tensor(matrix_h_0).to(device)
            self.matrix_low_1 = torch.Tensor(matrix_h_1).to(device)
            self.matrix_high_0 = torch.Tensor(matrix_g_0).to(device)
            self.matrix_high_1 = torch.Tensor(matrix_g_1).to(device)
            # self.matrix_low_0 = torch.Tensor(matrix_h_0).cuda()
            # self.matrix_low_1 = torch.Tensor(matrix_h_1).cuda()
            # self.matrix_high_0 = torch.Tensor(matrix_g_0).cuda()
            # self.matrix_high_1 = torch.Tensor(matrix_g_1).cuda()
        else:
            self.matrix_low_0 = torch.Tensor(matrix_h_0)
            self.matrix_low_1 = torch.Tensor(matrix_h_1)
            self.matrix_high_0 = torch.Tensor(matrix_g_0)
            self.matrix_high_1 = torch.Tensor(matrix_g_1)

    def forward(self, input):
        """
        input_lfc = \mathcal{L} * input * \mathcal{L}^T
        input_hfc_lh = \mathcal{H} * input * \mathcal{L}^T
        input_hfc_hl = \mathcal{L} * input * \mathcal{H}^T
        input_hfc_hh = \mathcal{H} * input * \mathcal{H}^T
        :param input: the 2D data to be decomposed
        :return: the low-frequency and high-frequency components of the input 2D data
        """
        assert len(input.size()) == 4
        self.input_height = input.size()[-2]
        self.input_width = input.size()[-1]
        self.get_matrix()
        return DWTFunction_2D.apply(input, self.matrix_low_0, self.matrix_low_1, self.matrix_high_0, self.matrix_high_1)


class Downsample_v2(nn.Module):
    """
        for ResNet_A
        X --> 1/4*(X_ll + X_lh + X_hl + X_hh)
    """
    def __init__(self, wavename = 'haar'):
        super(Downsample_v2, self).__init__()
        self.dwt = DWT_2D(wavename=wavename)

    def forward(self, input):
        LL, LH, HL, HH = self.dwt(input)
        return LL


class CA_Model(nn.Module):
    def __init__(self, dim):
        super(CA_Model, self).__init__()
        self.dim = dim
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(self.dim, self.dim//4, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(self.dim//4, self.dim, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc2(self.relu(self.fc1(out)))
        return self.sigmoid(out)


class MFB(nn.Module):
    def __init__(self, dim):
        super(MFB, self).__init__()
        self.dim = dim
        self.conv = nn.Conv2d(2*self.dim, self.dim, kernel_size=1, padding=0)
        self.ca = CA_Model(2*self.dim)



    def forward(self, lr, lr2):
        in_add = lr + lr2
        in_max = torch.max(lr, lr2)
        in_all = torch.cat((in_add, in_max), 1)
        w = self.ca(in_all)
        x = in_all * w
        out = self.conv(x)

        return out


class MFB5(nn.Module):
    def __init__(self, dim):
        super(MFB5, self).__init__()
        self.dim = dim
        self.conv = nn.Conv2d(3*self.dim, self.dim, kernel_size=1, padding=0)
        self.ca = CA_Model(2*self.dim)

    def forward(self, x, lr_conv, lr2_conv):
        identity = x
        in_add = lr_conv + lr2_conv
        in_max = torch.max(lr2_conv, lr_conv)
        in_all = torch.cat((in_add, in_max), 1)
        w = self.ca(in_all)
        x = in_all * w
        x = torch.cat((identity, x), 1)
        out = self.conv(x)

        return out


class MFB1(nn.Module):
    def __init__(self, dim):
        super(MFB1, self).__init__()
        self.dim = dim
        self.conv = nn.Conv2d(2*self.dim, self.dim, kernel_size=1, padding=0)

    def forward(self, x, lr_conv):
        x = torch.cat((lr_conv, x), 1)
        out = self.conv(x)

        return out


class Chan_Conv(nn.Module):
    def __init__(self, kernel_size=3, dims=256):
        super(Chan_Conv, self).__init__()
        self.conv_h = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.conv_w = nn.Conv2d(in_channels=dims, out_channels=dims, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x_h = torch.permute(x, (0, 3, 1, 2))  # B, W, C, H
        x_h = torch.permute(x_h, (0, 1, 3, 2))  # B, W, H, C
        x_h = self.conv_h(x_h)
        x_w = torch.permute(x_h, (0, 2, 1, 3))  # B, H, W, C
        x_w = self.conv_w(x_w)
        x_w = torch.permute(x_w, (0, 3, 1, 2))

        # x = x_w * x + x
        x = self.softmax(x_w) * x + x

        return x


class BasicBlock_E(nn.Module):
    def __init__(self, kernel_size = 3,dims = 32):
        super(BasicBlock_E, self).__init__()
        self.kernel_size = kernel_size
        self.dims = dims
        self.avgpool = nn.AvgPool2d(kernel_size=8)
        self.upsample = nn.Upsample(scale_factor=8, mode='bicubic')
        # self.avgpool = nn.AvgPool2d(kernel_size=6)
        # self.upsample = nn.Upsample(scale_factor=6, mode='bicubic')
        block_1 = [
            nn.Sequential(
                nn.Conv2d(dims, dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.LeakyReLU(inplace=True)
            )
        ]
        self.block_1 = nn.Sequential(*block_1)
        block_2 = [
            nn.Sequential(
                nn.Conv2d(dims, dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.ELU(inplace=True),
                nn.Conv2d(dims, dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.ELU(inplace=True),
                nn.Conv2d(dims, dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.ELU(inplace=True)
            )
        ]
        self.block_2 = nn.Sequential(*block_2)

    def forward(self, x):
        lp = self.block_1(self.upsample(self.avgpool(x)))
        hp = self.block_2(x - lp)
        x = lp + hp

        return x


class BasicBlock_D(nn.Module):
    def __init__(self, kernel_size = 3,dims = 32):
        super(BasicBlock_D, self).__init__()
        self.kernel_size = kernel_size
        self.dims = dims
        block_1 = [
            nn.Sequential(
                nn.Conv2d(dims, dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.ELU(inplace=True),
                nn.Conv2d(dims, dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.ELU(inplace=True),
                nn.Conv2d(dims, dims, kernel_size=kernel_size, stride=1, padding=kernel_size // 2),
                nn.ELU(inplace=True),
            )
        ]
        self.block_1 = nn.Sequential(*block_1)

    def forward(self, x):
        x_ = self.block_1(x)
        x = x + x_

        return x


class LH_Conv_E(nn.Module):
    def __init__(self, kernel_size=3, in_dims=2, out_dims=32):
        super(LH_Conv_E, self).__init__()
        self.conv = nn.Conv2d(in_dims, out_dims, kernel_size=1)
        block = [
            nn.Sequential(
                BasicBlock_E(kernel_size=kernel_size, dims=out_dims),
                BasicBlock_E(kernel_size=kernel_size, dims=out_dims),
                BasicBlock_E(kernel_size=kernel_size, dims=out_dims),
            )
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)

        return x


class LH_Conv_D(nn.Module):
    def __init__(self, kernel_size=3, in_dims=2, out_dims=32):
        super(LH_Conv_D, self).__init__()
        self.conv = nn.Conv2d(in_dims, out_dims, kernel_size=1)
        block = [
            nn.Sequential(
                BasicBlock_D(kernel_size=kernel_size, dims=out_dims)
            )
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        x = self.conv(x)
        x = self.block(x)

        return x


class Encoder1(nn.Module):
    def __init__(self):
        super(Encoder1, self).__init__()
        kernel_size = 5
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.mfb2 = MFB(64)
        self.mfb3 = MFB(128)
        self.bl_1 = LH_Conv_E(kernel_size=kernel_size, in_dims=2, out_dims=32)
        self.bl_2 = LH_Conv_E(kernel_size=kernel_size, in_dims=32, out_dims=64)
        self.bl_3 = LH_Conv_E(kernel_size=kernel_size, in_dims=64, out_dims=128)
        self.bl_4 = LH_Conv_E(kernel_size=kernel_size, in_dims=128, out_dims=256)
        self.chan_conv1 = Chan_Conv(kernel_size=1, dims=256)
        self.chan_conv2 = Chan_Conv(kernel_size=1, dims=128)
        self.chan_conv3 = Chan_Conv(kernel_size=1, dims=64)
        self.chan_conv4 = Chan_Conv(kernel_size=1, dims=32)
        # self.chan_conv1 = Chan_Conv(kernel_size=1, dims=240)
        # self.chan_conv2 = Chan_Conv(kernel_size=1, dims=120)
        # self.chan_conv3 = Chan_Conv(kernel_size=1, dims=60)
        # self.chan_conv4 = Chan_Conv(kernel_size=1, dims=30)

    def forward(self, x):
        s_conv = []

        # 卷积分支
        x = self.bl_1(x)
        x = self.chan_conv1(x)
        s_conv.append(x)
        x = self.max_pool(x)

        x = self.bl_2(x)
        x = self.chan_conv2(x)
        s_conv.append(x)
        x = self.max_pool(x)

        x = self.bl_3(x)
        x = self.chan_conv3(x)
        s_conv.append(x)
        x = self.max_pool(x)

        x = self.bl_4(x)
        x = self.chan_conv4(x)

        return x, s_conv


class Encoder2(nn.Module):
    def __init__(self):
        super(Encoder2, self).__init__()
        kernel_size = 3
        self.bl_2 = LH_Conv_E(kernel_size=kernel_size, in_dims=2, out_dims=64)
        self.bl_3 = LH_Conv_E(kernel_size=kernel_size, in_dims=64, out_dims=128)
        self.bl_4 = LH_Conv_E(kernel_size=kernel_size, in_dims=128, out_dims=256)
        self.chan_conv2 = Chan_Conv(kernel_size=1, dims=128)
        self.chan_conv3 = Chan_Conv(kernel_size=1, dims=64)
        self.chan_conv4 = Chan_Conv(kernel_size=1, dims=32)
        # self.chan_conv2 = Chan_Conv(kernel_size=1, dims=120)
        # self.chan_conv3 = Chan_Conv(kernel_size=1, dims=60)
        # self.chan_conv4 = Chan_Conv(kernel_size=1, dims=30)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.down_sample = nn.Sequential(*[Downsample_v2()])

    def forward(self, x):
        s_conv = []
        x =self.down_sample(x)

        # 卷积分支
        x = self.bl_2(x)
        x = self.chan_conv2(x)
        s_conv.append(x)
        x = self.max_pool(x)

        x = self.bl_3(x)
        x = self.chan_conv3(x)
        s_conv.append(x)
        x = self.max_pool(x)

        x = self.bl_4(x)
        x = self.chan_conv4(x)

        return x, s_conv


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        kernel_size = 3
        self.mfb1 = MFB5(128)
        self.mfb2 = MFB5(64)
        self.mfb3 = MFB1(32)
        self.bl_5 = LH_Conv_D(kernel_size=kernel_size, in_dims=256, out_dims=256)
        self.bl_6 = LH_Conv_D(kernel_size=kernel_size, in_dims=128, out_dims=128)
        self.bl_7 = LH_Conv_D(kernel_size=kernel_size, in_dims=64, out_dims=64)
        self.bl_8 = LH_Conv_D(kernel_size=kernel_size, in_dims=32, out_dims=32)
        self.chan_conv5 = Chan_Conv(kernel_size=1, dims=32)
        self.chan_conv6 = Chan_Conv(kernel_size=1, dims=64)
        self.chan_conv7 = Chan_Conv(kernel_size=1, dims=128)
        self.chan_conv8 = Chan_Conv(kernel_size=1, dims=256)
        # self.chan_conv5 = Chan_Conv(kernel_size=1, dims=30)
        # self.chan_conv6 = Chan_Conv(kernel_size=1, dims=60)
        # self.chan_conv7 = Chan_Conv(kernel_size=1, dims=120)
        # self.chan_conv8 = Chan_Conv(kernel_size=1, dims=240)
        self.conv = nn.Conv2d(32, 1, kernel_size=1, stride=1, padding=0)
        self.upscale_1 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.upscale_2 = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        self.upscale_3 = nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=2, stride=2)

    def forward(self, x, s1_conv, s2_conv):
        x = self.bl_5(x)
        x = self.chan_conv5(x)

        x = self.upscale_1(x)
        x = self.mfb1(x, s1_conv[-1], s2_conv[-1])
        x = self.bl_6(x)
        x = self.chan_conv6(x)

        x = self.upscale_2(x)
        x = self.mfb2(x, s1_conv[-2], s2_conv[-2])
        x = self.bl_7(x)
        x = self.chan_conv7(x)

        x = self.upscale_3(x)
        x = self.mfb3(x, s1_conv[-3])
        x = self.bl_8(x)
        x = self.chan_conv8(x)

        x = self.conv(x)

        return x



class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder1 = Encoder1()    # lr
        self.encoder2 = Encoder2()    # lrx2
        self.decoder = Decoder()
        self.mfb = MFB(256)


    def forward(self, lr, pd):
        lr = torch.cat((lr, pd), 1)
        x2, s2_conv= self.encoder2(lr)
        x, s1_conv = self.encoder1(lr)
        x = self.mfb(x, x2)
        x = self.decoder(x, s1_conv, s2_conv)

        return x
