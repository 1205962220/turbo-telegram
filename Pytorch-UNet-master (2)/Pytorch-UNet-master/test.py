import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """
        self.width= in_channels
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes  # Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes

        self.scale = (1 / (in_channels * out_channels))
        self.con = nn.Conv2d(self.width, self.width, 1)
        self.weights1 = nn.Parameter \
            (self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2,dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)

        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        pre = x
        batchsize = x.shape[0]
        # Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)
        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, x.size(-2), x.size(-1) // 2 + 1, dtype=torch.cfloat,
                             device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        pre = self.con(pre)
        x = x + pre
        q = nn.PReLU()
        x = q(x)

        return x

class FNO2d(nn.Module):
    def __init__(self, in_clannes, out_clannes,  modes):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
                                                                                                                                
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes
        self.modes2 = modes
        self.in_clannes = in_clannes
        self.out_clannes = out_clannes
        # self.padding = 9 # pad the domain if input is non-periodic
        # self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        self.conv0 = SpectralConv2d(self.in_clannes, self.out_clannes, self.modes)
        self.conv1 = SpectralConv2d(self.in_clannes, self.out_clannes, self.modes)
        self.conv2 = SpectralConv2d(self.in_clannes, self.out_clannes, self.modes)
        self.conv3 = SpectralConv2d(self.in_clannes, self.out_clannes, self.modes)

        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):

        # grid = self.get_grid(x.shape, x.device)
        # x = torch.cat((x, grid), dim=-1)
        # x = self.fc0(x)/
        # x = x.permute(0, 3, 1, 2)

        # x = F.pad(x, [0 ,self.padding, 0 ,self.padding])

        x1 = self.conv0(x)
        # x2 = self.w0(x)
        x3 = x1 + x
        # x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x4 = x1 + x3
        # x = F.gelu(x)

        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.gelu(x)
        #
        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2

        # x = x[..., :-self.padding, :-self.padding]
        # x = x.permute(0, 2, 3, 1)
        # x = self.fc1(x)
        # x = F.gelu(x)
        # x = self.fc2(x)
        x4 = self.w1(x4)
        x5 = x4 + x
        return x5

# 、
#

class FIN(nn.Module):
    def __init__(self, modes1, modes2, width):
        super(FIN, self).__init__()
        self.width = width
        self.modes1 = modes1
        self.modes2 = modes2
        self.conv1 = nn.Conv2d(self.width, self.width, 1)
        self.conv2 = nn.Conv2d(self.width, self.width, 1)
        self.fno = FNO2d(self.modes1, self.modes2, self.width)
    def forward(self, x):
        res = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.fno(x)
        x = self.fno(x)
        x = self.fno(x)
        x = self.fno(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x + res
        return x





# class SPAF(nn.Module):
#     def __init__(self, modes1, modes2, width):
#         super(SPAF, self).__init__()
#         self.in_clannel = width
#         self.out_clannel = width
#         self.width = width
#         self.modes1 = modes1
#         self.modes2 = modes2
#         self.spafconv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.spafconv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
#         self.con = nn.Conv2d(self.in_clannel, self.out_clannel, 1)
#     def forward(self,x):
#         res1 = x
#         x = self.spafconv1(x)
#         x = res1 + x
#         res2 = x
#         x = self.spafconv2(x)
#         x = x + res2
#         x = self.con(x)
#         x = x + res1
#         return x

modes = 2
width = 3
s=torch.rand(6,3,64,64)
# model = FIN(modes, modes, width)
# model(s)

print(s[:,:,:62,:62].shape)