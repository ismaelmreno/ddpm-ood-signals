 #####################################################################################
 # MIT License                                                                       #
 #                                                                                   #
 # Copyright (C) 2018 Sungwon Kim                                                    #
 #                                                                                   #
 #   Permission is hereby granted, free of charge, to any person obtaining a copy    #
 #   of this software and associated documentation files (the "Software"), to deal   #
 #   in the Software without restriction, including without limitation the rights    #
 #   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell       #
 #   copies of the Software, and to permit persons to whom the Software is           #
 #   furnished to do so, subject to the following conditions:                        #
 #                                                                                   #
 #   The above copyright notice and this permission notice shall be included in all  #
 #   copies or substantial portions of the Software.                                 #
 #                                                                                   #
 #   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR      #
 #   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,        #
 #   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE     #
 #   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER          #
 #   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,   #
 #   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE   #
 #   SOFTWARE.                                                                       #
 #####################################################################################

from clarinet.modules import Conv, ResBlock
from clarinet.loss import sample_from_gaussian

import torch
from torch import nn


class Wavenet(nn.Module):
    def __init__(self, out_channels=1, num_blocks=3, num_layers=10,
                 residual_channels=512, gate_channels=512, skip_channels=512,
                 kernel_size=2, cin_channels=128,
                 upsample_scales=None, causal=True):
        super(Wavenet, self). __init__()

        self.causal = causal
        self.num_blocks = num_blocks
        self.num_layers = num_layers
        self.out_channels = out_channels
        self.gate_channels = gate_channels
        self.residual_channels = residual_channels
        self.skip_channels = skip_channels
        self.cin_channels = cin_channels
        self.kernel_size = kernel_size

        self.front_channels = 32
        self.front_conv = nn.Sequential(
            Conv(1, self.residual_channels, self.front_channels, causal=self.causal),
            nn.ReLU()
        )

        self.res_blocks = nn.ModuleList()
        for b in range(self.num_blocks):
            for n in range(self.num_layers):
                self.res_blocks.append(ResBlock(self.residual_channels, self.gate_channels, self.skip_channels,
                                                self.kernel_size, dilation=self.kernel_size**n,
                                                cin_channels=self.cin_channels, local_conditioning=True,
                                                causal=self.causal, mode='SAME'))

        self.final_conv = nn.Sequential(
            nn.ReLU(),
            Conv(self.skip_channels, self.skip_channels, 1, causal=self.causal),
            nn.ReLU(),
            Conv(self.skip_channels, self.out_channels, 1, causal=self.causal)
        )

        self.upsample_conv = nn.ModuleList()
        for s in upsample_scales:
            convt = nn.ConvTranspose2d(1, 1, (3, 2 * s), padding=(1, s // 2), stride=(1, s))
            convt = nn.utils.weight_norm(convt)
            nn.init.kaiming_normal_(convt.weight)
            self.upsample_conv.append(convt)
            self.upsample_conv.append(nn.LeakyReLU(0.4))

    def forward(self, x, c):
        c = self.upsample(c)
        out = self.wavenet(x, c)
        return out

    def generate(self, num_samples, c=None):
        # Only a waveform generation
        x = torch.zeros(1, 1, num_samples + 1)
        c = self.upsample(c)
        for i in range(num_samples):
            if i % 100 == 0:
                print(i)
            if i >= self.receptive_field_size():
                start_idx = i - self.receptive_field_size() + 1
            else:
                start_idx = 0
            x_in = x[:, :, start_idx:i+1].to(torch.device("cuda"))
            if c is not None:
                cond = c[:, :, start_idx:i + 1]
            else:
                cond = None
            out = self.wavenet(x_in, cond)
            # sampling input
            x[:, :, i + 1] = sample_from_gaussian(out[:, :, -1:]).to(torch.device("cpu"))
            del out, x_in, cond
        return x[:, :, 1:]

    def upsample(self, c):
        if self.upsample_conv is not None:
            # B x 1 x C x T'
            c = c.unsqueeze(1)
            for f in self.upsample_conv:
                c = f(c)
            # B x C x T
            c = c.squeeze(1)
        return c

    def wavenet(self, tensor, c=None):
        h = self.front_conv(tensor)
        skip = 0
        for i, f in enumerate(self.res_blocks):
            h, s = f(h, c)
            skip += s
        out = self.final_conv(skip)
        return out

    def receptive_field_size(self):
        num_dir = 1 if self.causal else 2
        dilations = [2 ** (i % self.num_layers) for i in range(self.num_layers * self.num_blocks)]
        return num_dir * (self.kernel_size - 1) * sum(dilations) + self.front_channels
