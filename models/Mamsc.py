#MaMSC framework
import torch
import torch.nn as nn
from mamba_ssm import Mamba
from pytorch_wavelets import DWT1DForward, DWT1DInverse

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5):
        super(RMSNorm, self).__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class MamscEncoder(nn.Module):
    def __init__(self, configs, km_layers, norm_layer=None):
        super(MamscEncoder, self).__init__()
        self.km_layers = nn.ModuleList(km_layers)
        self.norm = norm_layer
        self.WT = DWT1DForward(wave='db4', J=1, mode='symmetric')
        self.IWT = DWT1DInverse(wave='db4')

    def forward(self, x):
        yl, yhs = self.WT(x)
        xl, xhs = yl, yhs[0]
        for km_layer in self.km_layers:
            x, xl, xhs = km_layer(x, xl, xhs)
        x_out = self.IWT((xl, [xhs])) + x
        if self.norm is not None:
            x_out = self.norm(x_out.transpose(1, 2)).transpose(1, 2)
        return x_out


class MamscBlock(nn.Module):
    def __init__(self, configs):
        super(MamscBlock, self).__init__()
        self.d_model = configs.d_model
        self.d_state = configs.d_state

        self.align_x1 = None
        self.align_x2 = None

        self.freq1 = Mamba(d_model=self.d_model, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        self.freq2 = Mamba(d_model=self.d_model, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        self.time1 = Mamba(d_model=self.d_model, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)
        self.time2 = Mamba(d_model=self.d_model, d_state=configs.d_state, d_conv=configs.d_conv, expand=configs.expand)

        self.norm = RMSNorm(self.d_model)
        self.norm1 = RMSNorm(self.d_model)

    def forward(self, x, x1, x2):
        B, D1, L = x1.shape
        B, D2, _ = x2.shape

        if self.align_x1 is None:
            self.align_x1 = nn.Linear(D1, self.d_model).to(x.device)
        if self.align_x2 is None:
            self.align_x2 = nn.Linear(D2, self.d_model).to(x.device)

        x1 = self.align_x1(x1.transpose(1, 2))
        x2 = self.align_x2(x2.transpose(1, 2))

        x_freq1 = self.freq1(x1)
        x_freq2 = self.freq2(x2)

        x = x.transpose(1, 2)
        x_time = self.time1(self.norm(x)) + self.time2(self.norm1(x).flip(dims=[1])).flip(dims=[1]) + x
        x_time = x_time.transpose(1, 2)

        return x_time, x_freq1.transpose(1, 2), x_freq2.transpose(1, 2)


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.configs = configs

        self.seq_len = configs.seq_len
        self.enc_in = configs.enc_in
        self.num_classes = configs.c_out
        self.d_model = configs.d_model

        self.embedding = nn.Conv1d(self.enc_in, self.d_model, kernel_size=1)

        self.encoder = MamscEncoder(
            configs,
            km_layers=[MamscBlock(configs) for _ in range(configs.e_layers)],
            norm_layer=nn.LayerNorm(self.d_model)
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(self.d_model, self.num_classes)
        )

        self.get_parameter_number()
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def get_parameter_number(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total} | Trainable: {trainable} | Memory: {total * 4 / 1024 / 1024:.2f} MB")

    def forward(self, x):
        assert not torch.isnan(x).any(), "Input has NaN in Model.forward()"
        assert not torch.isinf(x).any(), "Input has Inf in Model.forward()"
        assert x.shape[1] == self.seq_len, f"Expected seq_len {self.seq_len}, got {x.shape[1]}"
        assert x.shape[2] == self.enc_in, f"Expected enc_in {self.enc_in}, got {x.shape[2]}"
        x = x.permute(0, 2, 1)
        x = self.embedding(x)
        x = self.encoder(x)
        x = self.global_pool(x)
        out = self.classifier(x)

        return out