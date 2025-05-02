import torch
import torch.nn as nn


class RSABlock(nn.Module):
    def __init__(self, n_feats):
        super(RSABlock, self).__init__()
        self.att_c = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(n_feats, n_feats//2, 1),
            nn.ReLU(),
            nn.Conv1d(n_feats//2, n_feats, 1),
            nn.Sigmoid()
        )
        self.att_p = nn.Sequential(
            nn.Conv1d(n_feats, n_feats, 7, 1, 7 // 2),
            nn.ReLU(),
            nn.Conv1d(n_feats, n_feats, 9, stride=1, padding=(9 // 2) * 3, dilation=3),
            nn.ReLU(),
            nn.Conv1d(n_feats, n_feats, 1, 1),
            nn.Sigmoid()
        )
        self.conv = nn.Sequential(
            nn.Conv1d(n_feats, n_feats//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(n_feats//2),
            nn.ReLU(),
            nn.Conv1d(n_feats//2, n_feats//2, kernel_size=5, stride=1, padding=(5//2)*5, dilation=5),
            nn.BatchNorm1d(n_feats//2),
            nn.ReLU(),
            nn.Conv1d(n_feats//2, n_feats, kernel_size=7, stride=1, padding=(7//2)*7, dilation=7),
            nn.BatchNorm1d(n_feats),
            nn.ReLU()
        )

    def forward(self, x):
        x1 = self.att_c(x)
        x1 = x1 * x

        x2 = self.att_p(x1)
        x2 = x2 * x

        x3 = self.conv(x)

        return x2 + x3
    

class ResBlock(nn.Module):
    def __init__(self, n_feats):
        super(ResBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(n_feats, n_feats//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(n_feats//2),
            nn.ReLU(),
            nn.Conv1d(n_feats//2, n_feats//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(n_feats//2),
            nn.ReLU(),
            nn.Conv1d(n_feats//2, n_feats, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(n_feats),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.conv(x)

        return x


class Encoder(nn.Module):
    def __init__(self, c_list):
        super(Encoder, self).__init__()
        self.proj_first = nn.Sequential(
            nn.Conv1d(in_channels=3, out_channels=c_list[0], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c_list[0]),
            nn.ReLU()
        )

        self.down1 = nn.Sequential(
            nn.Conv1d(in_channels=c_list[0], out_channels=c_list[1], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(c_list[1]),
            nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv1d(in_channels=c_list[1], out_channels=c_list[2], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(c_list[2]),
            nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv1d(in_channels=c_list[2], out_channels=c_list[3], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(c_list[3]),
            nn.ReLU()
        )
        self.down4 = nn.Sequential(
            nn.Conv1d(in_channels=c_list[3], out_channels=c_list[4], kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(c_list[4]),
            nn.ReLU()
        )

        self.RA1 = RSABlock(c_list[1])
        self.RA2 = RSABlock(c_list[2])
        self.RA3 = RSABlock(c_list[3])
        self.RA4 = RSABlock(c_list[4])

    def forward(self, x):
        x = self.proj_first(x)    # c_num=16, 6000
        res0 = x

        x = self.down1(x)         # c_num=32, 3000
        x = self.RA1(x)
        res1 = x

        x = self.down2(x)         # c_num=64, 1500
        x = self.RA2(x)
        res2 = x

        x = self.down3(x)         # c_num=96, 750
        x = self.RA3(x)
        res3 = x

        x = self.down4(x)         # c_num=128, 375
        x = self.RA4(x)

        return x, res0, res1, res2, res3


class Decoder(nn.Module):
    def __init__(self, c_list):
        super(Decoder, self).__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=c_list[4], out_channels=c_list[3], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c_list[3]),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=c_list[3], out_channels=c_list[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c_list[2]),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=c_list[2], out_channels=c_list[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c_list[1]),
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=c_list[1], out_channels=c_list[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c_list[0]),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv1d(c_list[0], c_list[0]//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c_list[0]//2),
            nn.ReLU(),
            nn.Conv1d(c_list[0]//2, c_list[0]//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c_list[0]//4),
            nn.ReLU(),
            nn.Conv1d(c_list[0]//4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.RA5 = RSABlock(c_list[3])
        self.RA6 = RSABlock(c_list[2])
        self.RA7 = RSABlock(c_list[1])
        self.RA8 = RSABlock(c_list[0])

    def forward(self, x):     
        x = self.up1(x)           # c_num=96, 750
        x = self.RA5(x)

        x = self.up2(x)           # c_num=64, 1500
        x = self.RA6(x)

        x = self.up3(x)           # c_num=32, 3000
        x = self.RA7(x)

        x = self.up4(x)           # c_num=16, 6000
        x = self.RA8(x)

        x = self.conv(x)

        return x
    

class Decoder2(nn.Module):
    def __init__(self, c_list):
        super(Decoder2, self).__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=c_list[4], out_channels=c_list[3], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c_list[3]),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=c_list[3], out_channels=c_list[2], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c_list[2]),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=c_list[2], out_channels=c_list[1], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c_list[1]),
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose1d(in_channels=c_list[1], out_channels=c_list[0], kernel_size=4, stride=2, padding=1),
            nn.BatchNorm1d(c_list[0]),
            nn.ReLU()
        )

        self.conv = nn.Sequential(
            nn.Conv1d(c_list[0], c_list[0]//2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c_list[0]//2),
            nn.ReLU(),
            nn.Conv1d(c_list[0]//2, c_list[0]//4, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(c_list[0]//4),
            nn.ReLU(),
            nn.Conv1d(c_list[0]//4, 8, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(8),
            nn.ReLU(),
            nn.Conv1d(8, 1, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.RS5 = ResBlock(c_list[3])
        self.RS6 = ResBlock(c_list[2])
        self.RS7 = ResBlock(c_list[1])
        self.RS8 = ResBlock(c_list[0])

        self.RA5 = RSABlock(c_list[3])
        self.RA6 = RSABlock(c_list[2])
        self.RA7 = RSABlock(c_list[1])
        self.RA8 = RSABlock(c_list[0])

    def forward(self, x, res0, res1, res2, res3):     
        x = self.up1(x)           # c_num=96, 750
        x = self.RA5(x)
        s1 = self.RS5(res3)
        x = x + s1

        x = self.up2(x)           # c_num=64, 1500
        x = self.RA6(x)
        s2 = self.RS6(res2)
        x = x + s2

        x = self.up3(x)           # c_num=32, 3000
        x = self.RA7(x)
        s3 = self.RS7(res1)
        x = x + s3

        x = self.up4(x)           # c_num=16, 6000
        x = self.RA8(x)
        s4 = self.RS8(res0)
        x = x + s4

        x = self.conv(x)

        return x


class MAN(nn.Module):
    def __init__(self, c_list=[64, 96, 96, 128, 256]):
        super(MAN, self).__init__()
        self.encoder = Encoder(c_list)
        self.decoder_d = Decoder(c_list)
        self.decoder_p = Decoder2(c_list)
        self.decoder_s = Decoder2(c_list)

    def forward(self, x):
        x, res0, res1, res2, res3 = self.encoder(x)
        d = self.decoder_d(x)
        p = self.decoder_p(x, res0, res1, res2, res3)
        s = self.decoder_s(x, res0, res1, res2, res3)

        return d, p, s
