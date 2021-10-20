import torch.nn as nn
import torch
import torch.nn.functional as F
from models.utils import *
from pytorch3d.loss import chamfer_distance
from models.hyper_encoder import HyperEncoder
from models.hyper_decoder import HyperDecoder
from models.bitEstimator import BitEstimator
import math



class get_model(nn.Module):
    def __init__(self, normal_channel=False, bottleneck_size=512, use_hyperprior=True, recon_points=2560):
        super(get_model, self).__init__()
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.bottleneck_size = bottleneck_size
        self.normal_channel = normal_channel
        self.sa1 = NGS(npoint=recon_points//4, radius=0.2, nsample=32, in_channel=7, mlp=[64, 128], first_layer=True)
        self.sa2 = NGS(npoint=recon_points//16, radius=0.4, nsample=32, in_channel=128 + 7, mlp=[128, 256])
        self.sa3 = PointNetSetAbstraction(in_channel=256 + 3, mlp=[512, bottleneck_size])
        self.use_hyper = use_hyperprior
        self.recon_points = recon_points
        self.decompression1 = ReconstructionLayer(recon_points//16, bottleneck_size, 256)
        self.decompression2 = ReconstructionLayer(4, 256, 256)
        self.decompression3 = ReconstructionLayer(4, 256, 256)
        self.coor_reconstruction_layer = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 3)
        )
        if use_hyperprior:
            self.he = HyperEncoder(bottleneck_size)
            self.hd = HyperDecoder(bottleneck_size//32)
            self.bitEstimator_z = BitEstimator(bottleneck_size//32)
        else:
            self.bitEstimator = BitEstimator(bottleneck_size)

    def forward(self, xyz, global_step=None):
        # Set Abstraction layers
        B, C, N = xyz.shape
        xyz = xyz.permute(0, 2, 1)
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz

        pc_gd = l0_xyz
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points, global_step)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points, global_step)
        l3_xyz, l3_points = self.sa3(l2_xyz.transpose(1, 2), l2_points.transpose(1, 2))

        x = l3_points.view(B, self.bottleneck_size)

        if self.use_hyper:
            z = self.he(x)
            z_noise = torch.nn.init.uniform_(torch.zeros_like(z), -0.5, 0.5)
            if self.training:
                compressed_z = z + z_noise
            else:
                compressed_z = torch.round(z)
        x_noise = torch.nn.init.uniform_(torch.zeros_like(x), -0.5, 0.5)
        if self.training:
            compressed_x = x + x_noise
        else:
            compressed_x = torch.round(x)

        # decoder
        if self.use_hyper:
            recon_sigma = self.hd(compressed_z)
        lf1 = self.decompression1(compressed_x.unsqueeze(1))
        lf2 = self.decompression2(lf1)
        lf3 = self.decompression3(lf2)
        coor_recon = self.coor_reconstruction_layer(lf3)


        def feature_probs_based_sigma(feature, sigma):
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-10, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, probs

        def feature_probs_based_sigma_test(feature, sigma):
            mu = torch.zeros_like(sigma)
            sigma = sigma.clamp(1e-10, 1e10)
            gaussian = torch.distributions.laplace.Laplace(mu, sigma)
            probs = gaussian.cdf(feature + 0.5) - gaussian.cdf(feature - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(probs + 1e-10) / math.log(2.0), 0, 50), dim=1)
            return total_bits, probs

        def iclr18_estimate_bits_z(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50))
            return total_bits, prob

        def iclr18_estimate_bits_z_test(z):
            prob = self.bitEstimator_z(z + 0.5) - self.bitEstimator_z(z - 0.5)
            total_bits = torch.sum(torch.clamp(-1.0 * torch.log(prob + 1e-10) / math.log(2.0), 0, 50), dim=1)
            return total_bits, prob

        # if not self.training:
        #     total_bits_feature, _ = feature_probs_based_sigma_test(compressed_x, recon_sigma)
        #     total_bits_z, _ = iclr18_estimate_bits_z_test(compressed_z)
        #     bpp = (total_bits_z + total_bits_feature) / N
        #     return bpp, coor3
        total_bits_feature, _ = feature_probs_based_sigma(compressed_x, recon_sigma)
        total_bits_z, _ = iclr18_estimate_bits_z(compressed_z)
        bpp = (total_bits_z+total_bits_feature)/(B*N)
        cd = chamfer_distance(pc_gd, coor_recon)[0]
        return bpp, coor_recon, cd


class get_loss(nn.Module):
    def __init__(self, lam=1):
        super(get_loss, self).__init__()
        self.lam = lam

    def forward(self, bpp, cd_loss):
        return self.lam*cd_loss + bpp, cd_loss, bpp
