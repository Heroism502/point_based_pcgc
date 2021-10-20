import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import knn_points

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def ball_knn(radius, K, xyz, padding=True):
    B, N, C = xyz.shape
    with torch.no_grad():
        dist, idx, _ = knn_points(xyz, xyz, K=K, return_sorted=False)
        if padding:
            m = torch.arange(N, device=xyz.device, requires_grad=False).repeat(B,1)
            xyz_first = m.unsqueeze(2).repeat([1, 1, K])
            mask = dist > radius * radius
            idx[mask] = xyz_first[mask]
            dist[mask] = 0
            return idx, dist
        else:
            mask = dist <= radius * radius
            return idx, dist, mask

def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points


class FeatureEmbeddingModule(nn.Module):
    def __init__(self, K, radius, input_size, mlp, shortcut=True):

        super(FeatureEmbeddingModule, self).__init__()
        self.input_size = input_size
        self.shortcut = shortcut
        self.K = K
        self.radius = radius
        self.mlp_layers = nn.ModuleList()
        self.dynamic_filter_generator = nn.Sequential(
            nn.Linear(7, input_size),
            nn.ReLU(inplace=True),
            nn.Linear(input_size, input_size)
        )
        if shortcut:
            self.shortcut_layer = nn.Linear(input_size-7, mlp[-1])
        last_channel = input_size + 3
        for output_channel in mlp:
            self.mlp_layers.append(nn.Linear(last_channel, output_channel))
            last_channel = output_channel

    def forward(self, xyz, points):
        B, N, _ = xyz.size()
        residual = points
        idx, dist = ball_knn(self.radius, self.K, xyz)
        knn_xyz = index_points(xyz, idx)
        relative_knn_xyz = knn_xyz - xyz.unsqueeze(2)
        grouped_xyz = torch.cat([knn_xyz, relative_knn_xyz, dist.unsqueeze(3)], dim=-1)
        grouped_points = torch.cat([grouped_xyz, index_points(points, idx)], dim=-1)
        kernel = self.dynamic_filter_generator(grouped_xyz)
        grouped_points = torch.mean(kernel * grouped_points, dim=2)
        grouped_points = torch.cat([xyz, grouped_points], dim=-1)

        for i, layer in enumerate(self.mlp_layers):
            grouped_points = F.relu(layer(grouped_points)) \
                if i != len(self.mlp_layers)-1 else layer(grouped_points)
        if self.shortcut:
            residual = self.shortcut_layer(residual)
            return F.relu(residual + grouped_points)
        else:
            return F.relu(grouped_points)


class SampleLayer(nn.Module):
    def __init__(self, npoints, input_dim):
        super(SampleLayer, self).__init__()
        self.selector = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, npoints))

    def forward(self, xyz, points,):
        weights = F.softmax(self.selector(torch.cat([xyz, points], dim=-1)).transpose(1, 2), dim=-1)
        return torch.matmul(weights, xyz), torch.matmul(weights, points)


class NGS(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, first_layer=False):
        super(NGS, self).__init__()
        self.first_layer = first_layer
        if first_layer:
            self.fa1 = FeatureEmbeddingModule(nsample, radius, in_channel, mlp, shortcut=False)
        else:
            self.fa1 = FeatureEmbeddingModule(nsample, radius, in_channel, mlp, shortcut=True)
        self.fa2 = FeatureEmbeddingModule(nsample, radius, mlp[-1]+7, [mlp[-1] for i in range(len(mlp))])
        self.sample = SampleLayer(npoint, mlp[-1]+3)

    def forward(self, xyz, points, global_step=None):
        if self.first_layer:
            points = xyz
        new_points = self.fa1(xyz, points)
        new_points = self.fa2(xyz, new_points)
        sampled_xyz, sampled_points = self.sample(xyz, new_points)
        return sampled_xyz, sampled_points


class PointNetSetAbstraction(nn.Module):
    def __init__(self, in_channel, mlp):
        super(PointNetSetAbstraction, self).__init__()
        self.mlp_convs = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            last_channel = out_channel

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        new_xyz, new_points = sample_and_group_all(xyz, points)
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            new_points = F.relu(conv(new_points))
        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points


class ReconstructionLayer(nn.Module):
    def __init__(self, ratio, input_channel, output_channel):
        super(ReconstructionLayer, self).__init__()
        self.deconv_features = nn.ConvTranspose1d(input_channel, output_channel, ratio, stride=ratio)


    def forward(self, x):
        feature = self.deconv_features(x.permute(0, 2, 1)).permute(0, 2, 1)
        return feature