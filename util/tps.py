import torch
import itertools
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def grid_sample(input, grid, canvas = None):
    output = F.grid_sample(input, grid)
    if canvas is None:
        return output
    else:
        input_mask = Variable(input.data.new(input.size()).fill_(1))
        output_mask = F.grid_sample(input_mask, grid)
        padded_output = output * output_mask + canvas * (1 - output_mask)
        return padded_output

class TPSGridGen(nn.Module):
    def __init__(self, target_shape=None, target_control_points=None, device='cpu'):
        super(TPSGridGen, self).__init__()
        self.target_shape = target_shape
        self.device = device

        assert target_control_points.ndimension() == 2
        self.ndim = target_control_points.size(1)
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float().to(self.device)
        self.register_buffer('target_control_points', target_control_points)

        forward_kernel = torch.zeros(N + 1 + self.ndim, N + 1 + self.ndim, device=self.device)
        target_control_partial_repr = self.compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, N].fill_(1)
        forward_kernel[N, :N].fill_(1)
        forward_kernel[:N, N+1:].copy_(target_control_points)
        forward_kernel[N+1:, :N].copy_(target_control_points.transpose(0, 1))

        inverse_kernel = torch.inverse(forward_kernel)

        HW = target_shape.numel()
        Y, X = torch.meshgrid(*[torch.linspace(-1, 1, s, device=self.device) for s in target_shape])
        target_coordinate = torch.stack([X.flatten(), Y.flatten()], dim=1)
        target_coordinate_partial_repr = self.compute_partial_repr(target_coordinate, target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr,
            torch.ones(HW, 1, device=self.device),
            target_coordinate
        ], dim=1)

        self.register_buffer('target_coordinate', target_coordinate)
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(self.ndim + 1, self.ndim, device=self.device))
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

    def forward(self, source_control_points):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == self.ndim
        batch_size = source_control_points.size(0)

        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, -1, -1))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        new_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return new_coordinate

    def compute_partial_repr(self, input_points, control_points):
        N = input_points.size(0)
        M = control_points.size(0)
        pairwise_diff = input_points.view(N, 1, self.ndim) - control_points.view(1, M, self.ndim)
        pairwise_dist = (pairwise_diff * pairwise_diff).sum(-1)
        repr_matrix = 0.5 * pairwise_dist * pairwise_dist.log()
        mask = repr_matrix != repr_matrix
        repr_matrix.masked_fill_(mask, 0)
        return repr_matrix

    def tps_mesh(self, source_control_points=None, max_range=(0.1, ), batch_size=1):
        if source_control_points is None:
            source_control_points = self.target_control_points.expand(batch_size, -1, -1)
            source_control_points = source_control_points + source_control_points.new(source_control_points.shape).uniform_(-1, 1).to(self.device) * source_control_points.new(max_range).to(self.device)
        source_coordinate = self.forward(source_control_points)
        return source_coordinate

    def tps_trans(self, inputs, canvas=0.5, random_offset=None, point_mask=None):
        batch_size = inputs.shape[0]
        target_height, target_width = self.target_shape

        source_control_points = self.target_control_points.unsqueeze(0).expand(batch_size, -1, -1)

        if point_mask is not None:
            source_control_points = source_control_points * point_mask.to(self.device)

        if random_offset is not None:
            source_control_points = source_control_points + random_offset.to(self.device)

        source_coordinate = self.forward(source_control_points)
        grid = source_coordinate.view(batch_size, target_height, target_width, 2)

        if isinstance(canvas, float):
            canvas = torch.FloatTensor(batch_size, inputs.shape[1], target_height, target_width).fill_(canvas).to(self.device)

        target_image = grid_sample(inputs, grid, canvas)
        return target_image, source_control_points
