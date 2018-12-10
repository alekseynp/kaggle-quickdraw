import torch
from torch.autograd import Function


class PointsToImage(Function):
    @staticmethod
    def forward(ctx, i, v):
        device = i.device
        batch_size, _, num_input_points = i.size()
        feature_size = v.size()[2]

        batch_idx = torch.arange(batch_size, device=device).view(-1, 1).repeat(1, num_input_points).view(-1)
        idx_full = torch.cat([batch_idx.unsqueeze(0), i.permute(1, 0, 2).contiguous().view(2, -1)], dim=0)

        v_full = v.contiguous().view(batch_size * num_input_points, feature_size)
        mat_sparse = torch.cuda.sparse.FloatTensor(idx_full, v_full)
        mat_dense = mat_sparse.to_dense()

        ones_full = torch.ones(v_full.size(), device=device)
        mat_sparse_count = torch.sparse.FloatTensor(idx_full, ones_full)
        mat_dense_count = mat_sparse_count.to_dense()

        ctx.save_for_backward(idx_full, mat_dense_count)

        return mat_dense / torch.clamp(mat_dense_count, 1, 1e4)

    @staticmethod
    def backward(ctx, grad_output):
        idx_full, mat_dense_count = ctx.saved_tensors
        grad_i = grad_v = None

        batch_size, _, _, feature_size = grad_output.size()

        if ctx.needs_input_grad[0]:
            raise Exception("Indices aren't differentiable.")
        if ctx.needs_input_grad[1]:
            grad = grad_output[idx_full[0], idx_full[1], idx_full[2]]
            coef = mat_dense_count[idx_full[0], idx_full[1], idx_full[2]]
            grad_v = grad / coef
            grad_v = grad_v.view(batch_size, -1, feature_size)

        return grad_i, grad_v

points_to_image = PointsToImage.apply
