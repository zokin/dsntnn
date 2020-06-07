import torch
import numpy

from dsntnn import variance_reg_losses

if __name__ == "__main__":
    b, c, h, w = 5, 1, 256, 512
    batch = torch.rand(b, c, h, w)
    heatmaps = torch.nn.Softmax(dim=2)(
        batch.reshape(b, c, -1)
    ).reshape(b, c, h, w)
    var_loss = variance_reg_losses(heatmaps, 1.5)
    print(var_loss)