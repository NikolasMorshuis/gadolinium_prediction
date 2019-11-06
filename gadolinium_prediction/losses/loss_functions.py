import torch
import torch.nn as nn

class Aleatoric_Loss(nn.Module):
    """
    x[0]: Output image
    x[1]: ln(Sigma^2)
    """
    def __init__(self):
        super(Aleatoric_Loss, self).__init__()
        self.device=torch.device("cuda:0")

    def aleatoric_loss(self, x, y, s, mask):
        D = torch.sum(mask)
        if D != 0:
            L2 = 1 / D * torch.sum(torch.exp(-s) * torch.abs(x - y) + s)
        else:
            L2 = None
        return L2

    def forward(self, x, y, mask):
        sigma = x[:, 1:2]
        x = x[:, 0:1]
        loss = self.aleatoric_loss(x[mask == 1], y[mask == 1], sigma[mask == 1], mask=mask[mask == 1])
        return loss

class Aleatoric_Loss_gauss(nn.Module):
    """
    x[0]: Output image
    x[1]: ln(Sigma^2)
    """
    def __init__(self):
        super(Aleatoric_Loss_gauss, self).__init__()
        self.device=torch.device("cuda:0")

    def aleatoric_loss(self, x, y, s, mask):
        D = torch.sum(mask)
        if D != 0:
            L2 = 1 / D * torch.sum(0.5*torch.exp(-s) * (x - y)**2 + 0.5*s)
        else:
            L2 = None
        return L2

    def forward(self, x, y, mask):
        sigma = x[:, 1:2]
        x = x[:, 0:1]
        loss = self.aleatoric_loss(x[mask == 1], y[mask == 1], sigma[mask == 1], mask=mask[mask == 1])
        return loss

