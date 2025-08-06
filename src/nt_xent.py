import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/Lightning-Universe/lightning-bolts/blob/master/src/pl_bolts/losses/self_supervised_learning.py
class nt_xent(nn.Module):
    def __init__(self, temperature):
        super(nt_xent, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        out = torch.cat([z1, z2], dim=0)
        n_samples = len(out)

        cov = torch.mm(out, out.t().contiguous())
        sim = torch.exp(cov / self.temperature)

        # Negative similarity
        mask = ~torch.eye(n_samples, device=sim.device).bool()
        neg = sim.masked_select(mask).view(n_samples, -1).sum(dim=-1)

        # Positive similarity
        pos = torch.exp(torch.sum(z1 * z2, dim=-1) / self.temperature)
        pos = torch.cat([pos, pos], dim=0)

        return -torch.log(pos / neg).mean()
