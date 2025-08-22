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
        batch_size = z1.shape[0]

        z = torch.cat([z1, z2], dim=0)
        sim = torch.matmul(z, z.T) / self.temperature

        diag_mask = torch.eye(2 * batch_size, dtype=torch.bool, device=sim.device)
        sim_masked = sim.masked_fill(diag_mask, float('-inf'))

        pos_idx = torch.arange(batch_size, device=sim.device)
        pos = torch.cat([sim[pos_idx, pos_idx + batch_size], sim[pos_idx + batch_size, pos_idx]], dim=0)

        denom = torch.logsumexp(sim_masked, dim=1)

        loss = - (pos - denom).mean()

        return loss

