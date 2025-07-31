import torch
import torch.nn as nn
import torch.nn.functional as F

class nt_xent(nn.Module):
    def __init__(self, temperature):
        super(nt_xent, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        # Normalizing so cosine similarity works well
        z1 = nn.functional.normalize(z1, dim=1)
        z2 = nn.functional.normalize(z2, dim=1)

        batch_size = z1.shape[0]
        z = torch.cat([z1, z2], dim=0)

        # z[None, :, :] has shape (1, 2N, d)
        # z[:, None, :] has shape (2N, 1, d)
        # similarity_matrix has shape (2N, 2N)
        similarity_matrix = F.cosine_similarity(z[None, :, :], z[:, None, :], dim=-1)

        # Removing diagonal elements since they are the same
        eye = torch.eye(2 * batch_size, device=z.device).bool()
        y = similarity_matrix.clone()
        y[eye] = float('-inf')

        # [0, 1, 2, 3] -> [2, 3, 0, 1]
        targets = torch.arange(2 * batch_size, device=z.device)
        targets = (targets + batch_size) % (2 * batch_size)

        # Temperature
        logits = y / self.temperature

        return F.cross_entropy(logits, targets, reduction='mean')
