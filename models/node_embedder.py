"""Neural network for embedding node features."""
import torch
from torch import nn
from models.utils import get_index_embedding, get_time_embedding


class NodeEmbedder(nn.Module):

    def __init__(self, module_cfg):
        super(NodeEmbedder, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        self.linear = nn.Linear(
            self._cfg.c_pos_emb + self._cfg.c_timestep_emb, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)
    
    def embed_t_2(self, timesteps, mask, motif_mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)

        timesteps_mask = torch.ones_like(timesteps)

        timestep_mask_emb = get_time_embedding(
            timesteps_mask[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        timestep_emb = timestep_emb * mask.unsqueeze(-1)
        timestep_mask_emb = timestep_mask_emb * mask.unsqueeze(-1)
        timestep_emb[motif_mask == 0] = timestep_mask_emb[motif_mask == 0]
        return timestep_emb

    def forward(self, timesteps, mask, motif_mask = None):
        # s: [b]

        b, num_res, device = mask.shape[0], mask.shape[1], mask.device

        # [b, n_res, c_pos_emb]
        pos = torch.arange(num_res, dtype=torch.float32).to(device)[None]
        pos_emb = get_index_embedding(
            pos, self.c_pos_emb, max_len=2056
        )
        pos_emb = pos_emb.repeat([b, 1, 1])
        pos_emb = pos_emb * mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [pos_emb]
        # timesteps are between 0 and 1. Convert to integers.
        #input_feats.append(self.embed_t(timesteps, mask))
        input_feats.append(self.embed_t_2(timesteps, mask, motif_mask))
        return self.linear(torch.cat(input_feats, dim=-1))
