import torch
from torch import nn
from multiflow.models.utils import get_index_embedding, get_time_embedding


class NodeFeatureNet(nn.Module):

    def __init__(self, module_cfg):
        super(NodeFeatureNet, self).__init__()
        self._cfg = module_cfg
        self.c_s = self._cfg.c_s
        self.c_pos_emb = self._cfg.c_pos_emb
        self.c_timestep_emb = self._cfg.c_timestep_emb
        embed_size = self._cfg.c_pos_emb + self._cfg.c_timestep_emb * 2 + 1
        if self._cfg.embed_chain:
            embed_size += self._cfg.c_pos_emb
        if self._cfg.embed_aatype:
            self.aatype_embedding = nn.Embedding(21, self.c_s) # Always 21 because of 20 amino acids + 1 for unk
            embed_size += self.c_s + self._cfg.c_timestep_emb + self._cfg.aatype_pred_num_tokens
        if self._cfg.use_mlp:
            self.linear = nn.Sequential(
                nn.Linear(embed_size, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.ReLU(),
                nn.Linear(self.c_s, self.c_s),
                nn.LayerNorm(self.c_s),
            )
        else:
            self.linear = nn.Linear(embed_size, self.c_s)

    def embed_t(self, timesteps, mask):
        timestep_emb = get_time_embedding(
            timesteps[:, 0],
            self.c_timestep_emb,
            max_positions=2056
        )[:, None, :].repeat(1, mask.shape[1], 1)
        return timestep_emb * mask.unsqueeze(-1)

    def forward(
            self,
            *,
            so3_t,
            r3_t,
            cat_t,
            res_mask,
            diffuse_mask,
            chain_index,
            pos,
            aatypes,
            aatypes_sc,
        ):
        # s: [b]

        # [b, n_res, c_pos_emb]
        pos_emb = get_index_embedding(pos, self.c_pos_emb, max_len=2056)
        pos_emb = pos_emb * res_mask.unsqueeze(-1)

        # [b, n_res, c_timestep_emb]
        input_feats = [
            pos_emb,
            diffuse_mask[..., None],
            self.embed_t(so3_t, res_mask),
            self.embed_t(r3_t, res_mask)
        ]
        if self._cfg.embed_aatype:
            input_feats.append(self.aatype_embedding(aatypes))
            input_feats.append(self.embed_t(cat_t, res_mask))
            input_feats.append(aatypes_sc)
        if self._cfg.embed_chain:
            input_feats.append(
                get_index_embedding(
                    chain_index,
                    self.c_pos_emb,
                    max_len=100
                )
            )
        return self.linear(torch.cat(input_feats, dim=-1))
