import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_lens import utils

from src.type_utils import DTYPES


class AutoEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d_hidden = cfg["dict_size"]
        l1_coeff = cfg["l1_coeff"]
        dtype = DTYPES[cfg["enc_dtype"]]
        torch.manual_seed(cfg["seed"])
        self.W_enc = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(cfg["act_size"], d_hidden, dtype=dtype)
            )
        )
        self.W_dec = nn.Parameter(
            torch.nn.init.kaiming_uniform_(
                torch.empty(d_hidden, cfg["act_size"], dtype=dtype)
            )
        )
        self.b_enc = nn.Parameter(torch.zeros(d_hidden, dtype=dtype))
        self.b_dec = nn.Parameter(torch.zeros(cfg["act_size"], dtype=dtype))

        self.W_dec.data[:] = self.W_dec / self.W_dec.norm(dim=-1, keepdim=True)

        self.d_hidden = d_hidden
        self.l1_coeff = l1_coeff
        self.dtype = dtype
        self.device = cfg["device"]

        self.version = 0
        self.to(cfg["device"])

    def forward(self, x, per_token=False):
        x_cent = x - self.b_dec
        acts = F.relu(x_cent @ self.W_enc + self.b_enc)  # [batch_size, d_hidden]
        x_reconstruct = acts @ self.W_dec + self.b_dec  # [batch_size, act_size]
        if per_token:
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1)  # [batch_size]
            l1_loss = self.l1_coeff * (acts.float().abs().sum(dim=-1))  # [batch_size]
            loss = l2_loss + l1_loss  # [batch_size]
        else:
            l2_loss = (x_reconstruct.float() - x.float()).pow(2).sum(-1).mean(0)  # []
            l1_loss = self.l1_coeff * (acts.float().abs().sum(dim=-1).mean(dim=0))  # []
            loss = l2_loss + l1_loss  # []
        return loss, x_reconstruct, acts, l2_loss, l1_loss

    @classmethod
    def load_from_hf(cls, version, hf_repo="ckkissane/tinystories-1M-SAES"):
        """
        Loads the saved autoencoder from HuggingFace.
        """

        cfg = utils.download_file_from_hf(hf_repo, f"{version}_cfg.json")
        self = cls(cfg=cfg)
        self.load_state_dict(
            utils.download_file_from_hf(hf_repo, f"{version}.pt", force_is_torch=True)
        )
        return self


auto_encoder_run = "gpt2-small_L9_Hcat_z_lr1.20e-03_l11.20e+00_ds24576_bs4096_dc1.00e-06_rsanthropic_rie25000_nr4_v9"

encoder = AutoEncoder.load_from_hf(
    auto_encoder_run, hf_repo="ckkissane/attn-saes-gpt2-small-all-layers"
)

FEATURE_I = 18
FEATURE_BIAS = encoder.b_enc[FEATURE_I]
