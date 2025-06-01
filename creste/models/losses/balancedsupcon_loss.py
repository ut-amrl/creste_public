import numpy as np
import torch
from torch import nn
# from einops import rearrange

class UnagiLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self):
        raise NotImplementedError

class UnagiContrastiveLoss(UnagiLoss):
    def __init__(self, views):
        super().__init__()
        self.views = views

    def combine_views(self, *views):
        all_views = [view for view in views]
        return torch.stack(all_views, dim=1)

    def forward(self, *args):
        raise NotImplementedError

def weighted_logsumexp(mat, axis, weights):
    _max, _ = torch.max(mat, dim=axis, keepdim=True)
    lse = ((torch.exp(mat - _max) * weights).sum(dim=axis, keepdim=True)).log() + _max

    return lse.squeeze(axis)


class BalContrastiveLoss(UnagiContrastiveLoss):
    def __init__(
        self,
        views,
        type="l_spread",  # sup_con, sim_clr, l_attract, l_spread
        temp=0.5,
        pos_in_denom=False,  # as per dan, false by default
        log_first=True,  # TODO (ASN): should this be true (false originally)
        a_lc=1.0,
        a_spread=1.0,
        lc_norm=False,
        use_labels=True,
        clip_pos=1.0,
        pos_in_denom_weight=1.0,
    ):
        super().__init__(views)
        self.temp = temp
        self.log_first = log_first
        self.a_lc = a_lc
        self.a_spread = a_spread
        self.pos_in_denom = pos_in_denom
        self.lc_norm = lc_norm
        self.use_labels = use_labels
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.clip_pos = clip_pos
        self.pos_in_denom_weight = pos_in_denom_weight

        if type == "sup_con":
            print(f"Using {type} contrastive loss function")
            self.a_spread = 0
            # self.pos_in_denom = False  # this isn't doing anything
        # elif type == "l_attract":
        #     print(f"Using {type} contrastive loss function")
        #     self.a_spread = 0
        #     self.pos_in_denom = False  # working
        elif type == "l_repel":
            print(f"Using {type} contrastive loss function")
            self.a_spread = 1
            self.a_lc = 0
        elif type == "sim_clr":
            print(f"Using {type} contrastive loss function")
            self.a_spread = 0
            self.a_lc = 1
            self.use_labels = False

    def forward(self, feats, labels):
        """
        Inputs:
            feats: (B, V, F) where B is the batch size, V is the number of views and F is the feature dimension
            labels: (B) where B is the batch size
        """
        B, V, F = feats.size()
        assert V==self.views, f"Expected {self.views} views, got {V} views"

        #1 Compute f(xi) and f(a(xi))        
        anchor_feats = feats[:, 0, :]
        if self.views==1:
            augment_feats = anchor_feats
        else:
            augment_feats = torch.cat(torch.unbind(feats[:, 1:, :], dim=1), dim=0)

        labels = labels.view(B, 1)
        logits = torch.div(torch.matmul(anchor_feats, anchor_feats.t()), self.temp)   # B x B

        if logits.dim() < 2 or logits.size(0) == 1:
            print("Returning 0 loss on balsupcon")
            return 0 # If there is only one sample in the batch, return 0 loss

        # For numerical stability
        try:
            logits_max, _   = torch.max(logits, dim=1, keepdim=True)
        except Exception as e:
            print("Exception in balancedsupcon_loss.py, ", e)
            return torch.tensor(0, dtype=logits.dtype, device=logits.device)
        logits          = logits - logits_max.detach()
        exp_logits      = torch.exp(logits) # BV x BV
        
        #2 Compute masks
        posmask = torch.eq(labels, labels.T).bool().to(feats.device)   # B x B
        notselfmask = ~torch.eye(B).bool().to(feats.device)   # B x B

        posmask = posmask & notselfmask
        negmask = ~posmask & notselfmask

        #3 Compute lsup
        o_xixminus = torch.sum(exp_logits * negmask, dim=1, keepdim=True) # BV x 1
        denom = (exp_logits + o_xixminus) # B x B
        log_prob = logits - torch.log(denom) # B x B
        mask_pos_pairs = posmask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs) # Prevent divide by zero error
        mean_log_prob_pos = (log_prob * posmask).sum(1) / mask_pos_pairs
        lsup = -mean_log_prob_pos.view(B, 1).mean()

        #4 Compute lspread
        auglogits = torch.div(torch.matmul(anchor_feats, augment_feats.t()), self.temp)   # B x B*(V-1)
        auglogits_max, _   = torch.max(auglogits, dim=1, keepdim=True)
        auglogits          = auglogits - auglogits_max.detach()

        num_cols = B*(V-1)
        augmask    = torch.zeros((B, num_cols)).bool().to(feats.device)
        index = torch.arange(num_cols).repeat(B, 1)
        mask = ((index // (V-1) ) % B) == torch.arange(B).unsqueeze(1)
        augmask[mask] = 1

        log_prob = (auglogits - torch.logsumexp(logits*posmask, dim=1, keepdim=True)) * augmask
        lspread = -log_prob.sum(1).view(B, 1).mean()

        #5 Compute loss
        assert self.a_lc + self.a_spread != 0
        loss = (self.a_lc * lsup + self.a_spread * lspread) / (self.a_lc + self.a_spread)
     
        return loss
    
if __name__ == "__main__":
    views = 3
    batch_size = 5
    feature_dim = 32
    temp = 0.1
    model = BalContrastiveLoss(views, temp=temp)
    feats = torch.rand(batch_size, views, feature_dim)
    labels = torch.tensor([1, 0, 0, 1, 1]).view(-1, 1) # Adjust this manually
    loss = model(feats, labels)
    
    print("Inputs matrix shape", feats.shape)
    print("Labels matrix\n", labels)
    print(f'Balanced contrastive loss: {loss}')
