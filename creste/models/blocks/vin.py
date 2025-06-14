import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

from creste.models.blocks.conv import MultiScaleFCN
from omegaconf import DictConfig, OmegaConf

from torchvision.transforms import GaussianBlur
from creste.utils.visualization import visualize_bev_policy

"""
Code adapted from: https://github.com/kentsommer/pytorch-value-iteration-networks
Original paper by NIPS 2016: Value Iteration Networks: https://arxiv.org/abs/1602.02867
"""
DEBUG_VIN = 0
DEBUG_GOAL_ONLY_FEATURE = 0


class VIN(nn.Module):
    def __init__(self, reward_cfg, qvalue_cfg):
        super(VIN, self).__init__()
        self.reward_cfg = reward_cfg
        self.qvalue_cfg = qvalue_cfg
        self.discount = qvalue_cfg.get('discount', 0.95)

        self.r = globals()[self.reward_cfg['name']](
            OmegaConf.create(self.reward_cfg['net_kwargs'])
        )

        assert len(
            self.qvalue_cfg.kernels) == 1, "Only single layer Q value network supported"

        # Initialize 8 connected grid of actions
        self.register_buffer('w', torch.zeros(
            self.qvalue_cfg.dims[1], 1, 3, 3))
        left = [[1, 0], [0, 0], [0, 1], [2, 0], [0, 2], [2, 1], [2, 2], [1, 2]]
        center = [[0, 0], [0, 1], [0, 2], [1, 0],
                  [1, 2], [2, 0], [2, 1], [2, 2]]
        right = [[0, 1], [0, 2], [1, 2], [0, 0],
                 [2, 2], [1, 0], [2, 0], [2, 1]]
        for i in range(self.qvalue_cfg.dims[1]):
            self.w[i, 0, left[i][0], left[i][1]] = 0.1
            self.w[i, 0, center[i][0], center[i][1]] = 0.8
            self.w[i, 0, right[i][0], right[i][1]] = 0.1

    def value_iteration_manual(self, r, goal, threshold=0.001, discount=0.95):
        """
        Inputs:
            r (B, 1, H, W): Reward map
            goal (B, 2): Goal location
            threshold: Convergence threshold
            discount: Discount factor
        """
        B, _, H, W = r.shape
        v = torch.zeros_like(r).to(r.device)
        delta = torch.inf

        def eval_q(r, v):
            return F.conv2d(
                r + v * discount,
                self.w,
                stride=1,
                padding=1
            )  # [B, A, H, W]

        while delta > threshold:
            # Terminal states are 0 value
            old_v = v.clone()
            q = eval_q(r, old_v)
            new_v = q.max(dim=1, keepdim=True)[0]  # [B, 1, H, W]
            delta = (new_v - old_v).abs().max().item()
            v = new_v

        q = eval_q(r, v)
        policy = q - q.max(dim=1, keepdim=True)[0]
        exps = torch.exp(policy)
        policy = exps / exps.sum(dim=1, keepdim=True)
        return v, policy, q

    def set_goal_location(self, bev_map, goal_x, goal_y):
        """ Sets 3x3 grid around each goal in batch to 1
        Inputs:
        bev_map: (batch_sz, 1, map_size, map_size)
        goal_x, goal_y: (batch_sz,)
        """
        B, _, H, W = bev_map.shape
        for b in range(B):
            bev_map[b, :, goal_x[b]-1:goal_x[b]+2, goal_y[b]-1:goal_y[b]+2] = 1

        return bev_map

    def forward(self, feat_map, S, solve_mdp=False):
        """
        Inputs:
        :param input_view: (batch_sz, l_r, map_size, map_size)
        :param S: (batch_sz, seq_len, 2), goal location
        :param solve_mdp: Whether to solve MDP or not
        :return: dictionary with
            logits and softmaxed logits
            reward map
        """
        input_view = None
        for key in self.reward_cfg.input_keys:
            input_view = torch.cat(
                [input_view, feat_map[key]], dim=1) if input_view is not None else feat_map[key]
        assert input_view is not None, "Input view is None"

        Ho, Wo = input_view.shape[-2:]
        # Max pool and crop output to reduce feature map size
        input_view = F.max_pool2d(
            input_view, kernel_size=self.reward_cfg.ds, stride=self.reward_cfg.ds)  # [B, C, H, W]
        B, C, H, W = input_view.shape
        input_view = input_view[:, :, :H//2, :]  # [B, C, H//2, W]
        input_view = input_view.detach()
        input_view.requires_grad_(True)

        r = self.r(input_view)  # Reward map

        with torch.no_grad():
            # Upsample reward map to original size
            full_r = torch.zeros(B, 1, Ho, Wo).to(r.device)
            full_r[:, :, :Ho//2, :] = F.interpolate(
                r, size=(Ho//2, Wo), mode='bilinear', align_corners=False)

        outputs = {
            self.reward_cfg["output_prefix"][0]: r,
            f'{self.reward_cfg["output_prefix"][0]}_full': full_r,
            'input_view': input_view
        }
        if not solve_mdp:
            return outputs
        assert S is not None, "No expert demonstrations given but solve mdp is True"

        with torch.no_grad():  # Don't compute gradients for value iteration
            v, policy, q = self.value_iteration_manual(
                r, S[:, -1, :], threshold=0.001, discount=self.discount)
            outputs.update({
                'policy': policy,
                'q_estimate': q,
                'value_estimate': v,
            })

        if DEBUG_VIN:
            with torch.no_grad():
                import cv2
                norm_v = (v - v.min()) / (v.max() - v.min())
                cv2.imwrite("test_v.png", norm_v[0, 0].cpu().numpy()*255)

                # Visualize policy
                cv2.imwrite("test_policy.png", visualize_bev_policy(
                    policy, batch_idx=0, start=S[:, 0, :], goal=S[:, -1, :]))

        return outputs
