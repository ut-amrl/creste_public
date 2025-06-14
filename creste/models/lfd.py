import os

import torch
from torch import nn
import torch.nn.functional as F

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf, open_dict

from creste.models.terrainnet import TerrainNet
from creste.models.blocks.vin import VIN

import creste.utils.train_utils as tu
from creste.utils.visualization import visualize_bev_policy, visualize_bev_label
from creste.datasets.coda_utils import FSC_LABEL_DIR

DEBUG_INPUTS = 0
DEBUG_MODE = 0


class MaxEntIRL(nn.Module):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone_cfg = model_cfg.vision_backbone
        self.traversability_head_cfg = model_cfg.traversability_head
        self.policy_cfg = self.model_cfg.get('policy_kwargs', {})
        self.ckpt_path = self.model_cfg.get('ckpt_path', '')
        self.weights_path = self.model_cfg.get('weights_path', '')
        self.map_size = self.model_cfg.get('map_size', [64, 128])
        self.policy_method = self.model_cfg.get('policy_method', 'fc')
        self.goal_cfg = self.model_cfg.get('goal_kwargs', {})
        self.action_horizon = self.model_cfg.get('action_horizon')
        self.solve_mdp = self.model_cfg.get('solve_mdp', False)
        self.zero_terminal_state = self.model_cfg.get('zero_terminal_state', False)

        self.register_buffer('dynamics', torch.tensor([
            [-1, -1],
            [-1, 0],
            [-1, 1],
            [0, -1],
            [0, 1],
            [1, -1],
            [1, 0],
            [1, 1]
        ], dtype=torch.long))
        fov_mask = tu.create_trapezoidal_fov_mask(
            # TODO: change this. Hard coded map size for now
            self.map_size[0]*2,
            self.map_size[1],
            70, 70, 0, 100
        ).view(1, 1, self.map_size[0]*2, self.map_size[1])
        self.fov_mask = fov_mask[:, :, :self.map_size[0], :self.map_size[1]]
        # self.register_buffer('fov_mask', fov_mask)
        # self.register_buffer('transition_probs', self.build_transition_probs(self.map_size))

        # Inverse transition probabilities from prev state to center state
        if self.policy_method == 'pp':
            self.register_buffer('transition_probs', torch.zeros(8, 1, 3, 3))
            left = [[1, 2], [2, 2], [2, 1], [0, 2],
                    [2, 0], [0, 1], [0, 0], [1, 0]]
            center = [[2, 2], [2, 1], [2, 0], [1, 2],
                      [1, 0], [0, 2], [0, 1], [0, 0]]
            right = [[2, 1], [2, 0], [1, 0], [2, 2],
                     [0, 0], [1, 2], [0, 2], [0, 1]]
            n_actions = self.traversability_head_cfg['net_kwargs']['qvalue_cfg']['dims'][-1]
            for i in range(n_actions):
                self.transition_probs[i, :, left[i][0], left[i][1]] = 0.0
                self.transition_probs[i, :, center[i][0], center[i][1]] = 1.0
                self.transition_probs[i, :, right[i][0], right[i][1]] = 0.0

        try:
            model_name = None
            if "TerrainNet" in self.backbone_cfg['project_name']:
                model_name = "TerrainNet"
            else:
                raise ValueError(
                    f"Model {self.backbone_cfg['project_name']} not found.")

            # Override backbone cfg to ensure that the RGB-D backbone is frozen
            with open_dict(self.backbone_cfg):
                if self.backbone_cfg['load_setting'] not in ['strict_freeze', 'strict_unfreezesplat']:
                    self.backbone_cfg['load_setting'] = 'strict_freeze'

            self.backbone = globals()[model_name](
                OmegaConf.create(self.backbone_cfg)
            )

            if os.path.exists(self.backbone_cfg['weights_path']):
                self.backbone.load_weights(self.backbone_cfg['weights_path'])
            
            self.traversability_head = globals()[self.traversability_head_cfg['value_iterator']](
                **self.traversability_head_cfg['net_kwargs']
            )

            if self.policy_method == 'fc':
                print("Using FC policy")
                q_value_dim = self.traversability_head_cfg['net_kwargs']['qvalue_cfg']['dims'][-1]
                self.fc = nn.Linear(in_features=q_value_dim,
                                    out_features=8, bias=False)
                self.sm = nn.Softmax(dim=1)
            elif self.policy_method == 'pp':
                print("Using Policy Propagation policy")
            else:
                raise ValueError(
                    f"Policy method {self.policy_method} not found.")

        except Exception as e:
            raise ValueError(f"Backbone {self.backbone['name']} not found.")

        self.freeze_backbone = self.model_cfg.get('freeze_backbone', True)
        self.freeze_head = self.model_cfg.get('freeze_head', False)
        self.load_strict = self.model_cfg.get('load_strict', True)
        if os.path.isfile(self.weights_path) and not os.path.isfile(self.ckpt_path):
            self.load_weights(self.weights_path)

    def _state_to_coord(self, state, vectorized=False):
        if vectorized:
            return torch.stack([state // self.map_size[1], state % self.map_size[1]], dim=1)
        return torch.tensor([state // self.map_size[1], state % self.map_size[1]], dtype=torch.long)

    def _coord_to_state(self, coord, vectorized=False):
        if vectorized:
            return coord[:, 0] * self.map_size[1] + coord[:, 1]
        return coord[0] * self.map_size[1] + coord[1]

    def load_weights(self, weights_path):
        """
        Loads weights from a checkpoint
        """
        print(f"Loading weights from {weights_path}")
        state_dict = torch.load(weights_path)['state_dict']
        # Filter out unnecessary model key prefix from training on pytorch lightning
        state_dict = {k.replace('model.', '', 1): v for k,
                      v in state_dict.items() if k.startswith('model.')}

        if self.freeze_backbone or self.freeze_head:
            print("Freezing weights with freezing (strict)")
            self.load_state_dict(state_dict, strict=self.load_strict)

            if self.freeze_backbone:
                print("Freezing backbone")
                self.backbone.eval()
                for param in self.backbone.parameters():
                    param.requires_grad = False
            
            if self.freeze_head:
                print("Freezing head")
                self.traversability_head.eval()
                for param in self.traversability_head.parameters():
                    param.requires_grad = False
        else:
            print("Load all weights without freezing (strict)")
            self.load_state_dict(state_dict, strict=self.load_strict)

    def expected_state_visitation_frequency(self, policy, expert):
        """
        Compute expected state visitation frequencies iteratively.
        Assumes a deterministic policy.
        Inputs:
            policy: (batch_sz, actions, height, width)
            expert: (batch_sz, T, 3, 3) Expert poses in BEV (original map size)
            fov_mask: (batch_sz, 1, height, width) Mask of the field of view
        Outputs:
            mu: (batch_sz, height, width) Expected state visitation frequencies
        """
        # Convert to one hot for long horizon exploration
        B, A, H, W = policy.shape
        ds_map = self.traversability_head_cfg['net_kwargs']['reward_cfg']['ds']

        S = (expert[:, :, :2, 2]//ds_map).long()  # [B, T, 2]
        S[:, :, 0] = torch.clamp(S[:, :, 0], 0, H-1)
        S[:, :, 1] = torch.clamp(S[:, :, 1], 0, W-1)

        # Start S in field of view
        S0 = tu.earliest_pose_in_fov(S, self.fov_mask)  # [B, 2]
        S1 = S[:, -1, :2]  # [B, 2]

        n_states = H*W
        policy = policy.view(B, A, n_states)  # [B, A, H, W] -> [B, A, H*W]
        S0 = S0[:, 0] * W + S0[:, 1]
        S1 = S1[:, 0] * W + S1[:, 1]

        mu = torch.zeros(B, self.action_horizon,
                         n_states).float().to(policy.device)
        mu[torch.arange(0, B), 0, S0] = 1.0

        # Compute expected state visitation frequencies
        policy_2d = policy.view(B, A, H, W)  # [B, A, H*W] -> [B, A, H, W]
        if self.policy_cfg['method'] == 'sharpen':
            logits = policy_2d - \
                policy_2d.max(dim=1, keepdim=True)[0]  # (B, A, H, W)
            logits = logits / self.policy_cfg['temperature']
            policy_2d = F.softmax(logits, dim=1)
        elif self.policy_cfg['method'] == 'none':
            pass
        else:
            raise ValueError(
                f'Policy method {self.policy_cfg["method"]} not found.')

        for t in range(1, self.action_horizon):
            if self.zero_terminal_state:
                mu[torch.arange(0, B), t-1, S1] = 0.0 # Set terminal states to 0. Important!!

            """ BEGIN CONV IMPLEMENTATION """
            prev_mu = mu[:, t-1].clone().view(B, 1, H, W)
            # [B, A, H, W] x [B, 1, H, W] -> [B, A, H, W]
            policy_mu = policy_2d * prev_mu
            new_mu = F.conv2d(
                policy_mu,
                self.transition_probs,
                stride=1,
                padding=1,
                groups=A
            )  # Depthwise convolution [B, A, H, W] x [A, 1, 3, 3] -> [B, A, H, W]
            # [B, A, H, W] -> [B, H, W]
            new_mu = new_mu.sum(dim=1, keepdim=True)
            """ END CONV IMPLEMENTATION """

            mu[:, t] = new_mu.view(B, H*W)
            # import cv2
            # cv2.imwrite("mu_test.png", mu[0, t].view(H, W).cpu().numpy()*255)
            # import pdb; pdb.set_trace()
            # start_coord = self._state_to_coord(S0, vectorized=True)
            # end_coord = self._state_to_coord(S1, vectorized=True)
            # cv2.imwrite("test_policy.png", visualize_bev_policy(policy_2d, start=start_coord, goal=end_coord))
        mu = mu.sum(dim=1).view(B, H, W)  # [B, T, H*W] -> [B, H, W]
        
        # Get the sequence of states from start over finite time horizon using policy
        with torch.no_grad():
            states_grid = torch.zeros(B, H, W).float().to(policy.device)
            states = torch.zeros(B, self.action_horizon,
                                 2).long().to(policy.device)
            states[:, 0] = self._state_to_coord(S0, vectorized=True)
            St = S0
            states_grid[torch.arange(
                0, B), states[:, 0, 0], states[:, 0, 1]] += 1

            most_likely_action = policy.argmax(dim=1)  # [B, H*W]
            for t in range(1, self.action_horizon):
                action = most_likely_action[torch.arange(0, B), St]
                coord = self._state_to_coord(St, vectorized=True)
                coord = coord + self.dynamics[action]
                coord[:, 0] = torch.clamp(coord[:, 0], 0, H-1)
                coord[:, 1] = torch.clamp(coord[:, 1], 0, W-1)
                states[:, t] = coord
                states_grid[torch.arange(0, B), coord[:, 0], coord[:, 1]] += 1
                St = self._coord_to_state(coord, vectorized=True)

            if DEBUG_MODE:
                import cv2
                print(
                    "Saving final expected state visitation frequencies to mu_test.png")
                mu_test = mu[0].detach().cpu().numpy()
                mu_test = mu_test.reshape(H, W)
                mu_test = cv2.normalize(
                    mu_test, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite("mu_test.png", mu_test)

            if DEBUG_MODE:
                import cv2
                import numpy as np
                print("Saving state sequence to states_test.png")
                states_test = states[0].cpu().numpy()
                states_test = states_test.reshape(self.action_horizon, 2)
                dummy_states_grid = np.zeros((H, W), dtype=np.uint8)
                dummy_states_grid[states_test[:, 0], states_test[:, 1]] = 255
                cv2.imwrite("states_test.png", dummy_states_grid)
            
        outputs = {
            'exp_svf': mu,
            'state_preds_grid': states_grid,
            'state_preds': states
        }
        # Assert that exp_svf is non-negative
        assert torch.all(mu >= 0), "Expected state visitation frequencies are negative"
        return outputs

    def iterative_policy_rollout(self, q, expert, T):
        """
        This iteratives rollsout the next states using the Q values
        Inputs:
            q: (batch_sz, l_q, map_size, map_size)
            expert: (batch_sz, T, 2) Expert poses in BEV (original map size)
            T: (int) number of steps to rollout
        Returns:
            policy: (batch_sz, map_size, map_size)
            visited_states: (batch_sz, T, 2)
        """
        B, l_q, H, W = q.shape

        state_preds = torch.zeros(B, T, 2).long().to(q.device)
        state_preds[:, 0] = expert[:, 0, :2].long()
        action_preds = torch.zeros(B, T, 8).float().to(q.device)
        for t in range(1, T):
            cx, cy = expert[:, t-1].long().chunk(2, dim=-1)
            q_out = q[torch.arange(B), :, cx[:, 0].long(),
                      cy[:, 0].long()].view(B, l_q)
            logits = self.fc(q_out)
            policy = self.sm(logits)
            with torch.no_grad():
                action = policy.argmax(dim=1)
                state_preds[:, t] = state_preds[:, t-1] + self.dynamics[action]
                state_preds[:, t, 0] = state_preds[:, t, 0].clamp(0, H-1)
                state_preds[:, t, 1] = state_preds[:, t, 1].clamp(0, W-1)

            action_preds[:, t] = policy

        return {
            'policy_fc': action_preds,
            'state_preds': state_preds
        }

    def forward(self, inputs):
        """
        Inputs:
            x: dictionary of inputs with image and p2p
        Outputs:
            Dictionary with outputs from the BEV feature map, traversability head, and qvalue head
        """
        image, p2p = inputs[0], inputs[1]
        device = image.device
        if len(inputs) > 2:
            expert = inputs[2]

        outputs = self.backbone((image, p2p))

        if not self.solve_mdp:
            outputs.update(self.traversability_head(outputs, None, self.solve_mdp))
            return outputs
        assert len(inputs) > 2, "Goal location required for MDP solver"
        
        # Reduce the feature map using max pool operation and crop the output
        B, _, H, W = outputs['bev_features'].shape

        # Save depth image for debugging
        # cv2.imwrite("test.png", cv2.normalize(outputs['depth_preds_metric'][0].cpu().numpy(), None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U))
        # import pdb; pdb.set_trace()
        map_ds = W // self.map_size[1]
        S = expert[:, :, :2, 2].long() // map_ds  # [B, T, 2]
        S[:, :, 0] = S[:, :, 0].clamp(0, self.map_size[0]-1)
        S[:, :, 1] = S[:, :, 1].clamp(0, self.map_size[1]-1)

        # Add goal to outputs for use by vin
        if "method" in self.goal_cfg:
            goal = torch.zeros(B, 1, H//2, W).float().to(device)
            if self.goal_cfg['method'] == 'gaussian':
                goal = self.gaussian_2d(S[:, -1], sigma=H/12, H=H//2, W=W)
            elif self.goal_cfg['method'] == 'dot':
                goal[torch.arange(B), :, S[:, -1, 0], S[:, -1, 1]] = 1
            outputs['goal'] = goal

        traverse_head_outputs = self.traversability_head(outputs, S, solve_mdp=self.solve_mdp)
        outputs.update(traverse_head_outputs)

        with torch.no_grad():  # No gradients for policy rollout
            if self.policy_method == 'fc':
                outputs.update(self.iterative_policy_rollout(
                    outputs['q_estimate'], S, self.action_horizon
                ))
            elif self.policy_method == 'pp':
                outputs.update(self.expected_state_visitation_frequency(
                    outputs['policy'], expert
                ))

        if DEBUG_INPUTS:
            with torch.no_grad():
                if len(traverse_head_outputs) == 0:
                    import pdb
                    pdb.set_trace()
                import cv2
                # Convert and save image to visualize
                image = inputs[0][0].cpu().numpy()  # [1, 4, H, W]
                rgb = image[0, :3].transpose(1, 2, 0)
                rgb = cv2.normalize(
                    rgb, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite("test.png", rgb)

                # Semantic
                sem = outputs['inpainting_sam_preds'] # [C, H, W]
                sem = torch.stack([sem, sem], dim=0)
                visualize_bev_label(FSC_LABEL_DIR, sem, "sem.png")

                # Elevation
                elev = outputs['elevation_preds'][0].squeeze(
                ).cpu().numpy()  # [1, H, W]
                elev = cv2.normalize(
                    elev[0], None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                cv2.imwrite("elev.png", elev)
            import pdb; pdb.set_trace()

        return outputs

