import os
from os.path import join

import cv2
import numpy as np

import torch
from torch.optim import Adam
import torch.nn.functional as F
from torch import autograd

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy, ParallelStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

import hydra
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf

from creste.models.lfd import MaxEntIRL
import creste.utils.train_utils as tu
import creste.utils.loss_utils as lu
import creste.utils.visualization as vi
from creste.utils.utils import make_labels_contiguous_vectorized
from creste.datasets.coda_utils import TRAVERSE_LABEL_DIR, FSC_LABEL_DIR, TASK_TO_LABEL, SAM_LABEL_DIR, SSC_LABEL_DIR

from creste.utils.loss_utils import MaxEntIRLLoss
from creste.datasets.dataloader import CODaSSCModule

DEBUG_TRAIN = False


class MaxEntIRLModel(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig):
        super(MaxEntIRLModel, self).__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.automatic_optimization = False
        # Batch size = batch_size * num_gpus
        self.batch_size = int(model_cfg.batch_size)

        self.opt_cfg = model_cfg.optimizer
        self.lr_scheduler_cfg = model_cfg.lr_scheduler
        self.loss = lu.LossManager(
            model_cfg
        )
        self.model = MaxEntIRL(model_cfg)
        # For visualizing fdn features if predicted
        self.log_keys = model_cfg.get('log_keys', [])

        self.global_validation_step = 0

    def forward(self, x):
        return self.model(x)

    def _log_metrics(self, loss_dict, meta_data, batch_size, step=False, sync=False):
        for k, (w, v) in loss_dict.items():
            self.log(k, w*v.item(), on_step=step, on_epoch=True, prog_bar=False,
                     sync_dist=sync, rank_zero_only=True, batch_size=batch_size)

        for k, v in meta_data.items():
            self.log(k, v.item(), on_step=step, on_epoch=True, prog_bar=False,
                     sync_dist=sync, rank_zero_only=True, batch_size=batch_size)

    def training_step(self, inputs):
        batch, bidx, didx = inputs

        loss = 0.0
        loss_dict_full, meta_data_full = {}, {}
        for task, data in batch.items():
            image, p2p = data['image'], data['p2p']
            expert = data['traversability_label']

            opt = self.optimizers()
            opt.zero_grad()
            outputs = self.model((image, p2p, expert))

            with torch.no_grad():
                merged_dict = tu.merge_dict(
                    ('inputs', data), ('outputs', outputs))
                merged_dict['task'] = task
            loss_dict, meta_data = self.loss(merged_dict)

            with torch.no_grad():
                meta_data_full = tu.merge_loss_dict(
                    meta_data_full, meta_data
                )
                loss_dict_full = tu.merge_loss_dict(
                    loss_dict_full, loss_dict
                )

            loss += sum(w*v for w, v in loss_dict.values())

            self.manual_backward(loss)
            opt.step()

        loss_dict_full = tu.prefix_dict('train', loss_dict_full)
        meta_data_full = tu.prefix_dict('train', meta_data_full)
        self.log('train/loss', loss, on_step=True, prog_bar=True,
                 rank_zero_only=True, sync_dist=True, batch_size=self.batch_size)
        self._log_metrics(loss_dict_full, meta_data_full,
                          self.batch_size, step=True, sync=True)

        return {"loss": loss}

    def on_after_backward(self):
        grad_norm = self.compute_gradient_norm()
        self.log('train/grad_norm', grad_norm)

    def compute_gradient_norm(self):
        total_norm = 0
        for p in self.parameters():
            if p.requires_grad and p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        return total_norm

    def validation_step(self, inputs):
        batch, bidx, didx = inputs
        loss = 0.0
        loss_dict_full, meta_data_full = {}, {}
        for task, data in batch.items():
            image, p2p = data['image'], data['p2p']
            expert = data['traversability_label']
            outputs = self.model((image, p2p, expert))

            if DEBUG_TRAIN:
                with torch.no_grad():
                    if "inpainting_sam_preds" in outputs:
                        image = data['image'][0, 0, :3].cpu(
                        ).numpy().transpose(1, 2, 0)
                        image = (image - image.min()) / \
                            (image.max() - image.min())
                        image = (image * 255).astype(np.uint8)
                        cv2.imwrite("image.png", image)
                        mask = data['fov_mask']
                        dual_mask = torch.cat(
                            [mask, mask], axis=-1).unsqueeze(-1)[0].cpu().numpy()
                        sam_preds = outputs['inpainting_sam_preds']
                        # Visualize backbone outputs
                        feature_img = vi.visualize_bev_label(
                            FSC_LABEL_DIR,
                            torch.stack([sam_preds, sam_preds], axis=0),
                            batch_idx=0
                        ) * dual_mask
                        cv2.imwrite("sam_preds.png", feature_img)

            with torch.no_grad():
                merged_dict = tu.merge_dict(
                    ('inputs', data), ('outputs', outputs))
                merged_dict['task'] = task

                # Save visualization outputs
                self.log_img_outputs(data, outputs)
            self.global_validation_step += 1

            loss_dict, meta_data = self.loss(merged_dict)
            loss = sum(w*v for w, v in loss_dict.values())

            loss_dict = tu.prefix_dict('val', loss_dict)
            meta_data = tu.prefix_dict('val', meta_data)
            self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True,
                     sync_dist=True, rank_zero_only=True, batch_size=self.batch_size)
            self._log_metrics(loss_dict, meta_data,
                              self.batch_size, step=False, sync=True)

        return {"loss": loss}

    def log_img_outputs(self, inputs, outputs):
        label_type = None
        if '3d_ssc_label' in inputs:
            # Visualize costmap and predicted paths versus gt
            gt_prob = inputs[TASK_TO_LABEL[SSC_LABEL_DIR]] / (torch.sum(
                inputs[TASK_TO_LABEL[SSC_LABEL_DIR]], dim=1, keepdim=True
            ) + 1e-5)
            gt_mode = torch.argmax(gt_prob, dim=1, keepdim=True)
            label_type = SSC_LABEL_DIR
        else:
            gt_mode = inputs[TASK_TO_LABEL[SAM_LABEL_DIR]]
            label_type = SAM_LABEL_DIR

        cf_lab_key = self.loss.losses[0].lab_key.split('/')[-1]
        poses = None
        if "counterfactuals" in cf_lab_key:
            cf = inputs[cf_lab_key]
            poses, ranks = self.loss.losses[0].preprocess_counterfactuals(
                cf, self.device)

        mask = (gt_mode != 0) & inputs['fov_mask'].unsqueeze(1)

        B, _, H, W = outputs['traversability_preds'].shape
        Ho, Wo = outputs['bev_features'].shape[-2:]
        ds = Wo//W

        ssc_input = tu.resize_and_crop(
            gt_mode.byte(), (Ho//ds, Wo//ds), (0, H, 0, W))  # [B, C, H, W]
        mask = tu.resize_and_crop(
            mask.byte(), (Ho//ds, Wo//ds), (0, H, 0, W)
        ).bool()  # [B, 1, H, W]

        for i in range(0, B):
            reward_ssc_input = torch.stack(
                [outputs['traversability_preds'][i], ssc_input[i]], dim=0
            )  # [C+1, H, W]

            reward_img = vi.visualize_bev_label(
                TRAVERSE_LABEL_DIR, reward_ssc_input, kwargs={
                    'fov_mask': mask[i].squeeze(), 'label_type': label_type}
            )
            
            if self.model_cfg.solve_mdp:
                expert = inputs['traversability_label']
                # Visualize poses on reward image
                gt_bev_poses = expert.detach().clone()
                gt_bev_poses[:, :, :2, 2] = gt_bev_poses[:, :, :2, 2]//ds
                Tpred = outputs['state_preds'].shape[1]
                pred_bev_poses = torch.zeros((B, Tpred, 3, 3)).to(
                    gt_bev_poses.device)
                pred_bev_poses[:, :, :2,
                            2] = outputs['state_preds'].detach().clone()
                gt_action_img = vi.visualize_bev_poses(
                    gt_bev_poses, batch_idx=i, img=reward_img[:, W:, :])
                pred_action_img = vi.visualize_bev_poses(
                    pred_bev_poses, batch_idx=i, img=reward_img[:, :W, :])
            else:
                opt_bev_poses = poses[i][ranks[i]==0]
                subopt_bev_poses = poses[i][ranks[i]>0]
                No = opt_bev_poses.shape[0]
                Ns = subopt_bev_poses.shape[0]

                # Flip x and y order of poses for visualization
                gt_action_img = reward_img[:, W:, :].copy() 
                pred_action_img = reward_img[:, :W, :].copy()
                img_list = [pred_action_img, gt_action_img]
                img_color = [(0, 255, 0), (0, 0, 255)]

                for img_idx, img in enumerate(img_list):
                    for j in range(poses[i].shape[0]):
                        rank = ranks[i][j]
                        color = img_color[rank]
                        img = vi.visualize_bev_poses(
                            poses[i], batch_idx=j, img=img, 
                            color=color,
                            thickness=1,
                            indexing='xy'
                        )

            combined_img = np.concatenate(
                [pred_action_img, gt_action_img], axis=1)
            self.loggers[0].experiment.add_image(
                'val/traversability_preds',
                combined_img.transpose(2, 0, 1),
                self.global_validation_step
            )

            # Only add policy if solving MDP
            if self.model_cfg.solve_mdp:
                # Log comparison between ground truth and predicted visitation freq and value estimate
                # [B, 1, H, W] -> [1, H, W]
                value_estimate = outputs['value_estimate'][i][0]
                value_estimate_img = value_estimate.detach().cpu().numpy()
                value_estimate_img = cv2.normalize(
                    value_estimate_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                combined_img = value_estimate_img

                if self.model_cfg.policy_method == 'pp':
                    xy, svf = MaxEntIRLLoss.compute_expert_visitation(
                        expert,
                        ds,
                        (H, W)
                    )  # [B, H, W]
                    svf_img = svf[i].detach().cpu().numpy()
                    svf_img = cv2.normalize(
                        svf_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    combined_img = np.concatenate(
                        [combined_img, svf_img], axis=1)

                    exp_svf_img = outputs['exp_svf'][i].detach(
                    ).cpu().numpy()
                    exp_svf_img = cv2.normalize(
                        exp_svf_img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    combined_img = np.concatenate(
                        [combined_img, exp_svf_img], axis=1)

                combined_img = cv2.cvtColor(
                    combined_img, cv2.COLOR_GRAY2RGB).astype(np.uint8)
                self.loggers[0].experiment.add_image(
                    'val/value_svf_predsvf',
                    combined_img.transpose(2, 0, 1),
                    self.global_validation_step
                )

                # Save policy visualization
                if self.model_cfg.policy_method == 'pp' and 'policy' in outputs:
                    S = gt_bev_poses[:, :, :2, 2].long()  # [B, T, 2]
                    S[:, :, 0] = S[:, :, 0].clamp(0, H-1)
                    S[:, :, 1] = S[:, :, 1].clamp(0, W-1)
                    # start = tu.earliest_pose_in_fov(
                    #     S, mask[i:i+1])
                    policy_img = vi.visualize_bev_policy(
                        outputs['policy'], batch_idx=i, start=S[:,
                                                                0, :], goal=S[:, -1, :]
                    )
                    policy_img = cv2.resize(policy_img, (600, 600))
                    self.loggers[0].experiment.add_image(
                        'val/policy',
                        policy_img.transpose(2, 0, 1),
                        self.global_validation_step
                    )

    def configure_optimizers(self):
        if self.opt_cfg.name == 'Adam':
            optimizer = Adam(
                filter(lambda p: p.requires_grad, self.model.parameters()),
                betas=(self.opt_cfg.beta1, self.opt_cfg.beta2),
                lr=self.opt_cfg.lr
            )
        else:
            raise NotImplementedError

        if self.lr_scheduler_cfg.name == 'ExponentialLR':
            lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
                optimizer, gamma=self.lr_scheduler_cfg.gamma
            )
        else:
            raise NotImplementedError

        return [optimizer], [lr_scheduler]


def train(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

    is_zero_rank = os.environ.get(
        'LOCAL_RANK', 0) == 0 and os.environ.get('NODE_RANK', 0) == 0

    ckpt_path = cfg.model.get('ckpt_path', '')
    is_ckpt_valid = os.path.isfile(ckpt_path) and ckpt_path.endswith('.ckpt')
    if not is_ckpt_valid:
        if is_zero_rank:
            CKPT_SAVE_DIR = tu.get_save_paths(cfg, model_type='lfd_maxentirl')
            os.environ['CKPT_SAVE_DIR'] = CKPT_SAVE_DIR
        else:
            CKPT_SAVE_DIR = os.environ['CKPT_SAVE_DIR']
        tb_logdir = 'train'
    else:
        CKPT_SAVE_DIR = os.path.dirname(ckpt_path)
        tb_logdir = 'train'
    RUN_NAME = CKPT_SAVE_DIR.split('/')[-3]

    codatamodule = CODaSSCModule(
        OmegaConf.to_object(cfg.dataset),
        views=cfg.model.views,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.trainer.num_workers,
    )
    # codatamodule.setup("fit")
    # coda_train = codatamodule.train_dataloader()
    # coda_val = codatamodule.val_dataloader()
    model = MaxEntIRLModel(cfg.model)

    # Load monitor metric
    monitor_metric_dict = cfg.model.get('monitor_metric', None)
    assert monitor_metric_dict is not None, "Monitor metric not found in loss list"
    
    monitor_metric = monitor_metric_dict.get('name', None)
    monitor_mode = monitor_metric_dict.get('mode', None)
    ddp_find_unused_parameters = True

    assert monitor_metric is not None, "Monitor metric not found in loss list"

    # Load training callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=CKPT_SAVE_DIR,
        filename=f'{cfg.model.optimizer.name}'+'-{epoch:02d}',
        auto_insert_metric_name=True,  # Inserts the val acc to file
        save_top_k=5,
        mode=monitor_mode
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    wandb_logger = WandbLogger(
        project=cfg.model.project_name,
        name=RUN_NAME
    )

    tb_dir = join(CKPT_SAVE_DIR, 'tb')
    os.makedirs(tb_dir, exist_ok=True)
    tensorboard_logger = TensorBoardLogger(
        save_dir=tb_dir, name=tb_logdir
    )

    # Resume from checkpoint if specified
    if is_ckpt_valid:
        model = MaxEntIRLModel.load_from_checkpoint(ckpt_path)
        print(f"Resuming from checkpoint: {ckpt_path}")

    strategy = DDPStrategy(find_unused_parameters=ddp_find_unused_parameters)
    pl.seed_everything(1337, workers=True)
    trainer = pl.Trainer(
        default_root_dir=cfg.trainer.default_root_dir,
        max_epochs=cfg.trainer.max_epochs,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        logger=[tensorboard_logger],
        # logger=[tensorboard_logger, wandb_logger],
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=1,  # Log metrics every step
        strategy=strategy,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        deterministic=True,
        inference_mode=False  # Enables training in validation step
    )
    trainer.fit(model, datamodule=codatamodule)

@hydra.main(version_base=None, config_path="../configs", config_name="traversability")
def main(cfg: DictConfig):
    train(cfg)


if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
