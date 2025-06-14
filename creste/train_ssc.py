
import os
import random
import numpy as np
from os.path import join

# Debugging
# os.environ['NCCL_DEBUG_SUBSYS']="INIT,P2P"
# os.environ['NCCL_DEBUG']="INFO"
# os.environ["TORCH_CPP_LOG_LEVEL"]="INFO"
# os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
# Logging utils
import torch.multiprocessing as mp
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate
import wandb

# ML utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy, ParallelStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.optim import Adam
torch.autograd.set_detect_anomaly(True)

# Custom modules
from creste.datasets.coda_utils import SEM_LABEL_REMAP_CLASS_NAMES, REMAP_SEM_ID_TO_COLOR, FSC_LABEL_DIR, SAM_LABEL_DIR, SSC_LABEL_DIR, SAM_DYNAMIC_LABEL_DIR
from creste.datasets.dataloader import CODaSSCModule
from creste.models.terrainnet import TerrainNet
from models.blocks.cnnmlp import MultiLayerPerceptron
import creste.utils.loss_utils as lu
import creste.utils.train_utils as tu
import creste.utils.tb_utils as tbu
from creste.utils.visualization import visualize_bev_label

def worker_init_fn(worker_id):
    torch.initial_seed()  # Re-seed each worker process to ensure diversity

class TerrainNetModel(pl.LightningModule):
    def __init__(self, 
        model_cfg: DictConfig
    ):
        super(TerrainNetModel, self).__init__()
        self.save_hyperparameters()
        self.model_cfg = model_cfg
        self.opt_cfg = model_cfg.optimizer
        self.lr_scheduler_cfg = model_cfg.lr_scheduler
        self.loss = lu.LossManager(
            model_cfg
        ) 
        self.model = TerrainNet(model_cfg)
        self.freeze_backbone_epochs = model_cfg.get('freeze_backbone_epochs', 0)
        self.backbone_frozen = False
        # For visualizing fdn features if predicted
        self.log_keys = model_cfg.get('log_keys', [])

        self.global_validation_step = 0

    def freeze_backbone(self):
        for param in self.model.depthcomp.parameters():
            param.requires_grad = False
        self.backbone_frozen = True

    def unfreeze_backbone(self):
        # TODO: Move unfreezing logic to model
        self.model.depthcomp.unfreeze_backbone()
        self.backbone_frozen = False

    def on_train_epoch_start(self):
        # Unfreeze the backbone after the specified number of epochs
        if self.current_epoch >= self.freeze_backbone_epochs and self.backbone_frozen:
            self.unfreeze_backbone()
            print(f"Unfreezing backbone at epoch {self.current_epoch}")
        elif self.current_epoch < self.freeze_backbone_epochs and not self.backbone_frozen:
            self.freeze_backbone()
            print(f"Freezing backbone at epoch {self.current_epoch}")

    def forward(self, x):
        return self.model(x)

    def _log_metrics(self, loss_dict, meta_data, batch_size, step=False, sync=False):
        for k, (w, v) in loss_dict.items():
            self.log(k, w*v.item(), on_step=step, on_epoch=True, prog_bar=False, sync_dist=sync, rank_zero_only=True, batch_size=batch_size)

        for k, v in meta_data.items():
            self.log(k, v.item(), on_step=step, on_epoch=True, prog_bar=False, sync_dist=sync, rank_zero_only=True, batch_size=batch_size)

    def training_step(self, inputs):
        # TODO: Modify this loop to go through multiple tasks
        batch, bidx, didx = inputs

        loss = 0.0
        loss_dict_full, meta_data_full = {}, {}
        for task, data in batch.items():
            # Forward pass
            image, p2p, mv_mask = data['image'], data['p2p'], data.get('immovable_depth_label', None)
            batch_size = image.shape[0]

            # Forward pass
            outputs = self.model( (image, p2p, mv_mask) )
            
            # Compute Losses
            with torch.no_grad():
                merged_dict = tu.merge_dict(('inputs', data), ('outputs', outputs))
                merged_dict["task"] = task
            
            loss_dict, meta_data = self.loss(merged_dict)
            
            # Merge loss and meta data for loggin
            with torch.no_grad():
                meta_data_full = tu.merge_loss_dict(
                    meta_data_full, meta_data
                )
                loss_dict_full = tu.merge_loss_dict(
                    loss_dict_full, loss_dict
                )
            loss += sum(w*v for w, v in loss_dict.values())

        # Log metrics
        loss_dict_full = tu.prefix_dict('train', loss_dict_full)
        meta_data_full = tu.prefix_dict('train', meta_data_full)
        self.log('train/loss', loss, on_step=True, prog_bar=True, rank_zero_only=True, sync_dist=True, batch_size=batch_size)
        self._log_metrics(loss_dict_full, meta_data_full, batch_size, step=True, sync=True)

        return {"loss": loss}

    def on_validation_start(self):
        self.reducer_feats = None
        self.reducer_labels = None

    def validation_step(self, inputs):
        batch, bidx, didx = inputs

        loss = 0.0
        loss_dict_full, meta_data_full = {}, {}
        for task, data in batch.items():
            # Forward pass
            image, p2p, mv_mask = data['image'], data['p2p'], data.get('immovable_depth_label', None)
            batch_size = image.shape[0]
  
            with torch.no_grad():
                outputs = self.model( (image, p2p, mv_mask) )
                merged_dict = tu.merge_dict(('inputs', data), ('outputs', outputs))
                merged_dict["task"] = task

                loss_dict, meta_data = self.loss(merged_dict)
            
                meta_data_full = tu.merge_loss_dict(
                    meta_data_full, meta_data
                )
                loss_dict_full = tu.merge_loss_dict(
                    loss_dict_full, loss_dict
                )
                loss += sum(w*v for w, v in loss_dict.values())

            # Prioritize inpainting preds for visualization over inpainting_sam_preds
            if "inpainting_preds" in outputs:
                preds = outputs['inpainting_preds']
            if "inpainting_sam_preds" in outputs:
                sam_preds = outputs['inpainting_sam_preds']
            if "inpainting_sam_dynamic_preds" in outputs:
                sam_dynamic_preds = outputs['inpainting_sam_dynamic_preds']
            
            if '3d_ssc_label' in data:
                gt_prob = data['3d_ssc_label'] / (torch.sum(
                    data['3d_ssc_label'], dim=1, keepdim=True
                ) + 1e-5) 
                gt_mode = torch.argmax(gt_prob, dim=1)
                mask = (gt_mode!=0) & data['fov_mask']
            else:
                mask = data['fov_mask']

            # Visualize feature and labels for every N batches
            if self.global_validation_step % 2 == 0:
                img_list = []
                dual_mask = torch.cat([mask, mask], axis=-1).unsqueeze(-1)[0].cpu().numpy()
                if "inpainting_sam_preds" in outputs:
                    feature_img = visualize_bev_label(
                        FSC_LABEL_DIR, 
                        torch.stack([sam_preds, sam_preds], axis=0)
                    ) * dual_mask
                    img_list.append(feature_img)
                if '3d_sam_label' in data:
                    sam_label = data['3d_sam_label'].squeeze(1)
                    sam_img = visualize_bev_label(
                        SAM_LABEL_DIR, 
                        torch.stack([sam_label, sam_label], axis=0)
                    ) * dual_mask
                    img_list.append(sam_img)
                if "inpainting_sam_dynamic_preds" in outputs:
                    if self.model_cfg['loss'][1]['name'] == "CrossEntropy":
                        sam_dynamic_preds = F.softmax(sam_dynamic_preds, dim=1)
                        sam_dynamic_preds = torch.argmax(sam_dynamic_preds, dim=1)
                        feature_img = visualize_bev_label(
                            SAM_DYNAMIC_LABEL_DIR, 
                            torch.stack([sam_dynamic_preds, sam_dynamic_preds], axis=0)
                        ) * dual_mask
                    else:
                        feature_img = visualize_bev_label(
                            FSC_LABEL_DIR, 
                            torch.stack([sam_dynamic_preds, sam_dynamic_preds], axis=0)
                        ) * dual_mask
                    img_list.append(feature_img)
                if '3d_sam_dynamic_label' in data:
                    sam_dynamic_label = data['3d_sam_dynamic_label'][:, 1, :, :] # [B, H, W]
                    sam_dynamic_img = visualize_bev_label(
                        SAM_DYNAMIC_LABEL_DIR, 
                        torch.stack([sam_dynamic_label, sam_dynamic_label], axis=0)
                    ) * dual_mask
                    img_list.append(sam_dynamic_img)
                if "inpainting_preds" in outputs:
                    preds = torch.argmax(preds, dim=1)
                    feature_img = visualize_bev_label(
                        SSC_LABEL_DIR, 
                        torch.stack([preds, preds], axis=0)
                    ) * dual_mask
                    feature_img = feature_img[..., ::-1]
                    img_list.append(feature_img)
                if '3d_ssc_label' in data:
                    label_img = visualize_bev_label(
                        SSC_LABEL_DIR, 
                        torch.stack([gt_mode, gt_mode], axis=0)
                    ) * dual_mask
                    label_img = label_img[..., ::-1]
                    img_list.append(label_img)

                combined_img = np.concatenate(img_list, axis=0)
                self.loggers[1].experiment.add_image(
                    'val/feature_img', 
                    combined_img.transpose(2, 0, 1),
                    self.global_validation_step
                )
                if "fimg_label" in data and len(self.log_keys) > 0:
                    fimg_label = data["fimg_label"]
                    tbu.log_feat_img_to_tb(
                        self.loggers[1], data, outputs, self.log_keys, self.current_epoch, self.global_validation_step, prefix='val'
                    )

        # Log metrics
        loss_dict_full = tu.prefix_dict('val', loss_dict_full)
        meta_data_full = tu.prefix_dict('val', meta_data_full)
        self.log('val/loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True, rank_zero_only=True, batch_size=batch_size)
        self._log_metrics(loss_dict_full, meta_data_full, batch_size, step=False, sync=True)

        self.global_validation_step+=1
        return {"loss": loss}

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
    # set_random_seed(1337, deterministic=True)

    is_zero_rank = os.environ.get('LOCAL_RANK', 0)==0 and os.environ.get('NODE_RANK', 0)==0

    # Set save path to original ckpt directory
    ckpt_path = cfg.model.get('ckpt_path', '')
    is_ckpt_valid = os.path.isfile(ckpt_path) and ckpt_path.endswith('.ckpt')
    tb_logdir = 'train'
    if not is_ckpt_valid:
        if is_zero_rank:
            CKPT_SAVE_DIR = tu.get_save_paths(cfg)
            os.environ['CKPT_SAVE_DIR'] = CKPT_SAVE_DIR
        else:
            CKPT_SAVE_DIR = os.environ['CKPT_SAVE_DIR']
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
    codatamodule.setup("fit")
    coda_train = codatamodule.train_dataloader()
    coda_val = codatamodule.val_dataloader()

    model = TerrainNetModel(cfg.model)

    # Load monitor metric
    monitor_metric_dict = cfg.model.get('monitor_metric', None)
    assert monitor_metric_dict is not None, "Monitor metric not found in mode config"

    monitor_metric = monitor_metric_dict.get('name', None)
    monitor_mode = monitor_metric_dict.get('mode', None)
    ddp_find_unused_parameters = True

    assert monitor_metric is not None, "Monitor metric not found in mode config"

    checkpoint_callback = ModelCheckpoint(
        monitor=monitor_metric,
        dirpath=CKPT_SAVE_DIR,
        filename=f'{cfg.model.optimizer.name}'+'-{epoch:02d}',
        auto_insert_metric_name=True, # Inserts the val acc to file
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
    if not is_ckpt_valid:
        ckpt_path=None
    else:
        print(f"Resuming from checkpoint: {ckpt_path}")
    #     model = TerrainNetModel.load_from_checkpoint(ckpt_path)
    #     print(f"Resuming from checkpoint: {ckpt_path}")

    strategy = DDPStrategy(find_unused_parameters=ddp_find_unused_parameters)
    pl.seed_everything(1337, workers=True)
    trainer = pl.Trainer(
        default_root_dir=cfg.trainer.default_root_dir,
        max_epochs=cfg.trainer.max_epochs, 
        accelerator=cfg.trainer.accelerator, 
        devices=cfg.trainer.devices, 
        # logger=[tensorboard_logger],
        logger=[wandb_logger, tensorboard_logger],
        callbacks=[lr_monitor, checkpoint_callback],
        log_every_n_steps=1,  # Log metrics every step
        strategy=strategy,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        deterministic=True, 
        inference_mode=False # Enables training in validation step
    )
    trainer.fit(model, coda_train, coda_val, ckpt_path=ckpt_path)

@hydra.main(version_base=None, config_path="../configs", config_name="ssc_sam")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()

