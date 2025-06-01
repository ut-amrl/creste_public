import os
from os.path import join

os.environ['TMPDIR'] = '/tmp'

# Pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.distributed as dist
import torch.multiprocessing as mp

# Pytorch Lightning imports
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
from pytorch_lightning.strategies import DDPStrategy, SingleDeviceStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

# Custom imports
from creste.models.stereodepth import MSNet2D
from creste.models.distillation import DistillationBackbone
from creste.models.foundation import FoundationBackbone
import creste.utils.loss_utils as lu
from datasets.dataloader import CODaPEFreeModule
import creste.utils.train_utils as tu
import creste.utils.tb_utils as tbu

# Hydra Imports 
from omegaconf import DictConfig, OmegaConf
import hydra
from hydra.utils import instantiate

class DistillationModel(pl.LightningModule):
    def __init__(self, model_cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()

        self.model_cfg = model_cfg
        self.opt_cfg = model_cfg.optimizer 
        self.lr_scheduler_cfg = model_cfg.lr_scheduler
        try:
            self.model = None
            self.model = globals()[model_cfg.vision_backbone['class_name']](model_cfg)
        except KeyError:
            raise NotImplementedError(f"Model {model_cfg.vision_backbone['class_name']} not found")
        self.log_keys = model_cfg.get('log_keys', [])

        self.val_step_count = 0
        self.test_step_count = 0

        # self.loss = build_loss(model_cfg.criterion)
        self.loss = lu.LossManager(
            model_cfg
        ) 

        # Important: This property activates manual optimization.
        # self.automatic_optimization = True

    def forward(self, x):
        return self.model(x)

    def _log_metrics(self, loss_dict, meta_data, step=False, sync=False):
        for k, (w, v) in loss_dict.items():
            self.log(k, w*v.item(), on_step=step, on_epoch=True, prog_bar=False, sync_dist=sync, rank_zero_only=True)

        for k, v in meta_data.items():
            self.log(k, v.item(), on_step=step, on_epoch=True, prog_bar=False, sync_dist=sync, rank_zero_only=True)

    def training_step(self, inputs):
        # base_opt, pe_opt = self.optimizers()
        rgbd, p2p, gt_depth = inputs['image'], inputs['p2p'], inputs['depth_label']
        model_inputs = None
        if not self.model_cfg.get('multiview_distillation', False):
            model_inputs = rgbd
        else:
            model_inputs = (rgbd, p2p)

        outputs = self(model_inputs)
        with torch.no_grad():
            merged_dict = tu.merge_dict(('inputs', inputs), ('outputs', outputs))

        loss_dict, meta_data = self.loss(merged_dict)
        loss = sum(w*v for w, v in loss_dict.values())

        loss_dict = tu.prefix_dict('train', loss_dict)
        meta_data = tu.prefix_dict('train', meta_data)

        self.log('train/loss', loss, on_step=True, prog_bar=True, rank_zero_only=True, sync_dist=True)
        self._log_metrics(loss_dict, meta_data, step=True, sync=True)

        # base_opt.zero_grad()
        # pe_opt.zero_grad()
        # self.manual_backward(loss)
        # base_opt.step()
        # pe_opt.step()

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

    # def on_train_epoch_end(self):
    #     """
    #     Freeze PE map after epoch 2
    #     """
    #     if self.current_epoch == self.freeze_epoch:
    #         print(f'---- Freezing PE Map @ epoch {self.current_epoch} ----')
    #         self.model.freeze_pe_map()

    def validation_step(self, inputs):
        rgbd, p2p, gt_depth = inputs['image'], inputs['p2p'], inputs['depth_label']
        if not self.model_cfg.get('multiview_distillation', False):
            model_inputs = rgbd
        else:
            model_inputs = (rgbd, p2p)

        outputs = self( model_inputs )
        with torch.no_grad():
            merged_dict = tu.merge_dict(('inputs', inputs), ('outputs', outputs))

        loss_dict, meta_data = self.loss(merged_dict)
        loss = sum(w*v for w, v in loss_dict.values())

        loss_dict = tu.prefix_dict('val', loss_dict)
        meta_data = tu.prefix_dict('val', meta_data)

        # Log subset of outputs to tensorboard
        if len(self.loggers) > 0 and len(self.log_keys) > 0 and self.val_step_count % 10 == 0:
            tbu.log_feat_img_to_tb(
                self.loggers[0], inputs, outputs, self.log_keys, self.current_epoch, self.val_step_count, prefix='val'
            )
            tbu.log_depth_img_to_tb(
                self.loggers[0], inputs, outputs, ['depth_preds_metric'], self.current_epoch, self.val_step_count, prefix='val'
            )
        self.val_step_count += 1

        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=True)
        self._log_metrics(loss_dict, meta_data, step=False, sync=True)

        return {"loss": loss}

    def test_step(self, inputs):
        rgbd, p2p, gt_depth = inputs['image'], inputs['p2p'], inputs['depth_label']
        if not self.model_cfg.get('multiview_distillation', False):
            model_inputs = rgbd
        else:
            model_inputs = (rgbd, p2p)

        outputs = self( model_inputs )
        with torch.no_grad():
            merged_dict = tu.merge_dict(('inputs', inputs), ('outputs', outputs))

        loss_dict, meta_data = self.loss(merged_dict)
        loss = sum(w*v for w, v in loss_dict.values())

        loss_dict = tu.prefix_dict('test', loss_dict)
        meta_data = tu.prefix_dict('test', meta_data)

        # Log subset of outputs to tensorboard
        if len(self.log_keys) > 0 and self.test_step_count % 10 == 0:
            tbu.log_feat_img_to_tb(
                self.loggers[0], inputs, outputs, self.log_keys, self.current_epoch, self.test_step_count, prefix='test'
            )
        self.test_step_count += 1

        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, rank_zero_only=True, sync_dist=True)
        self._log_metrics(loss_dict, meta_data, step=False, sync=True)

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

    is_zero_rank = os.environ.get('LOCAL_RANK', 0)==0 and os.environ.get('NODE_RANK', 0)==0

    # Set save path to original ckpt directory
    stage = cfg.get("stage", "fit")
    print("---- Model Stage: ", stage, " ----")
    ckpt_path = cfg.model.get('ckpt_path', '')
    is_ckpt_valid = os.path.isfile(ckpt_path) and ckpt_path.endswith('.ckpt')
    weights_path = cfg.model.get('weights_path', '')
    is_test_weights = os.path.isfile(weights_path) and weights_path.endswith('.ckpt') and stage == "test"
    
    assert not (is_ckpt_valid and is_test_weights), "Both ckpt and weights path provided. Please provide only one"
    if is_ckpt_valid or is_test_weights:
        ckpt_path = ckpt_path if is_ckpt_valid else weights_path
        print("Using old checkpoint path: ", ckpt_path)
        CKPT_SAVE_DIR = os.path.dirname(ckpt_path)

        epoch = int(ckpt_path.split('-')[-1].split('.')[0].split('=')[-1])
        tb_logdir = f'testepoch_{epoch}'
    else:
        if is_zero_rank:
            CKPT_SAVE_DIR = tu.get_save_paths(cfg)
            os.environ['CKPT_SAVE_DIR'] = CKPT_SAVE_DIR
        else:
            CKPT_SAVE_DIR = os.environ['CKPT_SAVE_DIR']
        tb_logdir = 'train'

    print("---- Checkpoint Save Directory: ", CKPT_SAVE_DIR, " ----")
    RUN_NAME = CKPT_SAVE_DIR.split('/')[-3]
    WANDB_RUN_NAME = cfg.get("wandb_name", cfg.dataset["model_type"]) + "_" +  RUN_NAME

    # Get checkpoint save paths
    codatamodule = CODaPEFreeModule(
        OmegaConf.to_object(cfg.dataset),
        views=cfg.model.views,
        batch_size=cfg.model.batch_size,
        num_workers=cfg.trainer.num_workers
    )

    # Setup training pipeline
    model = DistillationModel(cfg.model)
    checkpoint_callback = ModelCheckpoint(
        monitor=cfg.model.monitor_metric,
        dirpath=CKPT_SAVE_DIR,
        filename=f'{cfg.model.optimizer.name}'+'-{epoch:02d}',
        auto_insert_metric_name=True, # Inserts the val acc to file
        save_top_k=-1, # Saves all models by default
        # mode=cfg.model.monitor_mode,
    )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    tb_dir = join(CKPT_SAVE_DIR, 'tb')
    os.makedirs(tb_dir, exist_ok=True)
    tensorboard_logger = TensorBoardLogger(
        save_dir=tb_dir, name=tb_logdir
    )
    
    strategy = DDPStrategy(find_unused_parameters=True)
    if stage == "fit":
        codatamodule.setup("fit")

        wandb_logger = WandbLogger(
            project=cfg.model.project_name,
            name=WANDB_RUN_NAME,
        )

        coda_train = codatamodule.train_dataloader()
        coda_val = codatamodule.val_dataloader()

        if is_ckpt_valid:
            model = model.load_from_checkpoint(ckpt_path)
            print("Loaded model from checkpoint")

        trainer = pl.Trainer(
            default_root_dir=cfg.trainer.default_root_dir,
            max_epochs=cfg.trainer.max_epochs, 
            accelerator=cfg.trainer.accelerator, 
            devices=cfg.trainer.devices, 
            logger=[tensorboard_logger, wandb_logger],
            callbacks=[lr_monitor, checkpoint_callback],
            log_every_n_steps=1,  # Log metrics every step
            strategy=strategy,
            accumulate_grad_batches=cfg.trainer.accumulate_grad_batches
        )
        trainer.fit(model, coda_train, coda_val)
    elif stage == "test":
        codatamodule.setup("test")
        coda_test = codatamodule.test_dataloader()

        model.eval()
        trainer = pl.Trainer(
            default_root_dir=cfg.trainer.default_root_dir,
            max_epochs=cfg.trainer.max_epochs, 
            accelerator=cfg.trainer.accelerator, 
            devices=cfg.trainer.devices, 
            logger=[tensorboard_logger],
            callbacks=[lr_monitor, checkpoint_callback],
            log_every_n_steps=1,  # Log metrics every step
            strategy=strategy,
            accumulate_grad_batches=cfg.trainer.accumulate_grad_batches
        )
        trainer.test(model, coda_test)

@hydra.main(version_base=None, config_path="../configs", config_name="distillation")
def main(cfg: DictConfig):
    train(cfg)

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    main()
