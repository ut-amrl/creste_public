import yaml
import copy
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler

import lightning as L
import pytorch_lightning as pl
from lightning.pytorch.utilities import CombinedLoader

from creste.datasets.coda_dataloader_depth import CODatasetDepth
from creste.datasets.codapefree_dataloader import CodaPEFreeDataset
from creste.datasets.coda_utils import SSC_LABEL_DIR, SOC_LABEL_DIR, TRAVERSE_LABEL_DIR, TASK_TO_LABEL

def worker_init_fn(worker_id):
    torch.initial_seed()  # Re-seed each worker process to ensure diversity

class CODaDepthModule(L.LightningDataModule):
    def __init__(self, cfg, batch_size, num_workers, do_shuffle=True, training=True):
        super().__init__()
        self.cfg = cfg
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.do_shuffle = do_shuffle
        self.training = training

        self.setup()

    def setup(self):
        if self.training:
            self.coda_train = CODatasetDepth(
                self.cfg, split="training", annos_type="Depth"
            )
            self.coda_val = CODatasetDepth(
                self.cfg, split="validation", annos_type="Depth"
            )
        self.coda_test = CODatasetDepth(
            self.cfg, split="testing", annos_type="Depth", do_augmentation=False
        )

    def train_dataloader(self):
        return DataLoader(
            self.coda_train,
            batch_size=self.batch_size,
            shuffle=self.do_shuffle,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.coda_train.collate_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.coda_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.coda_val.collate_fn
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.coda_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.coda_test.collate_fn
        )

class CODaPEFreeModule(L.LightningDataModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        print("/**              Loading dataloader overrides            **/")
        self.cfg = cfg
        for key, value in kwargs.items():
            if key in self.cfg:
                print(f'Overriding {key} in config: {self.cfg[key]}')
            print(f'Setting {key} to {value}')
            self.cfg[key] = value
        print("/**              Dataloader overrides complete            **/")

    def setup(self, stage: str):
        mode = self.cfg.get('mode', 'normal')
        if stage=='fit' and mode=='normal':
            self.coda_train = CodaPEFreeDataset(
                self.cfg,
                split="training",
                model_type=self.cfg['model_type'],
                views=self.cfg['views'],
                skip_sequences=self.cfg['skip_sequences'],
                do_augmentation=True,
                fload_keys=self.cfg['fload_keys'],
                sload_keys=self.cfg['sload_keys'],
                use_global_pose=self.cfg.get('use_global_pose', False),
                feature_pred_dir=self.cfg.get('feature_pred_dir', ''),
                camids=self.cfg.get('camids', ['cam0'])
            )
            self.coda_val = CodaPEFreeDataset(
                self.cfg,
                split="validation",
                model_type=self.cfg['model_type'],
                views=self.cfg['views'],
                skip_sequences=self.cfg['skip_sequences'],
                fload_keys=self.cfg['fload_keys'],
                sload_keys=self.cfg['sload_keys'],
                use_global_pose=self.cfg.get('use_global_pose', False),
                feature_pred_dir=self.cfg.get('feature_pred_dir', ''),
                do_augmentation=False,
                camids=self.cfg.get('camids', ['cam0'])
            )
        elif stage=='fit' and mode=='cluster_probe':
            self.coda_train = CodaPEFreeDataset(
                self.cfg,
                split="all",
                model_type=self.cfg['model_type'],
                views=self.cfg['views'],
                skip_sequences=self.cfg['skip_sequences'],
                fload_keys=self.cfg['train_keys']['fload_keys'],
                sload_keys=self.cfg['train_keys']['sload_keys'],
                use_global_pose=self.cfg.get('use_global_pose', False),
                feature_pred_dir=self.cfg.get('feature_pred_dir', ''),
                do_augmentation=False,
                task_cfgs=[self.cfg['task_cfgs'][0]],
                camids=self.cfg.get('camids', ['cam0'])
            )
            self.coda_val = CodaPEFreeDataset(
                self.cfg,
                split="all",
                model_type=self.cfg['model_type'],
                views=self.cfg['views'],
                skip_sequences=self.cfg['skip_sequences'],
                fload_keys=self.cfg['val_keys']['fload_keys'],
                sload_keys=self.cfg['val_keys']['sload_keys'],
                use_global_pose=self.cfg.get('use_global_pose', False),
                feature_pred_dir=self.cfg.get('feature_pred_dir', ''),
                do_augmentation=False,
                task_cfgs=self.cfg['task_cfgs'],
                camids=self.cfg.get('camids', ['cam0'])
            )
        elif stage=='full':
            self.coda_full = CodaPEFreeDataset(
                self.cfg,
                split="full",
                model_type=self.cfg['model_type'],
                views=self.cfg['views'],
                skip_sequences=self.cfg['skip_sequences'],
                fload_keys=self.cfg['fload_keys'],
                sload_keys=self.cfg['sload_keys'],
                use_global_pose=self.cfg.get('use_global_pose', False),
                feature_pred_dir=self.cfg.get('feature_pred_dir', ''),
                do_augmentation=False,
                camids=self.cfg.get('camids', ['cam0'])
            )
        else:
            self.coda_test = CodaPEFreeDataset(
                self.cfg,
                split="testing",
                model_type=self.cfg['model_type'],
                views=self.cfg['views'],
                skip_sequences=self.cfg['skip_sequences'],
                fload_keys=self.cfg['fload_keys'],
                sload_keys=self.cfg['sload_keys'],
                use_global_pose=self.cfg.get('use_global_pose', False),
                feature_pred_dir=self.cfg.get('feature_pred_dir', ''),
                do_augmentation=False,
                camids=self.cfg.get('camids', ['cam0'])
            )

    def full_dataloader(self):
        return DataLoader(
            self.coda_full,
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.coda_full.collate_fn,
            worker_init_fn=worker_init_fn
        )

    def train_dataloader(self):
        return DataLoader(
            self.coda_train,
            batch_size=self.cfg['batch_size'],
            shuffle=True,
            num_workers=self.cfg['num_workers'],
            pin_memory=True,
            collate_fn=self.coda_train.collate_fn,
            persistent_workers=True,
            drop_last=True, 
            worker_init_fn=worker_init_fn
        )

    def val_dataloader(self):
        return DataLoader(
            self.coda_val,
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=True,
            persistent_workers=True,
            collate_fn=self.coda_val.collate_fn,
        )
        
    def test_dataloader(self):
        return DataLoader(
            self.coda_test,
            batch_size=self.cfg['batch_size'],
            shuffle=False,
            num_workers=self.cfg['num_workers'],
            pin_memory=True,
            collate_fn=self.coda_test.collate_fn
        )

class CODaSSCModule(pl.LightningDataModule):
    def __init__(self, cfg, **kwargs):
        super().__init__()
        print("/**              Loading dataloader overrides            **/")
        self.cfg = cfg
        for key, value in kwargs.items():
            if key in self.cfg:
                print(f'Overriding {key} in config: {self.cfg[key]}')
            print(f'Setting {key} to {value}')
            self.cfg[key] = value

        # Convert task_cfgs to to keys to use for later
        self.cfg['task_dict'] = {task['name']: task['kwargs'] for task in self.cfg['task_cfgs']}
        
        # Check if validation config is different from training config
        self.cfg['validation_keys'] = self.cfg.get('validation_keys', [SSC_LABEL_DIR])

        # Set num views to 1 if not specified
        self.cfg['views'] = cfg.get('views', 1)
        self.datasets = cfg.get('datasets',
            [
                {
                    "ssc": { 
                        "tasks": ["3d_ssc"]
                    }
                }
            ]
        )
        print("/**              Dataloader overrides complete            **/")

        self.prepare_data_per_node = True

    def setup_keys(self, tasks):
        sload_keys = ['p2p', 'fov_mask']
        task_cfgs = {}
        for task in tasks:
            assert task in self.cfg['task_dict'], f"Task {task} not found in task_dict"
            task_kwargs = self.cfg['task_dict'][task] 
            sload_keys.append(f'{task}_label')
            task_cfgs[task] = copy.deepcopy(task_kwargs)

        return sload_keys, task_cfgs

    def setup(self, stage: str):
        self.train_dict = {}
        self.val_dict = {}
        self.test_dict = {}
        self.all_dict = {}

        print(f"Running with batch size per GPU: {self.cfg['batch_size']}")
        for dataset in self.datasets:
            name, tasks, _ = dataset.values()
            if stage=="fit":
                sload_keys, task_cfgs = self.setup_keys(tasks)
                self.train_dict[name] = CodaPEFreeDataset(
                    self.cfg,
                    split="training",
                    model_type=self.cfg['model_type'],
                    views=self.cfg['views'],
                    skip_sequences=self.cfg['skip_sequences'],
                    do_augmentation=True,
                    fload_keys=self.cfg['fload_keys'],
                    sload_keys=sload_keys,
                    use_global_pose=self.cfg.get('use_global_pose', False),
                    feature_pred_dir=self.cfg.get('feature_pred_dir', ''),
                    camids=self.cfg.get('camids', ['cam0'])
                )

                # # Add 3d_ssc for validation set
                val_cfg = copy.deepcopy(self.cfg)
                # for task in self.cfg['validation_keys']:
                #     if f'{task}_label' in val_cfg['sload_keys']:
                #         continue

                #     if task==SSC_LABEL_DIR:
                #         print(f"Adding {task} to validation set")
                #         val_cfg['task_cfgs'].append(
                #             {
                #                 'name': task,
                #                 "kwargs": {
                #                     "remap_labels": True,
                #                     "num_classes": 25,
                #                     "ext": "bin"
                #                 }
                #             },
                #         )
                #     val_cfg['sload_keys'] = sload_keys + [f'{task}_label']
                
                self.val_dict[name] = CodaPEFreeDataset(
                    val_cfg,
                    split="validation",
                    model_type=val_cfg['model_type'],
                    views=val_cfg['views'],
                    skip_sequences=val_cfg['skip_sequences'],
                    fload_keys=val_cfg['fload_keys'],
                    sload_keys=val_cfg['sload_keys'],
                    use_global_pose=val_cfg.get('use_global_pose', False),
                    feature_pred_dir=val_cfg.get('feature_pred_dir', ''),
                    do_augmentation=False,
                    camids=self.cfg.get('camids', ['cam0'])
                )
            elif stage=="test":
                self.test_dict[name] = CodaPEFreeDataset(
                    self.cfg,
                    split="testing",
                    model_type=self.cfg['model_type'],
                    views=1,
                    skip_sequences=self.cfg['skip_sequences'],
                    fload_keys=self.cfg['fload_keys'],
                    sload_keys=self.cfg['sload_keys'],
                    use_global_pose=self.cfg.get('use_global_pose', False),
                    feature_pred_dir=self.cfg.get('feature_pred_dir', ''),
                    do_augmentation=False,
                    camids=self.cfg.get('camids', ['cam0'])
                )
            elif stage=="all":
                self.all_dict[name] = CodaPEFreeDataset(
                    self.cfg,
                    split="all",
                    model_type=self.cfg['model_type'],
                    views=self.cfg['views'],
                    skip_sequences=self.cfg['skip_sequences'],
                    fload_keys=self.cfg['fload_keys'],
                    sload_keys=self.cfg['sload_keys'],
                    use_global_pose=self.cfg.get('use_global_pose', False),
                    feature_pred_dir=self.cfg.get('feature_pred_dir', ''),
                    do_augmentation=False,
                    camids=self.cfg.get('camids', ['cam0'])
                )

    def _create_dataloader(self, dataset_dict, shuffle):
        # Check if distributed training is enabled
        is_distributed = self.trainer and self.trainer.num_devices > 1

        # Distributed sampler has its only shuffler
        return CombinedLoader(
            {
                name: DataLoader(
                    dataset,
                    batch_size=self.cfg['batch_size'],
                    shuffle=shuffle if not is_distributed else False,
                    sampler=DistributedSampler(dataset, shuffle=shuffle) if is_distributed else None,
                    num_workers=self.cfg['num_workers'],
                    pin_memory=True,
                    drop_last=True if shuffle else False,
                    collate_fn=CodaPEFreeDataset.collate_fn,
                    worker_init_fn=worker_init_fn
                )
                for name, dataset in dataset_dict.items()
            },
            mode='max_size_cycle'
        )

    def train_dataloader(self):
        return self._create_dataloader(self.train_dict, shuffle=True)

    def val_dataloader(self):
        return self._create_dataloader(self.val_dict, shuffle=False)

    def test_dataloader(self):
        return self._create_dataloader(self.test_dict, shuffle=False)

    def all_dataloader(self):
        return self._create_dataloader(self.all_dict, shuffle=False)

    # def train_dataloader(self):
    #     return CombinedLoader(
    #         { name: DataLoader(
    #             dataset,
    #             batch_size=self.cfg['batch_size'],
    #             shuffle=True,
    #             num_workers=self.cfg['num_workers'],
    #             pin_memory=True,
    #             # persistent_workers=True,
    #             collate_fn=CodaPEFreeDataset.collate_fn,
    #             drop_last=True, 
    #             worker_init_fn=worker_init_fn
    #         ) for name, dataset in self.train_dict.items() },
    #         mode='max_size_cycle',
    #     )
        
    # def val_dataloader(self):
    #     return CombinedLoader(
    #         { name: DataLoader(
    #             dataset,
    #             batch_size=self.cfg['batch_size'],
    #             shuffle=True,
    #             num_workers=self.cfg['num_workers'],
    #             pin_memory=True,
    #             # persistent_workers=True,
    #             collate_fn=CodaPEFreeDataset.collate_fn,
    #             drop_last=False, 
    #             worker_init_fn=worker_init_fn
    #         ) for name, dataset in self.val_dict.items() },
    #         mode='max_size_cycle',
    #     )
        
    # def test_dataloader(self):
    #     return CombinedLoader(
    #         { name: DataLoader(
    #             dataset,
    #             batch_size=self.cfg['batch_size'],
    #             shuffle=False,
    #             num_workers=self.cfg['num_workers'],
    #             pin_memory=True,
    #             persistent_workers=True,
    #             collate_fn=CodaPEFreeDataset.collate_fn,
    #             drop_last=False, 
    #             worker_init_fn=worker_init_fn
    #         ) for name, dataset in self.test_dict.items() },
    #         mode='max_size_cycle',
    #     )

    # def all_dataloader(self):
    #     return CombinedLoader(
    #         { name: DataLoader(
    #             dataset,
    #             batch_size=self.cfg['batch_size'],
    #             shuffle=False,
    #             num_workers=self.cfg['num_workers'],
    #             pin_memory=True,
    #             persistent_workers=True,
    #             collate_fn=CodaPEFreeDataset.collate_fn,
    #             drop_last=False, 
    #             worker_init_fn=worker_init_fn
    #         ) for name, dataset in self.all_dict.items() },
    #         mode='max_size_cycle',
    #     )

if __name__ == "__main__":
    import numpy as np
    cfg_file = "./configs/dataset/coda_pefree.yaml"
    with open(cfg_file, 'r') as f:
        cfg = yaml.safe_load(f)

    codatamodule = CODaPEFreeModule(
        cfg,
        views=3,
        batch_size=3,
        num_workers=4,
        stage="fit",
        skip_sequences=np.arange(1, 23).tolist()
    )
    codatamodule.setup("fit")
    coda_train = codatamodule.train_dataloader()

    for batch_idx, batch in enumerate(coda_train):
        print(batch.keys(), batch_idx)
        if batch_idx > 3:
            break
