import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
from omegaconf import DictConfig
from sklearn.model_selection import KFold
from torch.nn import functional as F
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from dataset import SSDDataset
from encoding import MolecularEncoder
from modeling import MoTConfig, MoTLayerNorm, MoTModel

try:
    from apex.optimizers import FusedAdam as AdamW
except ModuleNotFoundError:
    from torch.optim import AdamW


class FineTuningModule(pl.LightningModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config
        mot_config = MoTConfig(**MolecularEncoder.mot_config, **config.model.config)

        # Before creating and initializing models, we need to fix the random seed for
        # consistent training and performance. It is well-known that to control the
        # random seed is important in transformer-based finetuning tasks.
        torch.manual_seed(config.model.random_seed)

        self.model = MoTModel(mot_config)

        self.ex_fn1 = nn.Linear(mot_config.hidden_dim, int(np.sqrt(mot_config.hidden_dim)))
        self.ex_fn2 = nn.Linear(int(np.sqrt(mot_config.hidden_dim)), 1)

        self.g_fn1 = nn.Linear(mot_config.hidden_dim, int(np.sqrt(mot_config.hidden_dim)))
        self.g_fn2 = nn.Linear(int(np.sqrt(mot_config.hidden_dim)), 1)

        self.model.init_weights(self.ex_fn1)
        self.model.init_weights(self.ex_fn2)
        self.model.init_weights(self.g_fn1)
        self.model.init_weights(self.g_fn2)

        if self.config.model.pretrained_model_path is not None:
            state_dict = torch.load(self.config.model.pretrained_model_path)
            self.model.load_state_dict(state_dict)

    def forward(
        self, batch: List[Dict[str, torch.Tensor]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_g, batch_ex, labels = batch
        hidden_states_g = self.model(
            batch_g["input_ids"],
            batch_g["attention_mask"],
            batch_g["attention_type_ids"],
            batch_g["position_ids"],
        )
        hidden_states_ex = self.model(
            batch_ex["input_ids"],
            batch_ex["attention_mask"],
            batch_ex["attention_type_ids"],
            batch_ex["position_ids"],
        )

        # hidden_states = torch.cat([hidden_states_g, hidden_states_ex], dim=-1)
        g_g = self.g_fn2(F.relu(self.g_fn1(hidden_states_g[:,0, :])))
        g_ex = self.g_fn2(F.relu(self.g_fn1(hidden_states_ex[:,0, :])))
        ex_g = self.ex_fn2(F.relu(self.ex_fn1(hidden_states_g[:,0, :])))
        ex_ex = self.ex_fn2(F.relu(self.ex_fn1(hidden_states_ex[:,0, :])))

        lambda_g = torch.abs(g_ex - g_g)
        lambda_ex = torch.abs(ex_ex - ex_g)
        logits = torch.cat([lambda_g, lambda_ex], dim=-1)  # (B, 2)

        mse_loss = F.mse_loss(logits, labels.type_as(logits))
        mae_loss = F.l1_loss(logits, labels.type_as(logits))
        return mse_loss, mae_loss

    def training_step(self, batch: Dict[str, torch.Tensor], idx: int) -> torch.Tensor:
        mse_loss, mae_loss = self(batch[1])
        self.log("train/mse_loss", mse_loss)
        self.log("train/mae_loss", mae_loss)
        self.log("train/score", mae_loss)
        return mse_loss

    def validation_step(self, batch: Dict[str, torch.Tensor], idx: int):
        mse_loss, mae_loss = self(batch[1])
        self.log("val/mse_loss", mse_loss)
        self.log("val/mae_loss", mae_loss)
        self.log("val/score", mae_loss)

    def create_param_groups(self) -> List[Dict[str, Any]]:
        """Create parameter groups for the optimizer.

        Transformer-based models are usually optimized by AdamW (weight-decay decoupled
        Adam optimizer). And weight decaying are applied to only weight parameters, not
        bias and layernorm parameters. Hence, this method creates parameter groups which
        contain parameters for weight-decay and ones for non-weight-decay. Using this
        paramter groups, you can separate which parameters should not be decayed from
        entire parameters in this model.

        Returns:
            A list of parameter groups.
        """
        do_decay_params, no_decay_params = [], []
        for layer in self.modules():
            for name, param in layer.named_parameters(recurse=False):
                if isinstance(layer, MoTLayerNorm) or name == "bias":
                    no_decay_params.append(param)
                else:
                    do_decay_params.append(param)

        return [
            {"params": do_decay_params},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

    def adjust_learning_rate(self, current_step: int) -> float:
        """Calculate a learning rate scale corresponding to current step.

        MoT pretraining uses a linear learning rate decay with warmups. This method
        calculates the learning rate scale according to the linear warmup decaying
        schedule. Using this method, you can create a learning rate scheduler through
        `LambdaLR`.

        Args:
            current_step: A current step of training.

        Returns:
            A learning rate scale corresponding to current step.
        """
        training_steps = self.get_total_training_steps()
        warmup_steps = int(training_steps * self.config.train.warmup_ratio)

        if current_step < warmup_steps:
            return current_step / warmup_steps
        return max(0, (training_steps - current_step) / (training_steps - warmup_steps))

    def get_total_training_steps(self) -> int:
        """Calculate the total training steps from the trainer.

        If you are using epochs to limit training on pytorch lightning, you cannot
        directly get the entire training steps. It requires some complicated ways to get
        the training steps. This method uses the number of samples in the dataloader,
        distributed devices and accumulations to get approximately correct training
        steps.

        Returns:
            The total training steps.
        """
        if self.trainer.max_steps:
            return self.trainer.max_steps

        limit_batches = self.trainer.limit_train_batches
        batches = len(self.train_dataloader())
        batches = (
            min(batches, limit_batches)
            if isinstance(limit_batches, int)
            else int(limit_batches * batches)
        )

        num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)
        if self.trainer.tpu_cores:
            num_devices = max(num_devices, self.trainer.tpu_cores)

        effective_accum = self.trainer.accumulate_grad_batches * num_devices
        return (batches // effective_accum) * self.trainer.max_epochs

    def configure_optimizers(self) -> Tuple[List[Optimizer], List[Dict[str, Any]]]:
        optimizer = AdamW(self.create_param_groups(), **self.config.train.optimizer)
        scheduler = LambdaLR(optimizer, self.adjust_learning_rate)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]


class FineTuningDataModule(pl.LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()
        self.config = config

    def setup(self, stage: Optional[str] = None):
        self.encoder = MolecularEncoder()

        # Read label csv files and collect SDF molfiles from the configuration. The
        # dataframes will be concatenated and used to find the labels when loading
        # batches.
        datasets, structure_files = [], []
        for dataset in self.config.data.dataset_files:
            df = pd.read_csv(dataset["label"], index_col="index")
            datasets.append(df)

            for idx in df.index:
                g_str = os.path.join(dataset["structures"], idx+"_g.mol")
                ex_str = os.path.join(dataset["structures"], idx+"_ex.mol")
                structure_files.append((idx, g_str, ex_str))

            # structure_files += [
            #     os.path.join(dataset["structures"], filename)
            #     for filename in os.listdir(dataset["structures"])
            # ]
        dataset = pd.concat(datasets)

        # Split the structure file list into k-folds. Note that the splits will be same
        # because of the random seed fixing.
        kfold = KFold(self.config.data.num_folds, shuffle=True, random_state=42)
        train_val_sets = list(kfold.split(structure_files))[self.config.data.fold_index]

        train_structure_files = [structure_files[i] for i in train_val_sets[0]]
        val_structure_files = [structure_files[i] for i in train_val_sets[1]]

        self.train_dataset = SSDDataset(
            dataset,
            structure_files=train_structure_files,
            encoder=self.encoder,
            bond_drop_prob=self.config.data.bond_drop_prob,
        )
        self.val_dataset = SSDDataset(
            dataset,
            structure_files=val_structure_files,
            encoder=self.encoder,
            bond_drop_prob=0.0,
        )

    @property
    def num_dataloader_workers(self) -> int:
        """Get the number of parallel workers in each dataloader."""
        if self.config.data.dataloader_workers >= 0:
            return self.config.data.dataloader_workers
        return os.cpu_count()

    def dataloader_collate_fn(
        self, features: List[Dict[str, Union[torch.Tensor, int, str]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        """Simple datacollate binder for dataloaders."""
        uids = [dict_['uid'] for dict_ in features]
        encoding_gs = [dict_['encoding_g'] for dict_ in features]
        encoding_ex = [dict_['encoding_ex'] for dict_ in features]
        labels = [dict_['labels'] for dict_ in features]

        encoding_gs = self.encoder.collate(
            encoding_gs,
            max_length=self.config.data.max_length,
            pad_to_multiple_of=8,
        )
        encoding_ex = self.encoder.collate(
            encoding_ex,
            max_length=self.config.data.max_length,
            pad_to_multiple_of=8,
        )
        # list of numpy to tensor
        labels = torch.tensor(labels, dtype=torch.float32)
        return uids, (encoding_gs, encoding_ex, labels)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.num_dataloader_workers,
            collate_fn=self.dataloader_collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.train.batch_size,
            num_workers=self.num_dataloader_workers,
            collate_fn=self.dataloader_collate_fn,
            drop_last=True,
            persistent_workers=True,
        )
