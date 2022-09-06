import argparse
import os
from typing import Dict, List, Tuple, Union

import pandas as pd
import torch
import torch.nn as nn
import tqdm
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from dataset import SSDDataset, parse_mol_structure
from encoding import MolecularEncoder
from modeling import MoTConfig, MoTModel


class PredictionModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        mot_config = MoTConfig(**MolecularEncoder.mot_config, **config.model.config)

        self.model = MoTModel(mot_config)
        self.classifier = nn.Linear(mot_config.hidden_dim, 1)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden_states = self.model(
            batch["input_ids"],
            batch["attention_mask"],
            batch["attention_type_ids"],
            batch["position_ids"],
        )
        return self.classifier(hidden_states[:, 0, :]).squeeze(-1)


class TestSSDDataset(SSDDataset):
    def __init__(self, dataset:pd.DataFrame, structure_files: List[str], encoder: MolecularEncoder, bond_drop_prob:float=0.0):
        self.examples = []
        self.encoder = encoder
        self.bond_drop_prob = bond_drop_prob

        for structure_file in structure_files:
            # extract the uid from the filename
            uid = '_'.join(os.path.basename(structure_file).split("_")[:-1])
            type_ = os.path.basename(structure_file).split("_")[-1].split(".")[0]
            example = {"uid": uid, "type": type_}

            with open(structure_file, "r") as fp:
                example["structure"] = parse_mol_structure(fp.read())
                if example["structure"] is None:
                    continue

            self.examples.append(example)
        
    def __getitem__(self, index: int) -> Tuple[str, Dict[str, Union[str, List[Union[int, float]]]]]:
        example = self.examples[index]
        encoding = self.encoder.encode(example["structure"])
        return example["uid"], encoding, example['type']

def create_dataloader(config: DictConfig) -> DataLoader:
    # Read label csv files and collect SDF molfiles from the configuration. The
    # dataframes will be concatenated and used to find the labels when loading batches.
    datasets, structure_files = [], []
    for dataset in config.data.dataset_files:
        datasets.append(pd.read_csv(dataset["labels"], index_col="index"))
        structure_files += [
            os.path.join(dataset["structures"], filename)
            for filename in os.listdir(dataset["structures"])
        ]

    dataset = pd.concat(datasets)
    encoder = MolecularEncoder()

    # Define a collate function to stack individual samples into the batch. This process
    # is necessary because each sample has different length of sequence. To run the
    # model in parallel, it is important to match the lengths.
    def collate_fn(features: List) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        uids = [uid for uid, encoding, _ in features]
        encodings = [encoding for uid, encoding, _ in features]
        types = [type_ for _, _, type_ in features]
        return uids, encoder.collate(encodings, config.data.max_length, 8), types

    return DataLoader(
        dataset=TestSSDDataset(dataset, structure_files, encoder, bond_drop_prob=0.0),
        batch_size=config.predict.batch_size,
        collate_fn=collate_fn
    )


@torch.no_grad()
def main(config: DictConfig):
    dataloader = create_dataloader(config)

    model = PredictionModel(config).eval().cuda()
    model.load_state_dict(torch.load(config.model.pretrained_model_path))

    preds_g = []
    preds_ex = []
    for uids, batch, type_ in tqdm.tqdm(dataloader):
        batch = {
            "input_ids": [x.cuda() for x in batch["input_ids"]],
            "attention_mask": batch["attention_mask"].cuda(),
            "attention_type_ids": batch["attention_type_ids"].cuda(),
            "position_ids": batch["position_ids"].cuda(),
            "type": type_
        }
        for uid, target, t in zip(uids, model(batch).tolist(), type_):
            # target = target * ST1_ENERGY_GAP_STD + ST1_ENERGY_GAP_MEAN
            if t == "ex":
                preds_ex.append({"index": uid, "Reorg_ex":target})
            elif t == "g":
                preds_g.append({"index": uid, "Reorg_g": target})

    preds_g = pd.DataFrame(preds_g).set_index("index")
    preds_ex = pd.DataFrame(preds_ex).set_index("index")
    preds = pd.concat([preds_g, preds_ex], axis=1)

    # sort the predictions by the index_num
    preds['index_num'] = preds.index.str.split('_').str[1].astype(int)
    preds = preds.sort_values('index_num')
    preds = preds.drop('index_num', axis=1)

    preds.to_csv(config.model.pretrained_model_path.replace(".pth", ".csv"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)

    main(config)

