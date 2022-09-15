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

from dataset import REORG_G_MEAN, REORG_G_STD, REORG_EX_MEAN, REORG_EX_STD
class PredictionModel(nn.Module):
    def __init__(self, config: DictConfig):
        super().__init__()
        mot_config = MoTConfig(**MolecularEncoder.mot_config, **config.model.config)

        self.model = MoTModel(mot_config)
        self.classifier = nn.Linear(mot_config.hidden_dim*2, 2)

    def forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_g, batch_ex = batch
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
        hidden_states = torch.cat([hidden_states_g, hidden_states_ex], dim=-1)
        logits = self.classifier(hidden_states[:, 0, :]).squeeze(-1)  # [batch, 2]

        logits[:,0] = logits[:,0] * REORG_G_STD + REORG_G_MEAN
        logits[:,1] = logits[:,1] * REORG_EX_STD + REORG_EX_MEAN

        return logits
        # hidden_states = torch.cat([hidden_states_g, hidden_states_ex], dim=-1)
        # return self.classifier(hidden_states[:, 0, :]).squeeze(-1)


class TestSSDDataset(SSDDataset):
    def __init__(self, dataset: pd.DataFrame, structure_files: List[str], encoder: MolecularEncoder, bond_drop_prob: float = 0.0):
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
        df = pd.read_csv(dataset["label"], index_col="index")
        datasets.append(df)

        for idx in df.index:
            g_str = os.path.join(dataset["structures"], idx+"_g.mol")
            ex_str = os.path.join(dataset["structures"], idx+"_ex.mol")
            structure_files.append((idx, g_str, ex_str))

    dataset = pd.concat(datasets)
    encoder = MolecularEncoder()

    # Define a collate function to stack individual samples into the batch. This process
    # is necessary because each sample has different length of sequence. To run the
    # model in parallel, it is important to match the lengths.
    def collate_fn(features: List) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        uids = [dict_['uid'] for dict_ in features]
        encoding_gs = [dict_['encoding_g'] for dict_ in features]
        encoding_ex = [dict_['encoding_ex'] for dict_ in features]

        encoding_gs = encoder.collate(
            encoding_gs,
            max_length=config.data.max_length,
            pad_to_multiple_of=8,
        )
        encoding_ex = encoder.collate(
            encoding_ex,
            max_length=config.data.max_length,
            pad_to_multiple_of=8,
        )
        return uids, (encoding_gs, encoding_ex)

    return DataLoader(
        dataset=SSDDataset(dataset, structure_files, encoder, bond_drop_prob=0.0, predict=True),
        batch_size=config.predict.batch_size,
        collate_fn=collate_fn
    )

def convert_to_cuda(batch):
    batch = {
            "input_ids": [x.cuda() for x in batch["input_ids"]],
            "attention_mask": batch["attention_mask"].cuda(),
            "attention_type_ids": batch["attention_type_ids"].cuda(),
            "position_ids": batch["position_ids"].cuda(),
        }
    return batch

@torch.no_grad()
def main(config: DictConfig):
    dataloader = create_dataloader(config)

    model = PredictionModel(config).eval().cuda()
    model.load_state_dict(torch.load(config.model.pretrained_model_path))
    print(f"Loaded model from {config.model.pretrained_model_path}")

    preds = []
    for uids, batch_pair in tqdm.tqdm(dataloader):
        batch_g, batch_ex = batch_pair
        batch_g = convert_to_cuda(batch_g) #{k: v.cuda() for k, v in batch_g.items()}
        batch_ex = convert_to_cuda(batch_ex) #{k: v.cuda() for k, v in batch_ex.items()}

        for uid, target in zip(uids, model((batch_g, batch_ex)).tolist()):

            preds.append({"uid": uid, "Reorg_g": target[0], "Reorg_ex": target[1]})

    preds = pd.DataFrame(preds)
    preds['index_num'] = preds.uid.str.split('_').str[1].astype(int)
    preds = preds.sort_values('index_num')
    preds = preds.drop('index_num', axis=1)

    preds.columns = ['index', 'Reorg_g', 'Reorg_ex']

    preds.to_csv(config.model.pretrained_model_path.replace(".pth", ".csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args, unknown_args = parser.parse_known_args()

    config = OmegaConf.load(args.config)
    config.merge_with_dotlist(unknown_args)

    main(config)
