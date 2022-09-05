import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from encoding import MolecularEncoder

ST1_ENERGY_GAP_MEAN = 0.8486
ST1_ENERGY_GAP_STD = 0.3656


class SSDDataset(Dataset):
    """A dataset class for `Samsung AI Challenge For Scientific Discovery` competition.

    Args:
        two sets of data: g and ex
        data : g_structure, ex_structure, g_energy, ex_energy
    """

    def __init__(
        self,
        dataset: pd.DataFrame,
        structure_files: List[str],
        encoder: MolecularEncoder,
        bond_drop_prob: float = 0.1,
    ):
        self.examples = []
        self.encoder = encoder
        self.bond_drop_prob = bond_drop_prob

        for structure_file in structure_files:
            # extract the uid from the filename
            uid = '_'.join(os.path.basename(structure_file).split("_")[:-1])
            type_ = os.path.basename(structure_file).split("_")[-1].split(".")[0]
            example = {"uid": uid}

            with open(structure_file, "r") as fp:
                example["structure"] = parse_mol_structure(fp.read())
                if example["structure"] is None:
                    continue
            
            if type_ == 'g':
                label = dataset.loc[example["uid"], "Reorg_g"]

            elif type_ == 'ex':
                label = dataset.loc[example["uid"], "Reorg_ex"]

            example["labels"] = label
            self.examples.append(example)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self, index: int
    ) -> Tuple[str, Dict[str, Union[str, List[Union[int, float]]]]]:
        example = self.examples[index]

        if np.random.rand() < self.bond_drop_prob:
            # We will drop the molecular bonds with probability of 15%. That is, the
            # expectation of the number of dropped molecular bonds is 85% of the
            # original one. Note that you can only control the molecular selecting
            # probability, not the individual bond dropping probability.
            structure = example["structure"].copy()
            structure["bonds"] = [
                bond for bond in structure["bonds"] if np.random.rand() > 0.15
            ]
            example["structure"] = structure

        encoding = self.encoder.encode(example["structure"])

        if "labels" in example:
            encoding["labels"] = example["labels"]
        return (example["uid"], encoding)


def parse_mol_structure(data: str) -> Optional[Dict]:
    """Parse a SDF molecular file to the simple structure dictionary.

    Args:
        data: The content of SDF molfile.

    Returns:
        The parsed 3D molecular structure dictionary.
    """
    data = data.splitlines()
    if len(data) < 4:
        return None

    data = data[3:]
    num_atoms, num_bonds = int(data[0][:3]), int(data[0][3:6])

    atoms = []
    for line in data[1 : 1 + num_atoms]:
        x, y, z = float(line[:10]), float(line[10:20]), float(line[20:30])
        charge = [0, 3, 2, 1, "^", -1, -2, -3][int(line[36:39])]
        atoms.append([x, y, z, line[31:34].strip(), charge])

    bonds = []
    for line in data[1 + num_atoms : 1 + num_atoms + num_bonds]:
        bonds.append([int(line[:3]) - 1, int(line[3:6]) - 1, int(line[6:9])])

    for line in data[1 + num_atoms + num_bonds :]:
        if not line.startswith("M  CHG") and not line.startswith("M  RAD"):
            continue
        for i in range(int(line[6:9])):
            idx = int(line[10 + 8 * i : 13 + 8 * i]) - 1
            value = int(line[14 + 8 * i : 17 + 8 * i])

            atoms[idx][4] = (
                [":", "^", "^^"][value - 1] if line.startswith("M  RAD") else value
            )

    return {"atoms": atoms, "bonds": bonds}
