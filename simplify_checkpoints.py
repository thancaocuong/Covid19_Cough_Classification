from glob import glob
from tqdm import tqdm
import torch
import argparse
import os

parser = argparse.ArgumentParser("Remove Optimizer of ckpts")
parser.add_argument("--ckpt_dir", help="checkpoint directory")
args = parser.parse_args()

all_ckpts = glob(os.path.join(args.ckpt_dir, "*checkpoint*"))
progress = tqdm(total=len(all_ckpts))
for ckpt_path in all_ckpts:
    model = torch.load(ckpt_path, map_location="cpu")
    new_state_dict = dict(state_dict=model["state_dict"])
    torch.save(new_state_dict, ckpt_path)
    progress.update(1)