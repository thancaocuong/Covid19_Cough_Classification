import pandas as pd
import glob
import os
import numpy as np
import argparse
from copy import deepcopy
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('-d', '--result_dir', default="results/tiny_ast_224", type=str,
                      help='result directory')
    args = parser.parse_args()
    all_files = glob.glob(os.path.join(args.result_dir, "fold*", "*.csv"))
    all_results = [pd.read_csv(file_path) for file_path in all_files]
    all_label = np.concatenate([result["assessment_result"].to_numpy().reshape(1, -1) for result in all_results])
    refined_score = all_label.mean(0)
    final_pred = pd.DataFrame(columns=["uuid", "assessment_result"])
    final_pred["uuid"] = all_results[0]["uuid"]
    final_pred["assessment_result"] = refined_score
    final_pred.to_csv(os.path.join(args.result_dir, "results.csv"))