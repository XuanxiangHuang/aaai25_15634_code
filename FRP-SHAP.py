#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   SHAP score of (explanation) irrelevant feature > SHAP score of (explanation) relevant feature
#   
#
################################################################################
import sys, os
import pandas as pd
import numpy as np
################################################################################


# python3 XXX.py -bench dt_ijar_examples.txt
if __name__ == '__main__':
    args = sys.argv[1:]
    if len(args) >= 2 and args[0] == '-bench':
        bench_name = args[1]

        with open(bench_name, 'r') as fp:
            name_list = fp.readlines()

        for item in name_list:
            name = item.strip()
            print(f"################## {name} ##################")
            frp_file_path = os.path.join("results/frp", f"{name}.csv")
            sc_file_path = os.path.join("results/sc", f"{name}.csv")
            s_sc_file_path = os.path.join("results/s_sc", f"{name}.csv")

            frp_data = pd.read_csv(frp_file_path)
            sc_data = pd.read_csv(sc_file_path)
            s_sc_data = pd.read_csv(s_sc_file_path)

            frp_scores = np.round(np.abs(frp_data.to_numpy()), decimals=4)
            sc_scores = np.round(np.abs(sc_data.to_numpy()), decimals=4)
            s_sc_scores = np.round(np.abs(s_sc_data.to_numpy()), decimals=4)

            cnt_frp_sc = 0
            cnt_frp_s_sc = 0
            cnt_better = 0
            cnt_worse = 0
            # Iterate through each line in the scores
            for i, (frp_line, sc_line, s_sc_line) in enumerate(zip(frp_scores, sc_scores, s_sc_scores), start=1):
                # Check if there is no 0 in frp_score for the current line
                if 0 not in frp_line:
                    continue

                # Split sc_line based on frp_line
                rel_sc = [sc_val for frp_val, sc_val in zip(frp_line, sc_line) if frp_val != 0]
                irr_sc = [sc_val for frp_val, sc_val in zip(frp_line, sc_line) if frp_val == 0]

                rel_s_sc = [s_sc_val for frp_val, s_sc_val in zip(frp_line, s_sc_line) if frp_val != 0]
                irr_s_sc = [s_sc_val for frp_val, s_sc_val in zip(frp_line, s_sc_line) if frp_val == 0]

                if max(irr_sc) >= min(rel_sc):
                    cnt_frp_sc += 1
                if max(irr_s_sc) >= min(rel_s_sc):
                    cnt_frp_s_sc += 1
                if max(irr_sc) >= min(rel_sc) and max(irr_s_sc) < min(rel_s_sc):
                    cnt_better += 1
                if max(irr_sc) < min(rel_sc) and max(irr_s_sc) >= min(rel_s_sc):
                    cnt_worse += 1
            print(f"SHAP-FRP mismatch: {cnt_frp_sc}", f"sSHAP-FRP mismatch: {cnt_frp_s_sc}", f"sSHAP is better: {cnt_better}", f"sSHAP is worse: {cnt_worse}")
