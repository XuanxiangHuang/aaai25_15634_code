#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Using the SHAP tool with different value functions (similarity)
#   
#
################################################################################
import sys
import pickle
import pandas as pd
import numpy as np
import shap
from omdd import OMDD
from value_functions import valueFunctions

np.random.seed(73)
################################################################################


if __name__ == '__main__':
    args = sys.argv[1:]
    # example: python3 XXX.py -bench dt_ijar_examples.txt model (dt, rf)
    if len(args) >= 3 and args[0] == '-bench':
        bench_name = args[1]
        md = args[2]

        with open(bench_name, 'r') as fp:
            datasets = fp.readlines()

        if md == 'rf':
            raise NotImplementedError("not implemented yet.")

        elif md == 'dt':
            for ds in datasets:
                name = ds.strip()
                data = f"samples/{name}.csv"
                mdd_file = f"dt_models/{name}.mdd"
                print(f"############ {name} ############")
                df = pd.read_csv(data)
                features = list(df.columns)
                target = features.pop()
                Xs = df[features].values.astype(int)
                mdd_model = OMDD.from_file(mdd_file)
                nn = len(mdd_model.graph.nodes)
                nf = mdd_model.nf
                assert mdd_model.features == features
                assert mdd_model.target == target

                all_scores = []
                d_len = len(Xs)
                for i, x in enumerate(Xs):
                    print(f"{name}, {i}-th instance out of {d_len}")
                    valFunc = valueFunctions(mdd_model, mdd_model.predict_one(list(x)))
                    explainer_s = shap.KernelExplainer(model=valFunc.valSimilarity, data=Xs, feature_names=features)
                    # The values in the i-th column represent the Shapley values of the corresponding i-th feature.
                    s_sc = explainer_s.shap_values(x)
                    all_scores.append(s_sc)

                header_line = ",".join(features)
                header_line = header_line.lstrip("#")
                np.savetxt(f"results/s_sc/{name}.csv", np.array(all_scores), delimiter=",", header=header_line, comments="", fmt=f"%.3f")
