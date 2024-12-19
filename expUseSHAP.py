#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Using the SHAP tool
#   
#
################################################################################
import sys
import pickle
import pandas as pd
import numpy as np
import shap
from omdd import OMDD

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

                explainer = shap.KernelExplainer(model=mdd_model.predict, data=Xs, feature_names=features)
                sc = explainer.shap_values(Xs)
                header_line = ",".join(features)
                header_line = header_line.lstrip("#")
                np.savetxt(f"results/sc/{name}.csv", sc, delimiter=",", header=header_line, comments="", fmt=f"%.3f")
