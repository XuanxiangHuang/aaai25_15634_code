#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Reproduce experiments
#
################################################################################
import sys
import pickle
import numpy as np
import pandas as pd
from omdd import OMDD
from xpmdd import XpOMDD
################################################################################


if __name__ == '__main__':
    args = sys.argv[1:]
    # example: python3 XXX.py -bench dt_ijar_examples.txt model (dt, rf)
    if len(args) >= 3 and args[0] == '-bench':
        bench_name = args[1]
        md = args[2]

        with open(bench_name, 'r') as fp:
            name_list = fp.readlines()

        if md == 'rf':
            raise NotImplementedError("not implemented yet.")

        elif md == 'dt':
            for item in name_list:
                name = item.strip()
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

                all_feat_cnts = []
                d_len = len(Xs)
                for i, x in enumerate(Xs):
                    print(f"{name}, {i}-th instance out of {d_len}")
                    pred = mdd_model.predict_one(x)
                    inst = list(x)
                    assert len(inst) == nf
                    print(f"Instance: {x, pred}")

                    xpmdd = XpOMDD(dd=mdd_model, inst=inst, tar=pred, verb=0)
                    # relevancy/irrelevancy counter
                    feat_cnts = nf * [0]
                    axps, cxps = xpmdd.enum()
                    for axp in axps:
                        for feat in axp:
                            feat_cnts[feat] += 1
                    all_feat_cnts.append(feat_cnts)

                header_line = ",".join(features)
                header_line = header_line.lstrip("#")
                np.savetxt(f"results/frp/{name}.csv", np.array(all_feat_cnts), delimiter=",", header=header_line, comments="", fmt='%d')
