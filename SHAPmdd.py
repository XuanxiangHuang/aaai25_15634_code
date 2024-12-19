#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   Computing SHAP score for OMDD classifiers
#   
#
################################################################################
import math
from itertools import chain, combinations
from omdd import OMDD
################################################################################


def powerset_generator(input):
    # Generate all subsets of the input set
    for subset in chain.from_iterable(combinations(input, r) for r in range(len(input) + 1)):
        yield set(subset)


class SHAPoMDD(object):
    """
        Compute SHAP-score of OMDD, note that OMDD support polytime model counting.
    """

    def __init__(self, dd: OMDD, verb=0):
        self.dd = dd
        self.verbose = verb

    def model_counting(self, inst, tar, univ):
        """
            BE CAREFUL about the MULTI-EDGES!
            Given a list of universal features,
            count the number of models and universal features.

            :param univ: a list of universal features.
            :return: number of models and universal features
        """

        assert len(univ) == self.dd.nf
        assign = dict()
        G = self.dd.graph
        for nd in self.dd.dfs_postorder(self.dd.root):
            if not G.out_degree(nd):
                if G.nodes[nd]['target'] == tar:
                    assign.update({nd: 1})
                else:
                    assign.update({nd: 0})
        for nd in self.dd.dfs_postorder(self.dd.root):
            if not G.out_degree(nd):
                continue
            feat_nd = G.nodes[nd]['var']
            f_id_nd = self.dd.features.index(feat_nd)
            feat_lvl_nd = self.dd.feat2lvl[feat_nd]
            total = 0
            if univ[f_id_nd]:
                for s in G.successors(nd):
                    assert s in assign
                    if G.out_degree(s):
                        feat_s = G.nodes[s]['var']
                        feat_lvl_s = self.dd.feat2lvl[feat_s]
                    else:
                        feat_lvl_s = self.dd.nf
                    assert feat_lvl_nd < feat_lvl_s
                    prod = 1
                    for lvl_i in range(feat_lvl_nd+1, feat_lvl_s):
                        f_i = self.dd.features.index(self.dd.lvl2feat[lvl_i])
                        if univ[f_i]:
                            prod *= len(self.dd.feat_domain[self.dd.lvl2feat[lvl_i]])
                    # multi-edges between nd and s
                    total += assign[s] * prod * G.number_of_edges(nd, s)
            else:
                for s in G.successors(nd):
                    assert s in assign
                    # multi-edges case is not appliable
                    if tuple((nd, s, inst[f_id_nd])) in G.edges:
                        if G.out_degree(s):
                            feat_s = G.nodes[s]['var']
                            feat_lvl_s = self.dd.feat2lvl[feat_s]
                        else:
                            feat_lvl_s = self.dd.nf
                        assert feat_lvl_nd < feat_lvl_s
                        prod = 1
                        for lvl_i in range(feat_lvl_nd+1, feat_lvl_s):
                            f_i = self.dd.features.index(self.dd.lvl2feat[lvl_i])
                            if univ[f_i]:
                                prod *= len(self.dd.feat_domain[self.dd.lvl2feat[lvl_i]])
                        total += assign[s] * prod
            assert nd not in assign
            assign.update({nd: total})
        assert self.dd.root in assign
        n_model = assign[self.dd.root]
        return n_model

    def expect_value(self, inst, univ):
        """
            Compute the expectation value of the given instance.
        """
        label_cnt = dict()
        for i in self.dd.tar_range:
            cnt = self.model_counting(inst, i, univ)
            label_cnt.update({i: cnt})
        expect_val = sum(i * label_cnt[i] for i in label_cnt)
        for i in range(self.dd.nf):
            feat = self.dd.features[i]
            if univ[i]:
                expect_val *= self.dd.fv_probs[feat][self.dd.feat_domain[feat].index(inst[i])]
        return expect_val

    def similarity_func(self, inst, univ):
        """
            Compute the expectation value of the given instance using the similarity function.
        """
        pred = self.dd.predict_one(inst)
        cnt = self.model_counting(inst, pred, univ)
        for i in range(self.dd.nf):
            feat = self.dd.features[i]
            if univ[i]:
                cnt *= self.dd.fv_probs[feat][self.dd.feat_domain[feat].index(inst[i])]
        return cnt

    def algo_by_def(self, inst, target_feat, vtype='expected'):
        """
            Computing SHAP-score by definition (using model counting).
        :param inst: given instance
        :param target_feat: given feature
        :param vtype: value function type
        :return: the SHAP-score of the target feature on given instance
        with respect to given OMDD under uniform distribution
        """
        nf = self.dd.nf
        feats = list(range(nf))
        assert target_feat in feats
        feats.remove(target_feat)
        all_S = list(powerset_generator(feats))
        shap_score = 0
        for S in all_S:
            len_S = len(list(S))
            univ = [False] * nf
            for i in range(nf):
                if i not in list(S):
                    univ[i] = True

            univ[target_feat] = False
            if vtype == 'expected':
                mds_with_t = self.expect_value(inst, univ)
            elif vtype == 'similarity':
                mds_with_t = self.similarity_func(inst, univ)
            else:
                raise ValueError("Unknown value function.")

            univ[target_feat] = True
            if vtype == 'expected':
                mds_without_t = self.expect_value(inst, univ)
            elif vtype == 'similarity':
                mds_without_t = self.similarity_func(inst, univ)
            else:
                raise ValueError("Unknown value function.")

            if mds_with_t - mds_without_t == 0:
                continue
            shap_score += math.factorial(len_S) * math.factorial(nf-len_S-1) * (mds_with_t - mds_without_t) / math.factorial(nf)
        return shap_score
