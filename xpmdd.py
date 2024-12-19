#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   OMDD classifiers explainer
################################################################################
import time
from itertools import chain, combinations
from omdd import OMDD
from pysat.formula import IDPool
from pysat.solvers import Solver as SAT_Solver
################################################################################


def powerset_generator(input):
    # Generate all subsets of the input set
    for subset in chain.from_iterable(combinations(input, r) for r in range(len(input) + 1)):
        yield set(subset)


def checkMHS(in_axps: list, in_cxps: list):
    # given a list of axp and a list of cxp,
    # check if they are minimal-hitting-set (MHS) of each other
    # 1. uniqueness, and no subset(superset) exists;
    if not in_axps or not in_cxps:
        print(f"input empty: {in_axps}, {in_cxps}")
        return False
    axps = sorted(in_axps, key=lambda x: len(x))
    axps_ = axps[:]
    while axps:
        axp = axps.pop()
        set_axp = set(axp)
        for ele in axps:
            set_ele = set(ele)
            if set_axp.issuperset(set_ele) or set_axp.issubset(set_ele):
                print(f"axp is not unique: {set_axp}, {set_ele}")
                return False
    cxps = sorted(in_cxps, key=lambda x: len(x))
    cxps_ = cxps[:]
    while cxps:
        cxp = cxps.pop()
        set_cxp = set(cxp)
        for ele in cxps:
            set_ele = set(ele)
            if set_cxp.issuperset(set_ele) or set_cxp.issubset(set_ele):
                print(f"cxp is not unique: {set_cxp}, {set_ele}")
                return False
    # 2. minimal hitting set;
    for axp in axps_:
        set_axp = set(axp)
        for cxp in cxps_:
            set_cxp = set(cxp)
            if not (set_axp & set_cxp):  # not a hitting set
                print(f"not a hitting set: axp:{set_axp}, cxp:{set_cxp}")
                return False
    # axp is a MHS of cxps
    for axp in axps_:
        set_axp = set(axp)
        for ele in set_axp:
            tmp = set_axp - {ele}
            size = len(cxps_)
            for cxp in cxps_:
                set_cxp = set(cxp)
                if tmp & set_cxp:
                    size -= 1
            if size == 0:  # not minimal
                print(f"axp is not minimal hitting set: "
                      f"axp {set_axp} covers #{len(cxps_)}, "
                      f"its subset {tmp} covers #{len(cxps_) - size}, "
                      f"so {ele} is redundant")
                return False
    # cxp is a MHS of axps
    for cxp in cxps_:
        set_cxp = set(cxp)
        for ele in set_cxp:
            tmp = set_cxp - {ele}
            size = len(axps_)
            for axp in axps_:
                set_axp = set(axp)
                if tmp & set_axp:
                    size -= 1
            if size == 0:
                print(f"cxp is not minimal hitting set: "
                      f"cxp {set_cxp} covers #{len(axps_)}, "
                      f"its subset {tmp} covers #{len(axps_) - size}, "
                      f"so {ele} is redundant")
                return False
    return True


class XpOMDD(object):

    def __init__(self, dd: OMDD, inst, tar, verb=0):
        self.dd = dd                    # OMDD model
        self.inst = inst                # instance
        self.tar = tar                  # target value
        self.verbose = verb

    def find_axp(self, fixed=None):
        """
            Compute one abductive explanation (Axp).

            :param fixed: a list of features declared as fixed.
            :return: one abductive explanation,
                        each element in the return Axp is a feature index.
        """

        time_solving_start = time.perf_counter()

        # get/create fix array
        if not fixed:
            fix = [True] * self.dd.nf
        else:
            fix = fixed.copy()
        assert (len(fix) == self.dd.nf)

        for i in range(self.dd.nf):
            if fix[i]:
                fix[i] = not fix[i]
                if self.dd.path_to_other_class(self.inst, self.tar, [not v for v in fix]):
                    fix[i] = not fix[i]

        axp = [i for i in range(self.dd.nf) if fix[i]]
        assert len(axp)

        time_solving_end = time.perf_counter()
        solving_time = time_solving_end - time_solving_start

        if self.verbose:
            if self.verbose == 1:
                print(f"Axp: {axp}")
            elif self.verbose == 2:
                print(f"Axp: {axp} ({[self.dd.features[i] for i in axp]})")
            print("Runtime: {0:.3f}".format(solving_time))

        return axp

    def find_cxp(self, universal=None):
        """
            Compute one contrastive explanation (Cxp).

            :param universal: a list of features declared as universal.
            :return: one contrastive explanation,
                        each element in the return Cxp is a feature index.
        """

        time_solving_start = time.perf_counter()

        # get/create univ array
        if not universal:
            univ = [True] * self.dd.nf
        else:
            univ = universal.copy()
        assert (len(univ) == self.dd.nf)

        for i in range(self.dd.nf):
            if univ[i]:
                univ[i] = not univ[i]
                if not self.dd.path_to_other_class(self.inst, self.tar, univ):
                    univ[i] = not univ[i]

        cxp = [i for i in range(self.dd.nf) if univ[i]]
        assert len(cxp)

        time_solving_end = time.perf_counter()
        solving_time = time_solving_end - time_solving_start

        if self.verbose:
            if self.verbose == 1:
                print(f"Cxp: {cxp}")
            elif self.verbose == 2:
                print(f"Cxp: {cxp} ({[self.dd.features[i] for i in cxp]})")
            print("Runtime: {0:.3f}".format(solving_time))

        return cxp

    def enum(self):
        """
            Enumerate all (abductive and contrastive) explanations, using MARCO algorithm.

            :return: a list of all Axps, a list of all Cxps.
        """

        #########################################
        vpool = IDPool()

        def new_var(name):
            """
                Inner function,
                Find or new a PySAT variable.
                See PySat.

                :param name: name of variable
                :return: index of variable
            """
            return vpool.id(f'{name}')

        #########################################

        time_solving_start = time.perf_counter()

        axps = []
        cxps = []

        for i in range(self.dd.nf):
            new_var(f'u_{i}')
        # initially all features are fixed (in other words, no features are universal).
        univ = [False] * self.dd.nf

        with SAT_Solver(name="glucose4") as slv:
            while slv.solve():
                # first model is empty
                model = slv.get_model()
                for lit in model:
                    name = vpool.obj(abs(lit)).split(sep='_')
                    univ[int(name[1])] = False if lit < 0 else True
                if self.dd.path_to_other_class(self.inst, self.tar, univ):
                    cxp = self.find_cxp(univ)
                    slv.add_clause([-new_var(f'u_{i}') for i in cxp])
                    cxps.append(cxp)
                else:
                    axp = self.find_axp([not i for i in univ])
                    slv.add_clause([new_var(f'u_{i}') for i in axp])
                    axps.append(axp)

        time_solving_end = time.perf_counter()
        solving_time = time_solving_end - time_solving_start
        if self.verbose:
            print('#AXp:', len(axps))
            print('#CXp:', len(cxps))
            print("Runtime: {0:.3f}".format(solving_time))

        return axps, cxps

    def check_one_axp(self, axp):
        """
            Check if given axp is 1) a weak AXp and 2) subset-minimal.

            :param axp: potential abductive explanation.
            :return: true if given axp is an AXp
                        else false.
        """

        univ = [True] * self.dd.nf
        for i in axp:
            univ[i] = not univ[i]
        # 1) axp is a weak AXp if there are no path to 0.
        if self.dd.path_to_other_class(self.inst, self.tar, univ):
            print(f'given axp {axp} is not a weak AXp')
            return False
        # 2) axp is subset-minimal if axp \ {i} will activate a path to 0.
        for i in range(len(univ)):
            if not univ[i]:
                univ[i] = not univ[i]
                if self.dd.path_to_other_class(self.inst, self.tar, univ):
                    univ[i] = not univ[i]
                else:
                    print(f'given axp {axp} is not subset-minimal')
                    return False
        return True

    def check_one_cxp(self, cxp):
        """
            Check if given cxp is 1) a weak CXp and 2) subset-minimal.

            :param cxp: given cxp.
            :return: true if given cxp is an CXp
                        else false.
        """

        univ = [False] * self.dd.nf
        for i in cxp:
            univ[i] = True
        # 1) cxp is a weak CXp if there is a path to 0.
        if not self.dd.path_to_other_class(self.inst, self.tar, univ):
            print(f'given cxp {cxp} is not a weak CXp')
            return False
        # 2) cxp is subset-minimal if cxp \ {i} will block all paths to 0.
        for i in range(self.dd.nf):
            if univ[i]:
                univ[i] = not univ[i]
                if not self.dd.path_to_other_class(self.inst, self.tar, univ):
                    univ[i] = not univ[i]
                else:
                    print(f'given cxp {cxp} is not subset-minimal')
                    return False
        return True
