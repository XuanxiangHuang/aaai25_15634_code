#!/usr/bin/env python
# -*- coding:utf-8 -*-
#
#   value functions (characteristic function)
#   
#
################################################################################
import numpy as np
################################################################################


class valueFunctions(object):
    def __init__(self, classifier, out_label):
        self.classifier = classifier
        self.out_label = out_label

    def valSimilarity(self, data_points):
        """
            Return mapped predictions (use similarity value function) given a list of data points.
            :param data_points: input data points
            :return: mapped predictions of these data points.
        """
        original_predictions = np.array(self.classifier.predict(data_points))
        # Map each element to 1 if it's equal to self.out_label, otherwise to 0
        mapped_predictions = (original_predictions == self.out_label).astype(int)
        return mapped_predictions

    def valWeakAXp(self, data_points):
        # This function is conceptually correct, but it does not work as expected.
        # maybe it is not technically correct, as it implicitly violates that a classifier is a function.
        """
            Return mapped predictions (use weak AXp value function) given a list of data points.
            :param data_points: input data points
            :return: mapped predictions of these data points.
        """
        original_predictions = np.array(self.classifier.predict(data_points))
        # Map each element to 1 if it's equal to self.out_label, otherwise to 0
        mapped_predictions = (original_predictions == self.out_label).astype(int)
        # Replace all elements with 0 if any element is 0
        mapped_predictions *= int(not np.any(mapped_predictions == 0))
        return mapped_predictions

    def valWeakCXp(self, data_points):
        # This function is conceptually correct, but it does not work as expected.
        # maybe it is not technically correct, as it implicitly violates that a classifier is a function.
        """
            Return mapped predictions (use weak CXp value function) given a list of data points.
            :param data_points: input data points
            :return: mapped predictions of these data points.
        """
        original_predictions = np.array(self.classifier.predict(data_points))
        # Map each element to 1 if it's equal to self.out_label, otherwise to 0
        mapped_predictions = (original_predictions == self.out_label).astype(int)
        # Replace all elements with 0 if all elements are 1
        if np.all(mapped_predictions == 1):
            mapped_predictions = np.zeros(len(mapped_predictions)).astype(int)
        else:
            # Replace all elements with 1 if any element is 0
            mapped_predictions = np.ones(len(mapped_predictions)).astype(int)
        return mapped_predictions
