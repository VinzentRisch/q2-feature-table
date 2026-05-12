# ----------------------------------------------------------------------------
# Copyright (c) 2016-2026, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

from unittest import main

import numpy as np
import pandas as pd
import pandas.testing as pdt
from rachis.plugin.testing import TestPluginBase

from q2_feature_table import core_score
from q2_feature_table._core_score import _minmax_scale


class TestCoreScore(TestPluginBase):
    package = 'q2_feature_table.tests'

    def setUp(self):
        super().setUp()
        self.table = pd.DataFrame(
            np.array([
                [0.0, 0.005, 0.0],
                [0.02, 0.005, 0.0],
                [0.04, 0.005, 0.3],
            ]),
            index=['S1', 'S2', 'S3'],
            columns=['O1', 'O2', 'O3']
        )

    def test_core_score(self):
        observed = core_score(self.table)

        prevalence = np.array([2 / 3, 1, 1 / 3])
        mean_abundance = np.array([0.06 / 3, 0.005, 0.1])
        log_mean = np.log10(mean_abundance + 1e-6)
        exp_prevalence = _minmax_scale(prevalence, 1e-6)
        exp_log_mean = _minmax_scale(log_mean, 1e-6)
        expected = pd.DataFrame(
            {'core_score': exp_prevalence * exp_log_mean},
            index=pd.Index(['O1', 'O2', 'O3'], name='id')
        )

        pdt.assert_frame_equal(observed, expected)

    def test_core_score_values_are_finite(self):
        observed = core_score(self.table)
        self.assertTrue(np.isfinite(observed.to_numpy()).all())

    def test_core_score_custom_parameters(self):
        observed = core_score(
            self.table, min_rel_abundance=0.01, eps=1e-3)

        prevalence = np.array([2 / 3, 0, 1 / 3])
        mean_abundance = np.array([0.06 / 3, 0.005, 0.1])
        log_mean = np.log10(mean_abundance + 1e-3)
        exp_prevalence = _minmax_scale(prevalence, 1e-3)
        exp_log_mean = _minmax_scale(log_mean, 1e-3)
        expected = pd.DataFrame(
            {'core_score': exp_prevalence * exp_log_mean},
            index=pd.Index(['O1', 'O2', 'O3'], name='id')
        )

        pdt.assert_frame_equal(observed, expected)

    def test_core_score_mean_abundance_on_presence(self):
        observed = core_score(
            self.table, min_rel_abundance=0.01,
            mean_abundance_on_presence=True)

        prevalence = np.array([2 / 3, 0, 1 / 3])
        mean_abundance = np.array([0.03, 0, 0.3])
        log_mean = np.log10(mean_abundance + 1e-6)
        exp_prevalence = _minmax_scale(prevalence, 1e-6)
        exp_log_mean = _minmax_scale(log_mean, 1e-6)
        expected = pd.DataFrame(
            {'core_score': exp_prevalence * exp_log_mean},
            index=pd.Index(['O1', 'O2', 'O3'], name='id')
        )

        pdt.assert_frame_equal(observed, expected)

    def test_minmax_scale(self):
        observed = _minmax_scale(pd.Series([2, 2, 2]), eps=1e-6)
        self.assertTrue(np.isfinite(observed.to_numpy()).all())


if __name__ == "__main__":
    main()
