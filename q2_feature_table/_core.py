# ----------------------------------------------------------------------------
# Copyright (c) 2016-2026, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------

import numpy as np
import pandas as pd


def core_score(
        table: pd.DataFrame,
        min_rel_abundance: float = 1e-3,
        mean_abundance_on_presence: bool = False,
        eps: float = 1e-6
) -> pd.DataFrame:
    """Compute prevalence-abundance core scores for feature table columns."""
    prevalence = (table > min_rel_abundance).mean(axis=0)
    if mean_abundance_on_presence:
        mean_abundance = table.where(
            table > min_rel_abundance).mean(axis=0).fillna(0)
    else:
        mean_abundance = table.mean(axis=0)
    log_mean = np.log10(mean_abundance + eps)

    prevalence_scaled = _minmax_scale(prevalence, eps)
    log_mean_scaled = _minmax_scale(log_mean, eps)
    score = prevalence_scaled * log_mean_scaled

    result = pd.DataFrame({'core_score': score})
    result.index.name = 'id'
    return result


def _minmax_scale(x, eps):
    """Scale values to the range [0, 1] with an offset denominator."""
    return (x - x.min()) / (x.max() - x.min() + eps)
