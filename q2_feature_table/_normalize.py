# ----------------------------------------------------------------------------
# Copyright (c) 2016-2023, QIIME 2 development team.
#
# Distributed under the terms of the Modified BSD License.
#
# The full license is in the file LICENSE, distributed with this software.
# ----------------------------------------------------------------------------
from typing import Union

import biom
import numpy as np


def rarefy(table: biom.Table,
           sampling_depth: int,
           with_replacement: bool = False,
           seed: Union[int, str] = 1
           ) -> biom.Table:
    # Generate a random seed if seed = "random"
    if seed == "random":
        rng = np.random.default_rng()
        seed = rng.integers(0, 2 ** 32 - 1)

    if with_replacement:
        table = table.filter(lambda v, i, m: v.sum() >= sampling_depth,
                             inplace=False, axis='sample')

    table = table.subsample(sampling_depth, axis='sample', by_id=False,
                            with_replacement=with_replacement, seed=seed)

    if table.is_empty():
        raise ValueError('The rarefied table contains no samples or features. '
                         'Verify your table is valid and that you provided a '
                         'shallow enough sampling depth.')

    return table
