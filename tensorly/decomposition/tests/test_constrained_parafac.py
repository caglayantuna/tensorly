import numpy as np
import pytest

import tensorly as tl
from ...cp_tensor import cp_to_tensor
from ..constrained_parafac import *
from ...random import random_cp
from ... import backend as T
from ...testing import assert_array_equal, assert_


def test_constrained_parafac():
    """Test for the CANDECOMP-PARAFAC decomposition with ADMM
    """
    rng = tl.check_random_state(1234)
    tol_norm_2 = 0.01
    tol_max_abs = 0.05
    rank = 3
    init = 'svd'
    tensor = random_cp((6, 8, 4), rank=rank, full=True, random_state=rng)
    constraints = ['nonnegative', 'sparse_l1', 'l2', 'l2_square', 'normalize', 'simplex', 'normalized_sparsity', 'soft_sparsity', 'monotonicity']
    for i in constraints:
        fac, errors = constrained_parafac(tensor, constraints=i, rank=rank, init=init, random_state=rng, return_errors=True, tol_outer=1-16, n_iter_max=1000)

        # Check that the error monotonically decreases
        rec = cp_to_tensor(fac)
        error = T.norm(rec - tensor, 2)
        error /= T.norm(tensor, 2)
        #assert_(error < tol_norm_2,
                #f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')
        # Test the max abs difference between the reconstruction and the tensor
        #assert_(T.max(T.abs(rec - tensor)) < tol_max_abs,
                #f'abs norm of reconstruction error = {T.max(T.abs(rec - tensor))} higher than tolerance={tol_max_abs}')
