import numpy as np
import pytest
from ...cp_tensor import cp_to_tensor
from ..constrained_parafac import *
from ... import backend as T
from ...testing import assert_array_equal, assert_, assert_array_almost_equal


def test_constrained_parafac_nonnegative():
    """Test for the CANDECOMP-PARAFAC decomposition with ADMM under nonnegativity constraints
    """
    rng = T.check_random_state(1234)
    tol_norm_2 = 0.5
    tol_max_abs = 0.5
    rank = 3
    init = 'random'
    weightsinit, facinit = initialize_constrained_parafac(T.zeros([6, 8, 4]), rank, constraints='nonnegative', init=init)
    tensor = cp_to_tensor((weightsinit, facinit))
    for i in range(len(facinit)):
        facinit[i] += T.tensor(0.1 * rng.random_sample(T.shape(facinit[i])), **T.context(facinit[i]))
    tensorinit = CPTensor((weightsinit, facinit))
    nn_res, errors = constrained_parafac(tensor, constraints='nonnegative', rank=rank, init=tensorinit, random_state=rng, return_errors=True)
    # Make sure all components are positive
    _, nn_factors = nn_res
    for factor in nn_factors:
        assert_(T.all(factor >= 0))
    nn_res = cp_to_tensor(nn_res)

    error = T.norm(nn_res - tensor, 2)
    error /= T.norm(tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')
    # Test the max abs difference between the reconstruction and the tensor
    assert_(T.max(T.abs(nn_res - tensor)) < tol_max_abs,
            f'abs norm of reconstruction error = {T.max(T.abs(nn_res - tensor))} higher than tolerance={tol_max_abs}')


def test_constrained_parafac_l1():
    """Test for the CANDECOMP-PARAFAC decomposition with ADMM and l1 regularization
    """
    rng = T.check_random_state(1234)
    tol_norm_2 = 0.5
    tol_max_abs = 0.5
    rank = 3
    reg_par = [1e-2, 1e-2, 1e-2]
    init = 'random'
    weightsinit, facinit = initialize_constrained_parafac(T.zeros([6, 8, 4]), rank, constraints='sparse_l1', init=init,
                                                          reg_par=reg_par)
    tensor = cp_to_tensor((weightsinit, facinit))
    for i in range(len(facinit)):
        facinit[i] += T.tensor(0.1 * rng.random_sample(T.shape(facinit[i])), **T.context(facinit[i]))
    tensorinit = CPTensor((weightsinit, facinit))
    res, errors = constrained_parafac(tensor, constraints='sparse_l1', rank=rank, init=tensorinit,
                                      random_state=rng, return_errors=True, reg_par=reg_par,
                                      tol_outer=1e-16, n_iter_max=1000)
    res = cp_to_tensor(res)
    error = T.norm(res - tensor, 2)
    error /= T.norm(tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')
    # Test the max abs difference between the reconstruction and the tensor
    assert_(T.max(T.abs(res - tensor)) < tol_max_abs,
            f'abs norm of reconstruction error = {T.max(T.abs(res - tensor))} higher than tolerance={tol_max_abs}')


def test_constrained_parafac_l2():
    """Test for the CANDECOMP-PARAFAC decomposition with ADMM and l2 norm regularization
    """
    rng = T.check_random_state(1234)
    tol_norm_2 = 0.5
    tol_max_abs = 0.5
    rank = 3
    reg_par = [1e-2, 1e-2, 1e-2]
    init = 'random'
    weightsinit, facinit = initialize_constrained_parafac(T.zeros([6, 8, 4]), rank, constraints='l2', init=init,
                                                          reg_par=reg_par)
    tensor = cp_to_tensor((weightsinit, facinit))
    for i in range(len(facinit)):
        facinit[i] += T.tensor(0.1 * rng.random_sample(T.shape(facinit[i])), **T.context(facinit[i]))
    tensorinit = CPTensor((weightsinit, facinit))
    res, errors = constrained_parafac(tensor, constraints='l2', rank=rank, init=tensorinit,
                                      random_state=rng, return_errors=True, reg_par=reg_par,
                                      tol_outer=1-16)
    res = cp_to_tensor(res)
    error = T.norm(res - tensor, 2)
    error /= T.norm(tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')
    # Test the max abs difference between the reconstruction and the tensor
    assert_(T.max(T.abs(res - tensor)) < tol_max_abs,
            f'abs norm of reconstruction error = {T.max(T.abs(res - tensor))} higher than tolerance={tol_max_abs}')


def test_constrained_parafac_squared_l2():
    """Test for the CANDECOMP-PARAFAC decomposition with ADMM and squared l2 norm regularization
    """
    rng = T.check_random_state(1234)
    tol_norm_2 = 0.5
    tol_max_abs = 0.5
    rank = 3
    init = 'random'
    reg_par = [1e-2, 1e-2, 1e-2]
    weightsinit, facinit = initialize_constrained_parafac(T.zeros([6, 8, 4]), rank, constraints='l2_square', init=init)
    tensor = cp_to_tensor((weightsinit, facinit))
    for i in range(len(facinit)):
        facinit[i] += T.tensor(0.1 * rng.random_sample(T.shape(facinit[i])), **T.context(facinit[i]))
    tensorinit = CPTensor((weightsinit, facinit))
    res, errors = constrained_parafac(tensor, constraints='l2_square', rank=rank, init=tensorinit,
                                      random_state=rng, reg_par=reg_par,
                                      return_errors=True, tol_outer=1-16)
    res = cp_to_tensor(res)
    error = T.norm(res - tensor, 2)
    error /= T.norm(tensor, 2)
    assert_(error < tol_norm_2,
            f'norm 2 of reconstruction higher = {error} than tolerance={tol_norm_2}')
    # Test the max abs difference between the reconstruction and the tensor
    assert_(T.max(T.abs(res - tensor)) < tol_max_abs,
            f'abs norm of reconstruction error = {T.max(T.abs(res - tensor))} higher than tolerance={tol_max_abs}')


def test_constrained_parafac_monotonicity():
    """Test for the CANDECOMP-PARAFAC decomposition with ADMM under monotonicity constraints
    """
    rng = T.check_random_state(1234)
    rank = 3
    init = 'random'
    tensorinit = initialize_constrained_parafac(T.zeros([6, 8, 4]), rank, constraints='monotonicity', init=init)
    tensor = cp_to_tensor(tensorinit)
    _, factors = constrained_parafac(tensor, constraints='monotonicity', rank=rank, init=tensorinit, random_state=rng)
    # Testing if estimated factors are monotonic
    for factor in factors:
        assert_(np.all(np.diff(factor, axis=0) >= 0))


def test_constrained_parafac_simplex():
    """Test for the CANDECOMP-PARAFAC decomposition with ADMM simplex constraint
    """
    rng = T.check_random_state(1234)
    rank = 3
    prox_par = [3, 3, 3]
    init = 'random'
    tensor = T.tensor(rng.random_sample([6, 8, 4]))
    _, factors = constrained_parafac(tensor, constraints='simplex', rank=rank, init=init, random_state=rng, prox_par=prox_par)
    for factor in factors:
        assert_array_almost_equal(np.sum(T.to_numpy(factor), axis=0)[0], 3,  decimal=0)


def test_constrained_parafac_normalize():
    """Test for the CANDECOMP-PARAFAC decomposition with ADMM under normalization constraints
    """
    rng = T.check_random_state(1234)
    tol_norm_2 = 0.5
    tol_max_abs = 0.5
    rank = 3
    init = 'random'
    weightsinit, facinit = initialize_constrained_parafac(T.zeros([6, 8, 4]), rank, constraints='normalize', init=init)
    tensor = cp_to_tensor((weightsinit, facinit))
    for i in range(len(facinit)):
        facinit[i] += T.tensor(0.1 * rng.random_sample(T.shape(facinit[i])), **T.context(facinit[i]))
    tensorinit = CPTensor((weightsinit, facinit))
    res, errors = constrained_parafac(tensor, constraints='normalize', rank=rank, init=tensorinit,
                                      random_state=rng, return_errors=True)
    # Check if maximum values is 1
    for i in range(len(facinit)):
        assert_(T.max(res.factors[i]) == 1)


def test_constrained_parafac_soft_sparsity():
    """Test for the CANDECOMP-PARAFAC decomposition with ADMM under soft_sparsity constraints
    """
    rng = T.check_random_state(1234)
    rank = 3
    prox_par = [1, 1, 1]
    init = 'random'
    weightsinit, facinit = initialize_constrained_parafac(T.zeros([6, 8, 4]), rank, constraints='soft_sparsity', init=init)
    tensor = cp_to_tensor((weightsinit, facinit))
    for i in range(len(facinit)):
        facinit[i] += T.tensor(0.1 * rng.random_sample(T.shape(facinit[i])), **T.context(facinit[i]))
    tensorinit = CPTensor((weightsinit, facinit))
    res = constrained_parafac(tensor, constraints='soft_sparsity', rank=rank, init=tensorinit, random_state=rng, prox_par=prox_par)
    # Check if factors have l1 norm smaller than threshold
    for i in range(len(facinit)):
        assert_(np.all(T.to_numpy(T.norm(res.factors[i], 1, axis=0))) <= 1)


def test_constrained_parafac_unimodality():
    """Test for the CANDECOMP-PARAFAC decomposition with ADMM under unimodality constraints
    """
    rng = T.check_random_state(1234)
    rank = 3
    init = 'random'
    tensorinit = initialize_constrained_parafac(T.zeros([6, 8, 4]), rank, constraints='unimodality', init=init)
    tensor = cp_to_tensor(tensorinit)
    _, factors = constrained_parafac(tensor, constraints='unimodality', rank=rank, init=tensorinit, random_state=rng)
    for factor in factors:
        max_location = T.argmax(factor[:, 0])
        assert_(np.all(np.diff(factor[:int(max_location), 0], axis=0) >= 0))
        assert_(np.all(np.diff(factor[int(max_location):, 0], axis=0) <= 0))


def test_constrained_parafac_normalized_sparsity():
    """Test for the CANDECOMP-PARAFAC decomposition with ADMM under normalized sparsity constraints
    """
    rng = T.check_random_state(1234)
    rank = 3
    init = 'random'
    prox_par = [5, 5, 5]
    weightsinit, facinit = initialize_constrained_parafac(T.zeros([6, 8, 4]), rank, constraints='normalized_sparsity',
                                                          init=init, prox_par=prox_par)
    tensor = cp_to_tensor((weightsinit, facinit))
    for i in range(len(facinit)):
        facinit[i] += T.tensor(0.1 * rng.random_sample(T.shape(facinit[i])), **T.context(facinit[i]))
    tensorinit = CPTensor((weightsinit, facinit))
    res, errors = constrained_parafac(tensor, constraints='normalized_sparsity', rank=rank, init=tensorinit,
                                      random_state=rng, return_errors=True, prox_par=prox_par,
                                      tol_outer=1-16)
    # Check if factors are normalized and k-sparse
    for i in range(len(facinit)):
         assert_(T.norm(res.factors[i]) <= 1)
         assert_(np.count_nonzero(res.factors[i]) == 5)