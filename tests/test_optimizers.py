import numpy as np
import pytest

from axiom.core.optimizers import SGD, MomentumSGD


def make_params():
    return [np.array([1.0, -2.0, 3.0]), np.array([[0.5, 0.25], [-1.0, 2.0]])]


def make_grads():
    return [np.array([0.2, 0.4, -1.0]), np.array([[1.0, -1.0], [0.0, 3.0]])]


def test_sgd_step_moves_each_param_by_minus_lr_times_grad_in_place():
    params, grads = make_params(), make_grads()
    before = [p.copy() for p in params]
    ids = [id(p) for p in params]
    SGD(lr=0.1).step(params, grads)
    for p, b, g in zip(params, before, grads):
        np.testing.assert_allclose(p, b - 0.1 * g)
    assert [id(p) for p in params] == ids


def test_momentum_first_step_uses_one_minus_beta_of_the_gradient():
    params, grads = make_params(), make_grads()
    before = [p.copy() for p in params]
    MomentumSGD(lr=0.1, beta=0.9).step(params, grads)
    for p, b, g in zip(params, before, grads):
        np.testing.assert_allclose(p, b - 0.1 * 0.1 * g)


def test_momentum_accumulates_under_a_constant_gradient():
    lr, beta = 0.1, 0.9
    params, grads = make_params(), make_grads()
    opt = MomentumSGD(lr=lr, beta=beta)
    moves = []
    for t in range(1, 6):
        before = params[0].copy()
        opt.step(params, grads)
        move = before - params[0]
        # v_t = beta * v_{t-1} + (1 - beta) * g  =>  v_t = (1 - beta**t) * g
        np.testing.assert_allclose(move, lr * (1 - beta ** t) * grads[0])
        moves.append(np.linalg.norm(move))
    assert all(later > earlier for earlier, later in zip(moves, moves[1:]))


def test_momentum_keyword_is_an_alias_for_beta():
    assert MomentumSGD(lr=0.1, momentum=0.5).beta == 0.5
    assert MomentumSGD(0.1, 0.8).beta == 0.8


def test_mismatched_params_and_grads_raise():
    with pytest.raises(ValueError):
        SGD(lr=0.1).step(make_params(), make_grads()[:1])
    with pytest.raises(ValueError):
        MomentumSGD(lr=0.1).step(make_params(), [np.zeros(3), np.zeros(3)])
