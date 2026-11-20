
from dataclasses import replace
from itertools import permutations

import numpy as np
import pytest

from models.mechanical_properties import MechanicalProperties
from models.thermal_properties import ThermalProperties
from models.porous_medium import PorousMedium

from physics.darcy_resistance import DarcyResistance
from physics.biot_jkd_resistance import BiotJKDResistance
from physics.reciprocal_r_model import ReciprocalRModel

from numerics.coupling_continuation import CouplingContinuation
from numerics.mode_classifier import ModeClassifier
from numerics.polynomial_root_solver import PolynomialRootSolver


@pytest.fixture
def medium():
    """Cria o meio termo-poroelástico de referência."""

    mechanical = MechanicalProperties(
        phi=0.25,
        rho_s=2650.0,
        rho_f=1000.0,
        a_inf=2.0,
        k0=1e-12,
        eta=1e-3,
        lambda_=8e9,
        mu=6e9,
        m=1e10,
        alpha=0.8,
        viscous_length=1e-5,
    )

    thermal = ThermalProperties(
        C=2.3e6,
        beta_s=150000.0,
        beta_f=50000.0,
        kappa=2.0,
        tau0=0.001,
        theta0=293.15,
    )

    return PorousMedium(
        mechanical=mechanical,
        thermal=thermal,
    )


@pytest.fixture(
    params=[DarcyResistance, BiotJKDResistance],
    ids=["Darcy", "Biot-JKD"],
)
def model(medium, request):
    """Cria o modelo para cada resistência."""

    return ReciprocalRModel(
        medium=medium,
        resistance=request.param(),
    )


@pytest.fixture
def continuation(model):
    """Cria o procedimento de continuação."""

    return CouplingContinuation(model)


def calculate_roots(model, omega):
    """Calcula as raízes diretamente pelo polinômio."""

    solver = PolynomialRootSolver()

    coefficients = model.polynomial_coefficients(omega)

    return solver.solve(coefficients)


def matching_error(calculated, expected):
    """Compara raízes independentemente da ordem."""

    errors = []

    for order in permutations(range(3)):

        relative_errors = [
            abs(calculated[order[i]] - expected[i])
            / max(abs(expected[i]), 1e-30)
            for i in range(3)
        ]

        errors.append(max(relative_errors))

    return min(errors)


@pytest.mark.parametrize("omega", [1e3, 1e6])
def test_coupled_roots_match_direct_solution(
    continuation,
    model,
    omega
):
    """Compara a continuação com a solução direta."""

    modes = continuation.identify(
        omega=omega,
        num_steps=51
    )

    assert set(modes) == {"P1", "P2", "T"}

    calculated = np.array(
        [modes[name] for name in ("P1", "P2", "T")],
        dtype=complex
    )

    expected = calculate_roots(model, omega)

    assert matching_error(
        calculated,
        expected
    ) < 1e-7


def test_uncoupled_reference(model):
    """Verifica a identificação no limite desacoplado."""

    omega = 1e3

    thermal = replace(
        model.medium.thermal,
        beta_s=0.0,
        beta_f=0.0,
    )

    uncoupled_medium = PorousMedium(
        mechanical=model.medium.mechanical,
        thermal=thermal,
    )

    uncoupled_model = ReciprocalRModel(
        medium=uncoupled_medium,
        resistance=model.resistance,
    )

    roots = calculate_roots(
        uncoupled_model,
        omega
    )

    classifier = ModeClassifier(uncoupled_model)

    mapping = classifier.classify_uncoupled(
        omega,
        roots
    )

    continuation = CouplingContinuation(
        uncoupled_model
    )

    modes = continuation.identify(
        omega=omega,
        num_steps=11
    )

    for name in ("P1", "P2", "T"):

        expected = roots[mapping[name]]

        assert modes[name] == pytest.approx(
            expected,
            rel=1e-8
        )


def test_different_step_counts(continuation):
    """Compara discretizações do acoplamento."""

    omega = 1e5

    result_11 = continuation.identify(
        omega=omega,
        num_steps=11
    )

    result_51 = continuation.identify(
        omega=omega,
        num_steps=51
    )

    for name in ("P1", "P2", "T"):

        assert result_11[name] == pytest.approx(
            result_51[name],
            rel=1e-7
        )


def test_original_medium_unchanged(
    continuation,
    model
):
    """Verifica a preservação dos dados originais."""

    original_medium = model.medium
    original_thermal = model.medium.thermal

    continuation.identify(
        omega=1e3,
        num_steps=11
    )

    assert model.medium is original_medium
    assert model.medium.thermal is original_thermal

    assert original_thermal.beta_s == 150000.0
    assert original_thermal.beta_f == 50000.0


def test_intermediate_coupling(
    continuation,
    model
):
    """Verifica os coeficientes do acoplamento parcial."""

    factor = 0.4

    partial_model = continuation._build_model(
        factor
    )

    original = model.medium.thermal
    partial = partial_model.medium.thermal

    assert partial.beta_s == pytest.approx(
        factor * original.beta_s
    )

    assert partial.beta_f == pytest.approx(
        factor * original.beta_f
    )

    assert partial_model.medium.beta() == pytest.approx(
        factor * model.medium.beta()
    )

    assert partial.C == original.C
    assert partial.kappa == original.kappa
    assert partial.tau0 == original.tau0

    assert (
        partial_model.medium.mechanical
        is model.medium.mechanical
    )

    assert partial_model.resistance is model.resistance


@pytest.mark.parametrize(
    "omega",
    [
        0.0,
        -100.0,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_frequency(
    continuation,
    omega
):
    """Rejeita frequências inválidas."""

    with pytest.raises(ValueError):
        continuation.identify(
            omega=omega,
            num_steps=11
        )


@pytest.mark.parametrize(
    "num_steps",
    [
        0,
        1,
        -1,
        True,
        2.5,
    ],
)
def test_invalid_step_count(
    continuation,
    num_steps
):
    """Rejeita quantidades inválidas de passos."""

    with pytest.raises(ValueError):
        continuation.identify(
            omega=1e3,
            num_steps=num_steps
        )


def test_invalid_model():
    """Rejeita modelos incompatíveis."""

    with pytest.raises(TypeError):
        CouplingContinuation(model=None)


@pytest.mark.parametrize(
    "tolerance",
    [
        0.0,
        -1e-8,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_residual_tolerance(
    model,
    tolerance
):
    """Rejeita tolerâncias residuais inválidas."""

    with pytest.raises(ValueError):
        CouplingContinuation(
            model=model,
            residual_tolerance=tolerance
        )


def test_ambiguous_tracking(
    continuation,
    monkeypatch
):
    """Interrompe uma continuação ambígua."""

    def fail_tracking(previous, current):
        raise RuntimeError(
            "Associação ambígua."
        )

    monkeypatch.setattr(
        continuation.tracker,
        "track",
        fail_tracking
    )

    with pytest.raises(
        RuntimeError,
        match="Continuação ambígua"
    ):
        continuation.identify(
            omega=1e3,
            num_steps=3
        )


def test_excessive_residual(
    continuation,
    monkeypatch
):
    """Rejeita raízes com resíduos excessivos."""

    def invalid_residuals(coefficients, roots):
        return np.array([
            1e-2,
            1e-2,
            1e-2,
        ])

    monkeypatch.setattr(
        continuation.solver,
        "check_residuals",
        invalid_residuals
    )

    with pytest.raises(
        ArithmeticError,
        match="Resíduos excessivos"
    ):
        continuation.identify(
            omega=1e3,
            num_steps=11
        )
