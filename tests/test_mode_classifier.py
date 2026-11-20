
from cmath import sqrt
from dataclasses import replace

import numpy as np
import pytest

from models.mechanical_properties import MechanicalProperties
from models.thermal_properties import ThermalProperties
from models.porous_medium import PorousMedium

from physics.darcy_resistance import DarcyResistance
from physics.biot_jkd_resistance import BiotJKDResistance
from physics.reciprocal_r_model import ReciprocalRModel

from numerics.mode_classifier import ModeClassifier
from numerics.polynomial_root_solver import PolynomialRootSolver


@pytest.fixture
def medium():
    """Cria o meio termo-poroelástico desacoplado."""

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
        beta_s=0.0,
        beta_f=0.0,
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
    """Cria os dois modelos de resistência."""

    return ReciprocalRModel(
        medium=medium,
        resistance=request.param(),
    )


@pytest.fixture
def classifier(medium):
    """Cria o classificador de referência."""

    model = ReciprocalRModel(
        medium=medium,
        resistance=DarcyResistance(),
    )

    return ModeClassifier(model)


def calculate_roots(model, omega):
    """Calcula as três raízes características."""

    solver = PolynomialRootSolver()

    coefficients = model.polynomial_coefficients(
        omega
    )

    return solver.solve(coefficients)


def phase_velocity(omega, z):
    """Calcula a velocidade de fase de referência."""

    k = omega / sqrt(complex(z))

    return omega / k.real


@pytest.mark.parametrize("omega", [1e3, 1e6])
def test_identification_of_three_modes(model, omega):
    """Identifica P1, P2 e T nas duas resistências."""

    roots = calculate_roots(model, omega)

    classifier = ModeClassifier(model)

    modes = classifier.classify_uncoupled(
        omega=omega,
        roots=roots,
    )

    assert set(modes) == {"P1", "P2", "T"}
    assert set(modes.values()) == {0, 1, 2}

    p1_root = roots[modes["P1"]]
    p2_root = roots[modes["P2"]]
    thermal_root = roots[modes["T"]]

    # P1 apresenta maior velocidade que P2.
    assert phase_velocity(
        omega, p1_root
    ) > phase_velocity(
        omega, p2_root
    )

    # O modo T satisfaz a equação térmica.
    thermal = model.medium.thermal

    expected_thermal_root = thermal.kappa / (
        thermal.C * model.t_omega(omega)
    )

    assert thermal_root == pytest.approx(
        expected_thermal_root,
        rel=1e-7,
    )


def test_identification_independent_of_root_order(model):
    """Preserva a identificação após permutar as raízes."""

    omega = 1e5

    roots = calculate_roots(model, omega)
    reordered = roots[[2, 0, 1]]

    classifier = ModeClassifier(model)

    original_modes = classifier.classify_uncoupled(
        omega,
        roots,
    )

    reordered_modes = classifier.classify_uncoupled(
        omega,
        reordered,
    )

    for name in ("P1", "P2", "T"):
        original_root = roots[
            original_modes[name]
        ]

        reordered_root = reordered[
            reordered_modes[name]
        ]

        assert reordered_root == pytest.approx(
            original_root,
            rel=1e-10,
        )


def test_rejects_coupled_medium(medium):
    """Rejeita a classificação direta do modelo acoplado."""

    coupled_thermal = replace(
        medium.thermal,
        beta_s=150000.0,
        beta_f=50000.0,
    )

    coupled_medium = PorousMedium(
        mechanical=medium.mechanical,
        thermal=coupled_thermal,
    )

    model = ReciprocalRModel(
        medium=coupled_medium,
        resistance=DarcyResistance(),
    )

    omega = 1e3
    roots = calculate_roots(model, omega)

    classifier = ModeClassifier(model)

    with pytest.raises(ValueError, match="acoplamento"):
        classifier.classify_uncoupled(
            omega,
            roots,
        )


@pytest.mark.parametrize(
    "omega",
    [
        0.0,
        -100.0,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_frequency(classifier, omega):
    """Rejeita frequências inválidas."""

    roots = [4e6, 1e6, 2.5e5]

    with pytest.raises(ValueError):
        classifier.classify_uncoupled(
            omega,
            roots,
        )


@pytest.mark.parametrize(
    "roots",
    [
        [1.0, 2.0],
        [1.0, 2.0, float("nan")],
        [1.0, 2.0, float("inf")],
    ],
)
def test_invalid_roots(classifier, roots):
    """Rejeita conjuntos de raízes inválidos."""

    with pytest.raises(ValueError):
        classifier.classify_uncoupled(
            1000.0,
            roots,
        )


def test_invalid_model():
    """Rejeita um modelo físico inválido."""

    with pytest.raises(TypeError):
        ModeClassifier(model=None)


@pytest.mark.parametrize(
    "tolerance",
    [
        -0.1,
        0.0,
        1.0,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_tolerance(medium, tolerance):
    """Rejeita tolerâncias inválidas."""

    model = ReciprocalRModel(
        medium=medium,
        resistance=DarcyResistance(),
    )

    with pytest.raises(ValueError):
        ModeClassifier(
            model=model,
            relative_tolerance=tolerance,
        )


def test_roots_do_not_match_reference(classifier):
    """Rejeita raízes incompatíveis com a referência."""

    with pytest.raises(ValueError):
        classifier.classify_uncoupled(
            omega=1000.0,
            roots=[1e6, 2e6, 3e6],
        )
