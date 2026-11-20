
from itertools import permutations

import numpy as np
import pytest

from models.mechanical_properties import MechanicalProperties
from models.thermal_properties import ThermalProperties
from models.porous_medium import PorousMedium

from physics.darcy_resistance import DarcyResistance
from physics.biot_jkd_resistance import BiotJKDResistance
from physics.reciprocal_r_model import ReciprocalRModel

from numerics.polynomial_root_solver import PolynomialRootSolver


def create_uncoupled_medium(C=2.3e6):
    """Cria um meio sem acoplamento termomecânico."""

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
        C=C,
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


def analytical_roots(medium, resistance, omega):
    """Calcula as raízes do sistema desacoplado."""

    p = medium.mechanical
    t = medium.thermal

    rho = medium.rho()
    A = medium.A()
    B = medium.B()

    R = resistance.R_omega(omega, medium)

    # Coeficientes do polinômio poroelástico quadrático.
    c2 = rho * R - p.rho_f**2

    c1 = -(
        A * R
        + rho * p.m
        - 2.0 * B * p.rho_f
    )

    c0 = A * p.m - B**2

    mechanical_roots = np.roots([
        c2, c1, c0
    ])

    # Solução analítica do ramo térmico.
    thermal_root = t.kappa / (
        t.C * (t.tau0 + 1j / omega)
    )

    return np.array([
        mechanical_roots[0],
        mechanical_roots[1],
        thermal_root,
    ], dtype=complex)


def relative_matching_error(calculated, expected):
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


@pytest.mark.parametrize(
    "resistance_type",
    [DarcyResistance, BiotJKDResistance],
    ids=["Darcy", "Biot-JKD"],
)
@pytest.mark.parametrize(
    "omega",
    [1e3, 1e6],
)
def test_analytical_decoupling(
    resistance_type,
    omega
):
    """Compara as três raízes com a solução analítica."""

    medium = create_uncoupled_medium()
    resistance = resistance_type()

    model = ReciprocalRModel(
        medium=medium,
        resistance=resistance,
    )

    solver = PolynomialRootSolver()

    coefficients = model.polynomial_coefficients(omega)

    calculated = solver.solve(coefficients)

    expected = analytical_roots(
        medium,
        resistance,
        omega,
    )

    error = relative_matching_error(
        calculated,
        expected,
    )

    assert error < 1e-7


@pytest.mark.parametrize(
    "resistance_type",
    [DarcyResistance, BiotJKDResistance],
    ids=["Darcy", "Biot-JKD"],
)
def test_heat_capacity_decoupled(
    resistance_type
):
    """Verifica o efeito de C no sistema desacoplado."""

    omega = 1e5

    medium_1 = create_uncoupled_medium(C=2.3e6)
    medium_2 = create_uncoupled_medium(C=4.6e6)

    resistance = resistance_type()
    solver = PolynomialRootSolver()

    model_1 = ReciprocalRModel(
        medium=medium_1,
        resistance=resistance,
    )

    model_2 = ReciprocalRModel(
        medium=medium_2,
        resistance=resistance,
    )

    roots_1 = solver.solve(
        model_1.polynomial_coefficients(omega)
    )

    roots_2 = solver.solve(
        model_2.polynomial_coefficients(omega)
    )

    expected_1 = analytical_roots(
        medium_1,
        resistance,
        omega,
    )

    expected_2 = analytical_roots(
        medium_2,
        resistance,
        omega,
    )

    assert relative_matching_error(
        roots_1, expected_1
    ) < 1e-7

    assert relative_matching_error(
        roots_2, expected_2
    ) < 1e-7

    # As raízes mecânicas não dependem de C
    # quando o acoplamento térmico é nulo.
    np.testing.assert_allclose(
        np.sort_complex(expected_1[:2]),
        np.sort_complex(expected_2[:2]),
        rtol=1e-10,
        atol=0.0,
    )

    # Ao dobrar C, a raiz térmica é reduzida à metade.
    assert expected_2[2] == pytest.approx(
        expected_1[2] / 2.0,
        rel=1e-10,
    )
