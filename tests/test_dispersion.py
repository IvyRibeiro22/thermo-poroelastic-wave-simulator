
import pytest
import numpy as np

from cmath import sqrt

from models.mechanical_properties import MechanicalProperties
from models.thermal_properties import ThermalProperties
from models.porous_medium import PorousMedium

from physics.darcy_resistance import DarcyResistance
from physics.biot_jkd_resistance import BiotJKDResistance
from physics.reciprocal_r_model import ReciprocalRModel


@pytest.fixture
def medium() -> PorousMedium:
    """Cria um meio poroso com propriedades de referência."""

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

    return PorousMedium(mechanical, thermal)


@pytest.fixture(
    params=[DarcyResistance, BiotJKDResistance],
    ids=["Darcy", "Biot-JKD"]
)
def model(medium, request) -> ReciprocalRModel:
    """Executa os mesmos testes para os dois modelos."""

    resistance = request.param()

    return ReciprocalRModel(
        medium=medium,
        resistance=resistance,
    )


def test_cubic_polynomial(model):
    """Verifica o grau e a finitude dos coeficientes."""

    coefficients = model.polynomial_coefficients(1000.0)

    assert coefficients.shape == (4,)
    assert np.all(np.isfinite(coefficients))
    assert abs(coefficients[0]) > 0.0


@pytest.mark.parametrize("omega", [1e3, 1e6])
def test_extreme_coefficients(model, omega):
    """Verifica os coeficientes cúbico e constante."""

    medium = model.medium
    p = medium.mechanical
    t = medium.thermal

    R = model.resistance.R_omega(omega, medium)
    tw = model.t_omega(omega)

    # Coeficiente de z³.
    expected_c3 = -t.C * tw * (
        medium.rho() * R - p.rho_f ** 2
    )

    # Coeficiente constante.
    expected_c0 = t.kappa * (
        medium.A() * p.m - medium.B() ** 2
    )

    coefficients = model.polynomial_coefficients(omega)

    assert coefficients[0] == pytest.approx(expected_c3)
    assert coefficients[-1] == pytest.approx(expected_c0)


@pytest.mark.parametrize("omega", [1e3, 1e6])
def test_matrix_determinant(model, omega):
    """Verifica det(M) = P_R(z; omega)."""

    # Valor complexo ilustrativo de z = (omega/k)².
    z = 2e6 + 3e5j
    k = omega / sqrt(z)

    matrix = model.dispersion_matrix(omega, k)
    coefficients = model.polynomial_coefficients(omega)

    # Duas formas de calcular a equação característica.
    determinant = np.linalg.det(matrix)
    polynomial_value = np.polyval(coefficients, z)

    assert matrix.shape == (3, 3)

    assert determinant == pytest.approx(
        polynomial_value,
        rel=1e-10
    )


def test_invalid_frequency(model):
    """Rejeita frequências angulares inválidas."""

    for omega in (0.0, -1.0, float("inf")):
        with pytest.raises(ValueError):
            model.polynomial_coefficients(omega)


def test_invalid_wavenumber(model):
    """Rejeita números de onda nulos ou não finitos."""

    with pytest.raises(ValueError):
        model.dispersion_matrix(1000.0, 0j)

    with pytest.raises(ValueError):
        model.dispersion_matrix(
            1000.0,
            complex(float("inf"), 0.0)
        )


def test_invalid_model_components(medium):
    """Verifica os tipos exigidos pelo modelo físico."""

    with pytest.raises(TypeError):
        ReciprocalRModel(
            medium=None,
            resistance=DarcyResistance()
        )

    with pytest.raises(TypeError):
        ReciprocalRModel(
            medium=medium,
            resistance=None
        )
