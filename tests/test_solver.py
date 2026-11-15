
import numpy as np
import pytest

from models.mechanical_properties import MechanicalProperties
from models.thermal_properties import ThermalProperties
from models.porous_medium import PorousMedium

from physics.darcy_resistance import DarcyResistance
from physics.biot_jkd_resistance import BiotJKDResistance
from physics.reciprocal_r_model import ReciprocalRModel

from numerics.polynomial_root_solver import PolynomialRootSolver


@pytest.fixture
def solver() -> PolynomialRootSolver:
    """Cria o solucionador de raízes cúbicas."""
    return PolynomialRootSolver()


@pytest.fixture
def medium() -> PorousMedium:
    """Cria um meio poroso de referência para os testes."""

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


# ============================================================
# TESTE 1: RAÍZES REAIS CONHECIDAS
# ============================================================

def test_known_real_roots(solver):
    """Verifica as raízes do polinômio z³ - 6z² + 11z - 6."""

    coefficients = [1.0, -6.0, 11.0, -6.0]

    roots = solver.solve(coefficients)

    # As raízes esperadas são 1, 2 e 3.
    result = sorted(roots, key=lambda z: z.real)

    np.testing.assert_allclose(
        result,
        [1.0, 2.0, 3.0],
        rtol=1e-10,
        atol=1e-10,
    )


# ============================================================
# TESTE 2: RAÍZES COMPLEXAS
# ============================================================

def test_known_complex_roots(solver):
    """Verifica a solução de um polinômio complexo."""

    expected_roots = np.array([
        1.0 + 2.0j,
        2.0 - 3.0j,
        4.0 + 0.5j,
    ])

    # np.poly constrói um polinômio a partir das raízes.
    coefficients = np.poly(expected_roots)

    calculated_roots = solver.solve(coefficients)

    # A ordem das raízes retornadas não é garantida.
    result = sorted(calculated_roots, key=lambda z: z.real)
    expected = sorted(expected_roots, key=lambda z: z.real)

    np.testing.assert_allclose(
        result,
        expected,
        rtol=1e-10,
        atol=1e-10,
    )


# ============================================================
# TESTE 3: COEFICIENTES INVÁLIDOS
# ============================================================

@pytest.mark.parametrize(
    "coefficients, error",
    [
        ([0, 1, 2, 3], ValueError),
        ([1, 2, 3], ValueError),
        ([1, float("inf"), 2, 3], ValueError),
        (["invalido", 1, 2, 3], TypeError),
    ],
)
def test_invalid_coefficients(solver, coefficients, error):
    """Rejeita coeficientes que não formam um cúbico válido."""

    with pytest.raises(error):
        solver.solve(coefficients)


# ============================================================
# TESTE 4: RESÍDUOS NUMÉRICOS
# ============================================================

def test_root_residuals(solver):
    """Verifica se as raízes satisfazem o polinômio."""

    coefficients = [1.0, -6.0, 11.0, -6.0]

    roots = solver.solve(coefficients)
    residuals = solver.check_residuals(coefficients, roots)

    assert residuals.shape == (3,)
    assert np.all(np.isfinite(residuals))
    assert np.max(residuals) < 1e-12


# ============================================================
# TESTE 5: VALIDAÇÃO DAS RAÍZES RECEBIDAS
# ============================================================

def test_invalid_roots(solver):
    """Rejeita uma quantidade incorreta de raízes."""

    coefficients = [1.0, -6.0, 11.0, -6.0]

    with pytest.raises(ValueError):
        solver.check_residuals(
            coefficients,
            [1.0, 2.0],
        )


# ============================================================
# TESTE 6: INTEGRAÇÃO COM O MODELO R
# ============================================================

@pytest.mark.parametrize(
    "resistance_type",
    [DarcyResistance, BiotJKDResistance],
    ids=["Darcy", "Biot-JKD"],
)
@pytest.mark.parametrize(
    "omega",
    [1e3, 1e6],
)
def test_reciprocal_model_roots(
    solver,
    medium,
    resistance_type,
    omega,
):
    """
    Verifica as raízes da equação característica R
    para diferentes frequências e resistências.
    """

    resistance = resistance_type()

    model = ReciprocalRModel(
        medium=medium,
        resistance=resistance,
    )

    coefficients = model.polynomial_coefficients(omega)

    roots = solver.solve(coefficients)
    residuals = solver.check_residuals(coefficients, roots)

    assert roots.shape == (3,)
    assert np.all(np.isfinite(roots))

    # Resíduo relativo máximo admitido no teste.
    assert np.max(residuals) < 1e-10
