
import pytest

from models.mechanical_properties import MechanicalProperties
from models.thermal_properties import ThermalProperties
from models.porous_medium import PorousMedium

from physics.darcy_resistance import DarcyResistance
from physics.biot_jkd_resistance import BiotJKDResistance


def create_valid_medium() -> PorousMedium:
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

    return PorousMedium(
        mechanical=mechanical,
        thermal=thermal,
    )


# ============================================================
# TESTES DO MODELO DE DARCY
# ============================================================


def test_darcy_constant_resistance():
    """Verifica se G(omega) é independente da frequência."""

    medium = create_valid_medium()
    darcy = DarcyResistance()

    # b0 = eta/k0 = 10^9 Pa.s/m².
    expected = 1e9 + 0j

    assert darcy.G(100.0, medium) == pytest.approx(expected)
    assert darcy.G(10000.0, medium) == pytest.approx(expected)


def test_darcy_dynamic_inertia():
    """Verifica R(omega) = rho_w + i*b0/omega."""

    medium = create_valid_medium()
    darcy = DarcyResistance()

    result = darcy.R_omega(1000.0, medium)

    assert result == pytest.approx(8000.0 + 1e6j)

    result_double = darcy.R_omega(2000.0, medium)

    assert result_double == pytest.approx(8000.0 + 5e5j)


@pytest.mark.parametrize(
    "omega",
    [0.0, -1.0, float("inf")]
)
def test_invalid_frequency(omega):
    """Rejeita frequências nulas, negativas ou infinitas."""

    medium = create_valid_medium()
    darcy = DarcyResistance()

    with pytest.raises(ValueError):
        darcy.G(omega, medium)


def test_invalid_medium():
    """Rejeita objetos que não representam um meio poroso."""

    darcy = DarcyResistance()

    with pytest.raises(TypeError):
        darcy.G(1000.0, None)


# ============================================================
# TESTES DO MODELO BIOT-JKD
# ============================================================


def test_jkd_characteristic_parameters():
    """Verifica fc, P e Omega para o meio de referência."""

    medium = create_valid_medium()
    jkd = BiotJKDResistance()

    # fc (Hz), P (adimensional), Omega (rad/s).
    assert jkd.fc(medium) == pytest.approx(19894.367886)
    assert jkd.pride_number(medium) == pytest.approx(0.32)
    assert jkd.omega_relaxation(medium) == pytest.approx(390625.0)


def test_jkd_resistance_at_characteristic_frequency():
    """Verifica G(omega) na frequência Omega."""

    medium = create_valid_medium()
    jkd = BiotJKDResistance()

    omega = jkd.omega_relaxation(medium)

    # Valor analítico da raiz principal de sqrt(1 - 1j).
    expected = 1e9 * (
        1.09868411346781 - 0.455089860562227j
    )

    assert jkd.G(omega, medium) == pytest.approx(expected)


def test_jkd_low_frequency_limit():
    """Verifica a aproximação do modelo JKD ao de Darcy."""

    medium = create_valid_medium()
    jkd = BiotJKDResistance()

    omega = jkd.omega_relaxation(medium) * 1e-8
    resistance = jkd.G(omega, medium)

    # Para omega << Omega, G_JKD tende a b0.
    assert resistance.real == pytest.approx(
        medium.b0(), rel=1e-7
    )

    assert abs(resistance.imag) / medium.b0() < 1e-7


def test_jkd_frequency_dependence():
    """Verifica a dependência da resistência com a frequência."""

    medium = create_valid_medium()
    jkd = BiotJKDResistance()

    omega_c = jkd.omega_relaxation(medium)

    low = jkd.G(0.1 * omega_c, medium)
    high = jkd.G(10.0 * omega_c, medium)

    assert abs(high) > abs(low)

    # Verifica o ramo principal da raiz complexa.
    assert low.real > 0.0
    assert high.real > 0.0
    assert low.imag < 0.0
    assert high.imag < 0.0


def test_jkd_dynamic_inertia():
    """Verifica R_omega herdado de ResistanceModel."""

    medium = create_valid_medium()
    jkd = BiotJKDResistance()

    omega = jkd.omega_relaxation(medium)

    expected = (
        medium.rho_w()
        + 1j * jkd.G(omega, medium) / omega
    )

    assert jkd.R_omega(omega, medium) == pytest.approx(expected)


@pytest.mark.parametrize(
    "omega",
    [0.0, -100.0, float("inf")]
)
def test_jkd_invalid_frequency(omega):
    """Rejeita frequências inválidas."""

    medium = create_valid_medium()
    jkd = BiotJKDResistance()

    with pytest.raises(ValueError):
        jkd.G(omega, medium)


def test_jkd_invalid_medium():
    """Rejeita objetos que não representam um meio poroso."""

    jkd = BiotJKDResistance()

    with pytest.raises(TypeError):
        jkd.G(1000.0, None)
