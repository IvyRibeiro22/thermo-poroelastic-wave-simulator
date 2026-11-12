
import pytest
from dataclasses import FrozenInstanceError

from models.mechanical_properties import MechanicalProperties
from models.thermal_properties import ThermalProperties
from models.porous_medium import PorousMedium


def create_valid_medium() -> PorousMedium:
    """Cria um meio poroso com parâmetros ilustrativos válidos."""

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


def test_derived_properties():
    """Verifica os seis coeficientes derivados do meio."""

    medium = create_valid_medium()

    # Densidade efetiva e coeficiente inercial: kg/m³.
    assert medium.rho() == pytest.approx(2237.5)
    assert medium.rho_w() == pytest.approx(8000.0)

    # Resistência viscosa de Darcy: Pa·s/m².
    assert medium.b0() == pytest.approx(1e9)

    # Acoplamento térmico: Pa/K.
    assert medium.beta() == pytest.approx(190000.0)

    # Coeficientes mecânicos: Pa.
    assert medium.A() == pytest.approx(26.4e9)
    assert medium.B() == pytest.approx(8e9)


def test_invalid_component_types():
    """Rejeita objetos que não representam as propriedades esperadas."""

    medium = create_valid_medium()

    with pytest.raises(TypeError):
        PorousMedium(
            mechanical="valores incorretos",
            thermal=medium.thermal,
        )

    with pytest.raises(TypeError):
        PorousMedium(
            mechanical=medium.mechanical,
            thermal=None,
        )


def test_immutable_medium():
    """Impede a substituição das propriedades após a criação."""

    medium = create_valid_medium()

    with pytest.raises(FrozenInstanceError):
        medium.mechanical = None
