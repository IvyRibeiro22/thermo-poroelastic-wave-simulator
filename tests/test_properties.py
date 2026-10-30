
import pytest
from dataclasses import FrozenInstanceError

from models.mechanical_properties import MechanicalProperties


def create_valid_properties():
    """Cria um conjunto ilustrativo de propriedades válidas."""

    return MechanicalProperties(
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


def test_valid_properties():
    """Verifica a criação de um meio com parâmetros válidos."""

    properties = create_valid_properties()

    assert properties.phi == 0.25
    assert properties.rho_s == 2650.0
    assert properties.k0 == 1e-12


def test_invalid_porosity():
    """Verifica a rejeição de uma porosidade impossível."""

    properties = create_valid_properties()

    with pytest.raises(ValueError):
        MechanicalProperties(
            **{
                **vars(properties),
                "phi": 1.5,
            }
        )


def test_negative_viscosity():
    """Verifica a rejeição de uma viscosidade negativa."""

    properties = create_valid_properties()

    with pytest.raises(ValueError):
        MechanicalProperties(
            
            **{
                **vars(properties),
                "eta": -0.001,
            }
        )


def test_immutable_properties():
    """Verifica que as propriedades não podem ser alteradas."""

    properties = create_valid_properties()

    with pytest.raises(FrozenInstanceError):
        properties.phi = 0.40
