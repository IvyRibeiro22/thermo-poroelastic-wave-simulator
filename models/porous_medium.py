
from dataclasses import dataclass

from models.mechanical_properties import MechanicalProperties
from models.thermal_properties import ThermalProperties


@dataclass(frozen=True)
class PorousMedium:
    """
    Representa um meio poroso saturado.

    Reúne os parâmetros mecânicos e térmicos e calcula
    os coeficientes utilizados no modelo termo-poroelástico.
    """

    mechanical: MechanicalProperties
    thermal: ThermalProperties

    def __post_init__(self) -> None:
        """Valida o meio após sua inicialização."""
        self.validate()

    def rho(self) -> float:
        """Calcula a densidade efetiva do meio (kg/m³)."""

        p = self.mechanical

        return (1.0 - p.phi) * p.rho_s + p.phi * p.rho_f

    def rho_w(self) -> float:
        """Calcula o coeficiente inercial do fluido relativo (kg/m³)."""

        p = self.mechanical

        return (p.a_inf * p.rho_f) / p.phi

    def b0(self) -> float:
        """Calcula a resistência viscosa de Darcy (Pa·s/m²)."""

        p = self.mechanical

        return p.eta / p.k0

    def beta(self) -> float:
        """Calcula o coeficiente de acoplamento térmico total (Pa/K)."""

        p = self.mechanical
        t = self.thermal

        return t.beta_s + p.alpha * t.beta_f

    def A(self) -> float:
        """Calcula o coeficiente compressional acoplado (Pa)."""

        p = self.mechanical

        return p.lambda_ + 2.0 * p.mu + (p.alpha ** 2) * p.m

    def B(self) -> float:
        """Calcula o coeficiente de acoplamento mecânico (Pa)."""

        p = self.mechanical

        return p.alpha * p.m

    def validate(self) -> None:
        """
        Verifica a consistência dos parâmetros e as
        condições básicas de estabilidade do meio.
        """

        if not isinstance(self.mechanical, MechanicalProperties):
            raise TypeError(
                "mechanical deve ser um objeto MechanicalProperties."
            )

        if not isinstance(self.thermal, ThermalProperties):
            raise TypeError(
                "thermal deve ser um objeto ThermalProperties."
            )

        self.mechanical.validate()
        self.thermal.validate()

        p = self.mechanical

        # Positividade do determinante da matriz de inércia.
        inertia_determinant = (
            self.rho() * self.rho_w() - p.rho_f ** 2
        )

        if inertia_determinant <= 0.0:
            raise ValueError(
                "A matriz de inércia deve ter determinante positivo."
            )

        # A*m - B² = m*(lambda + 2*mu).
        # A expressão fatorada evita cancelamento numérico.
        elastic_determinant = p.m * (p.lambda_ + 2.0 * p.mu)

        if elastic_determinant <= 0.0:
            raise ValueError(
                "A matriz de rigidez compressional deve "
                "ter determinante positivo."
            )
