
from cmath import sqrt
from math import pi

from models.porous_medium import PorousMedium
from physics.resistance_model import ResistanceModel


class BiotJKDResistance(ResistanceModel):
    """
    Implementa a resistência viscosa dinâmica Biot-JKD.

    G(omega) = b0 * sqrt(1 - i*omega/Omega)

    A resistência depende da frequência angular da onda.
    """

    def fc(self, medium: PorousMedium) -> float:
        """Calcula a frequência característica fc (Hz)."""

        if not isinstance(medium, PorousMedium):
            raise TypeError(
                "medium deve ser um objeto PorousMedium."
            )

        return medium.b0() / (2.0 * pi * medium.rho_w())

    def pride_number(self, medium: PorousMedium) -> float:
        """Calcula o parâmetro adimensional P."""

        if not isinstance(medium, PorousMedium):
            raise TypeError(
                "medium deve ser um objeto PorousMedium."
            )

        p = medium.mechanical

        return (
            4.0 * p.a_inf * p.k0
            / (p.phi * p.viscous_length ** 2)
        )

    def omega_relaxation(self, medium: PorousMedium) -> float:
        """Calcula a frequência angular Omega (rad/s)."""

        fc = self.fc(medium)
        P = self.pride_number(medium)

        return 2.0 * pi * fc / P

    def G(self, omega: float, medium: PorousMedium) -> complex:
        """
        Calcula a resistência dinâmica G(omega).

        Unidade: Pa.s/m².
        """

        self._validate_inputs(omega, medium)

        b0 = medium.b0()
        omega_c = self.omega_relaxation(medium)

        # Raiz complexa principal, com parte real positiva.
        return b0 * sqrt(1.0 - 1j * omega / omega_c)
