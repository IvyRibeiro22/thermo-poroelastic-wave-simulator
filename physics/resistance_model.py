
from abc import ABC, abstractmethod
from math import isfinite

from models.porous_medium import PorousMedium


class ResistanceModel(ABC):
    """
    Classe abstrata para os modelos de resistência viscosa.

    Define a interface utilizada pelos modelos de Darcy
    e Biot-JKD na formulação termo-poroelástica.
    """

    @abstractmethod
    def G(self, omega: float, medium: PorousMedium) -> complex:
        """
        Calcula a função de resistência G(ω).

        Parâmetros:
            omega: frequência angular (rad/s).
            medium: propriedades do meio poroso.

        Retorno:
            Resistência viscosa, possivelmente complexa
            (Pa·s/m²).
        """
        pass

    def R_omega(
        self,
        omega: float,
        medium: PorousMedium
    ) -> complex:
        """
        Calcula o coeficiente inercial dinâmico R(ω).

        Expressão:
            R(ω) = rho_w + i*G(ω)/omega

        Unidade: kg/m³.
        """

        self._validate_inputs(omega, medium)

        rho_w = medium.rho_w()
        resistance = self.G(omega, medium)

        # Em Python, 1j representa a unidade imaginária.
        return rho_w + (1j * resistance / omega)

    @staticmethod
    def _validate_inputs(
        omega: float,
        medium: PorousMedium
    ) -> None:
        """Valida a frequência e o objeto do meio poroso."""

        if not isfinite(omega) or omega <= 0.0:
            raise ValueError(
                "A frequência angular deve ser finita e positiva."
            )

        if not isinstance(medium, PorousMedium):
            raise TypeError(
                "medium deve ser um objeto PorousMedium."
            )
