
from models.porous_medium import PorousMedium
from physics.resistance_model import ResistanceModel


class DarcyResistance(ResistanceModel):
    """
    Implementa o modelo de resistência viscosa de Darcy.

    A resistência é constante e definida por:

        G_D(omega) = eta / k0 = b0

    Unidade: Pa·s/m².
    """

    def G(self, omega: float, medium: PorousMedium) -> complex:
        """
        Retorna a resistência viscosa de Darcy.

        Args:
            omega: Frequência angular (rad/s).
            medium: Meio poroso com propriedades físicas válidas.

        Returns:
            Resistência viscosa constante, em formato complexo.
        """

        # Verifica se a frequência e o meio são válidos.
        self._validate_inputs(omega, medium)

        # b0 = eta / k0, calculado pela classe PorousMedium.
        b0 = medium.b0()

        return complex(b0)
