
from dataclasses import dataclass
from cmath import sqrt
from math import isfinite


@dataclass(frozen=True)
class ModeState:
    """
    Representa o estado de um ramo de onda em uma frequência.

    Armazena a raiz complexa e as propriedades de
    propagação calculadas a partir dela.
    """

    branch_index: int
    omega: float
    z: complex
    k: complex
    phase_velocity: float
    attenuation: float
    q_inverse: float

    @classmethod
    def from_root(
        cls,
        branch_index: int,
        omega: float,
        z: complex
    ) -> "ModeState":
        """Calcula as propriedades físicas a partir de z."""

        if (
            type(branch_index) is not int
            or branch_index not in (0, 1, 2)
        ):
            raise ValueError(
                "O ramo deve ser identificado por 0, 1 ou 2."
            )

        if not isfinite(omega) or omega <= 0.0:
            raise ValueError(
                "A frequência angular deve ser finita e positiva."
            )

        try:
            z = complex(z)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "A raiz z deve ser um número complexo."
            ) from exc

        if (
            not isfinite(z.real)
            or not isfinite(z.imag)
            or z == 0
        ):
            raise ValueError(
                "A raiz z deve ser finita e diferente de zero."
            )

        # Número de onda associado à raiz complexa.
        k = omega / sqrt(z)

        # Seleciona o sentido de propagação positivo.
        if k.real < 0.0:
            k = -k

        if not isfinite(k.real) or not isfinite(k.imag):
            raise ValueError(
                "O número de onda calculado não é finito."
            )

        if k.real <= 0.0:
            raise ValueError(
                "O número de onda não representa "
                "propagação no sentido positivo de x."
            )

        if k.imag < 0.0:
            raise ValueError(
                "A raiz produz atenuação espacial negativa. "
                "Verifique a solução e a convenção de sinais."
            )

        # Propriedades de dispersão e atenuação.
        phase_velocity = omega / k.real
        attenuation = k.imag
        q_inverse = 2.0 * k.imag / k.real

        if not all(
            isfinite(value)
            for value in (
                phase_velocity,
                attenuation,
                q_inverse
            )
        ):
            raise ValueError(
                "As propriedades de propagação não são finitas."
            )

        return cls(
            branch_index=branch_index,
            omega=omega,
            z=z,
            k=k,
            phase_velocity=phase_velocity,
            attenuation=attenuation,
            q_inverse=q_inverse,
        )
