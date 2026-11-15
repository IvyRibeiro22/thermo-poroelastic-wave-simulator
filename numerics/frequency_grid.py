
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrequencyGrid:
    """
    Representa a malha de frequências angulares da simulação.

    Todas as frequências são expressas em rad/s.
    """

    omega: np.ndarray

    def __post_init__(self) -> None:
        """Valida a malha e protege seus valores."""

        values = np.asarray(self.omega, dtype=float)

        if values.ndim != 1 or values.size < 2:
            raise ValueError(
                "A malha deve conter pelo menos duas frequências."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "As frequências devem ser finitas."
            )

        if np.any(values <= 0.0):
            raise ValueError(
                "As frequências devem ser positivas."
            )

        if np.any(np.diff(values) <= 0.0):
            raise ValueError(
                "As frequências devem estar em ordem crescente."
            )

        # Mantém uma cópia protegida da malha numérica.
        values = values.copy()
        values.setflags(write=False)

        object.__setattr__(self, "omega", values)

    @classmethod
    def build(
        cls,
        omega_min: float = 1.0,
        omega_max: float = 1e8,
        num_points: int = 1601,
        use_logspace: bool = True,
        characteristic_omegas: tuple[float, ...] = (),
    ) -> "FrequencyGrid":
        """
        Constrói uma malha linear ou logarítmica.

        Pode adicionar pontos próximos às frequências
        características dos fenômenos físicos.
        """

        if (
            not np.isfinite(omega_min)
            or not np.isfinite(omega_max)
            or omega_min <= 0.0
            or omega_max <= omega_min
        ):
            raise ValueError(
                "Os limites de frequência devem ser positivos "
                "e omega_max deve superar omega_min."
            )

        if (
            not isinstance(num_points, int)
            or isinstance(num_points, bool)
            or num_points < 2
        ):
            raise ValueError(
                "num_points deve ser um inteiro maior ou igual a 2."
            )

        if not isinstance(use_logspace, bool):
            raise TypeError(
                "use_logspace deve ser True ou False."
            )

        # Gera a distribuição inicial de frequências.
        if use_logspace:
            omega = np.geomspace(
                omega_min, omega_max, num_points
            )
        else:
            omega = np.linspace(
                omega_min, omega_max, num_points
            )

        # Acrescenta pontos perto das transições físicas.
        extra_points = []

        for characteristic in characteristic_omegas:

            if (
                not np.isfinite(characteristic)
                or characteristic <= 0.0
            ):
                raise ValueError(
                    "As frequências características devem "
                    "ser finitas e positivas."
                )

            for factor in (0.9, 1.0, 1.1):
                value = characteristic * factor

                if omega_min <= value <= omega_max:
                    extra_points.append(value)

        if extra_points:
            omega = np.unique(
                np.concatenate([omega, extra_points])
            )

        return cls(omega=omega)
