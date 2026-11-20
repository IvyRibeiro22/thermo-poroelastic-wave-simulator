
from cmath import sqrt
from itertools import permutations
from math import isfinite

import numpy as np

from physics.reciprocal_r_model import ReciprocalRModel


class ModeClassifier:
    """Identifica P1, P2 e T no limite desacoplado."""

    def __init__(
        self,
        model: ReciprocalRModel,
        relative_tolerance: float = 1e-7
    ) -> None:

        if not isinstance(model, ReciprocalRModel):
            raise TypeError(
                "O modelo deve ser ReciprocalRModel."
            )

        if (
            not isfinite(relative_tolerance)
            or not 0.0 < relative_tolerance < 1.0
        ):
            raise ValueError(
                "A tolerância deve estar entre 0 e 1."
            )

        self.model = model
        self.relative_tolerance = relative_tolerance

    @staticmethod
    def _phase_velocity(
        omega: float,
        z: complex
    ) -> float:
        """Calcula a velocidade de fase de uma raiz."""

        if z == 0:
            raise ValueError(
                "A raiz não pode ser zero."
            )

        k = omega / sqrt(complex(z))

        if k.real <= 0.0 or k.imag < 0.0:
            raise ValueError(
                "A raiz é incompatível com a propagação adotada."
            )

        velocity = omega / k.real

        if not isfinite(velocity):
            raise ValueError(
                "A velocidade de fase não é finita."
            )

        return velocity

    def classify_uncoupled(
        self,
        omega: float,
        roots
    ) -> dict[str, int]:
        """Identifica os três modos do sistema desacoplado."""

        if not isfinite(omega) or omega <= 0.0:
            raise ValueError(
                "A frequência deve ser finita e positiva."
            )

        try:
            values = np.asarray(roots, dtype=complex)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "As raízes devem ser numéricas."
            ) from exc

        if (
            values.shape != (3,)
            or not np.all(np.isfinite(values))
        ):
            raise ValueError(
                "São necessárias três raízes finitas."
            )

        medium = self.model.medium
        p = medium.mechanical
        t = medium.thermal

        # Exige desacoplamento termomecânico completo.
        if t.beta_s != 0.0 or t.beta_f != 0.0:
            raise ValueError(
                "A classificação requer acoplamento térmico nulo."
            )

        rho = medium.rho()
        A = medium.A()
        B = medium.B()

        R = self.model.resistance.R_omega(
            omega,
            medium
        )

        # Polinômio poroelástico quadrático.
        c2 = rho * R - p.rho_f**2

        c1 = -(
            A * R
            + rho * p.m
            - 2.0 * B * p.rho_f
        )

        c0 = A * p.m - B**2

        mechanical_roots = np.roots([
            c2, c1, c0
        ])

        # P1 e P2 são os dois modos poroelásticos.
        mechanical_roots = sorted(
            mechanical_roots,
            key=lambda z: self._phase_velocity(
                omega, z
            ),
            reverse=True
        )

        # Solução independente do modo térmico.
        thermal_root = t.kappa / (
            t.C * self.model.t_omega(omega)
        )

        reference = (
            mechanical_roots[0],
            mechanical_roots[1],
            thermal_root
        )

        candidates = []

        # Compara as seis associações possíveis.
        for order in permutations(range(3)):

            errors = [
                abs(values[order[i]] - reference[i])
                / max(
                    abs(reference[i]),
                    np.finfo(float).tiny
                )
                for i in range(3)
            ]

            candidates.append((
                max(errors),
                order
            ))

        candidates.sort(
            key=lambda item: item[0]
        )

        best_error, best_order = candidates[0]
        second_error = candidates[1][0]

        if best_error > self.relative_tolerance:
            raise ValueError(
                "As raízes não correspondem "
                "ao sistema desacoplado."
            )

        if (
            second_error - best_error
            <= self.relative_tolerance
        ):
            raise RuntimeError(
                "A identificação dos modos é ambígua."
            )

        return dict(zip(
            ("P1", "P2", "T"),
            best_order
        ))
