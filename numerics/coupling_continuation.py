
from dataclasses import replace
from math import isfinite

import numpy as np

from models.porous_medium import PorousMedium

from physics.reciprocal_r_model import ReciprocalRModel

from numerics.mode_classifier import ModeClassifier
from numerics.mode_tracker import ModeTracker
from numerics.polynomial_root_solver import PolynomialRootSolver


class CouplingContinuation:
    """Acompanha P1, P2 e T durante o acoplamento térmico."""

    def __init__(
        self,
        model: ReciprocalRModel,
        residual_tolerance: float = 1e-8
    ) -> None:

        if not isinstance(model, ReciprocalRModel):
            raise TypeError(
                "O modelo deve ser ReciprocalRModel."
            )

        if (
            not isfinite(residual_tolerance)
            or residual_tolerance <= 0.0
        ):
            raise ValueError(
                "A tolerância residual deve ser finita e positiva."
            )

        self.model = model
        self.residual_tolerance = residual_tolerance

        self.solver = PolynomialRootSolver()
        self.tracker = ModeTracker()

    def _build_model(
        self,
        factor: float
    ) -> ReciprocalRModel:
        """Constrói o modelo com acoplamento parcial."""

        original_medium = self.model.medium
        original_thermal = original_medium.thermal

        # Escala os coeficientes térmicos.
        thermal = replace(
            original_thermal,
            beta_s=factor * original_thermal.beta_s,
            beta_f=factor * original_thermal.beta_f,
        )

        # Preserva as propriedades mecânicas.
        medium = PorousMedium(
            mechanical=original_medium.mechanical,
            thermal=thermal,
        )

        # Preserva a lei de resistência.
        return ReciprocalRModel(
            medium=medium,
            resistance=self.model.resistance,
        )

    def _solve_roots(
        self,
        model: ReciprocalRModel,
        omega: float
    ) -> np.ndarray:
        """Calcula e verifica as três raízes."""

        coefficients = model.polynomial_coefficients(
            omega
        )

        roots = self.solver.solve(coefficients)

        residuals = self.solver.check_residuals(
            coefficients,
            roots
        )

        if (
            not np.all(np.isfinite(residuals))
            or np.max(residuals) > self.residual_tolerance
        ):
            raise ArithmeticError(
                "Resíduos excessivos na continuação."
            )

        return roots

    def identify(
        self,
        omega: float,
        num_steps: int = 51
    ) -> dict[str, complex]:
        """
        Acompanha os modos do sistema desacoplado
        até o acoplamento térmico completo.
        """

        if not isfinite(omega) or omega <= 0.0:
            raise ValueError(
                "A frequência deve ser finita e positiva."
            )

        if type(num_steps) is not int or num_steps < 2:
            raise ValueError(
                "A continuação exige ao menos dois pontos."
            )

        # Etapa inicial: acoplamento nulo.
        initial_model = self._build_model(0.0)

        initial_roots = self._solve_roots(
            initial_model,
            omega
        )

        # Identifica os modos na referência desacoplada.
        classifier = ModeClassifier(initial_model)

        mapping = classifier.classify_uncoupled(
            omega,
            initial_roots
        )

        mode_names = ("P1", "P2", "T")

        ordered_roots = np.array(
            [
                initial_roots[mapping[name]]
                for name in mode_names
            ],
            dtype=complex
        )

        # Aumenta gradualmente o acoplamento térmico.
        factors = np.linspace(
            0.0,
            1.0,
            num_steps
        )

        for factor in factors[1:]:

            current_model = self._build_model(
                float(factor)
            )

            current_roots = self._solve_roots(
                current_model,
                omega
            )

            # Preserva a correspondência entre os ramos.
            try:
                ordered_roots = self.tracker.track(
                    ordered_roots,
                    current_roots
                )
            except RuntimeError as exc:
                raise RuntimeError(
                    f"Continuação ambígua em s={factor:.6g}. "
                    "Refine a discretização do acoplamento."
                ) from exc

        # Retorna as raízes do sistema totalmente acoplado.
        return {
            name: complex(ordered_roots[index])
            for index, name in enumerate(mode_names)
        }
