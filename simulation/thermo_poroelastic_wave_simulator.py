
from math import isfinite

import numpy as np

from models.simulation_result import ModeState, SimulationResult
from numerics.frequency_grid import FrequencyGrid
from numerics.mode_tracker import ModeTracker
from numerics.polynomial_root_solver import PolynomialRootSolver
from physics.reciprocal_r_model import ReciprocalRModel


class ThermoPoroelasticWaveSimulator:
    """Executa a simulação espectral do modelo R."""

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

    def run(
        self,
        grid: FrequencyGrid
    ) -> SimulationResult:
        """Calcula os três ramos em todas as frequências."""

        if not isinstance(grid, FrequencyGrid):
            raise TypeError(
                "A malha deve ser FrequencyGrid."
            )

        frequencies = tuple(
            float(value) for value in grid.omega
        )

        results = []
        previous_roots = None

        for omega in frequencies:

            # Calcula a equação característica.
            coefficients = (
                self.model.polynomial_coefficients(omega)
            )

            # Resolve o polinômio cúbico.
            roots = self.solver.solve(coefficients)

            # Verifica os resíduos numéricos.
            residuals = self.solver.check_residuals(
                coefficients,
                roots
            )

            if (
                not np.all(np.isfinite(residuals))
                or np.max(residuals) > self.residual_tolerance
            ):
                raise ArithmeticError(
                    f"Resíduos excessivos em omega={omega:.6g}."
                )

            # Organiza e acompanha os ramos.
            if previous_roots is None:
                ordered_roots = (
                    self.tracker.initialize_modes(roots)
                )
            else:
                try:
                    ordered_roots = self.tracker.track(
                        previous_roots,
                        roots
                    )
                except RuntimeError as exc:
                    raise RuntimeError(
                        f"Falha no rastreamento em "
                        f"omega={omega:.6g}."
                    ) from exc

            # Calcula os estados dos três ramos.
            try:
                states = tuple(
                    ModeState.from_root(
                        branch_index=index,
                        omega=omega,
                        z=complex(root)
                    )
                    for index, root in enumerate(ordered_roots)
                )
            except ValueError as exc:
                raise ValueError(
                    f"Raiz incompatível em omega={omega:.6g}."
                ) from exc

            results.append(states)
            previous_roots = ordered_roots.copy()

        return SimulationResult(
            omega=frequencies,
            states_by_frequency=tuple(results)
        )
