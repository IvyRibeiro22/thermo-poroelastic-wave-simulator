
from itertools import permutations
from math import isfinite

import numpy as np

from models.simulation_result import (
    ModeState,
    SimulationResult,
)

from numerics.coupling_continuation import CouplingContinuation
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

            # Constrói a equação característica.
            coefficients = (
                self.model.polynomial_coefficients(omega)
            )

            # Calcula as três raízes complexas.
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

    @staticmethod
    def _match_mode_indices(
        reference: dict[str, complex],
        states: tuple[ModeState, ...],
        tolerance: float = 1e-7
    ) -> dict[str, int]:
        """Relaciona os modos físicos aos índices dos ramos."""

        mode_names = ("P1", "P2", "T")

        expected = np.array(
            [reference[name] for name in mode_names],
            dtype=complex
        )

        calculated = np.array(
            [state.z for state in states],
            dtype=complex
        )

        candidates = []

        # Avalia as seis associações possíveis.
        for order in permutations(range(3)):

            errors = [
                abs(calculated[order[i]] - expected[i])
                / max(
                    abs(expected[i]),
                    abs(calculated[order[i]]),
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

        if best_error > tolerance:
            raise RuntimeError(
                "As raízes da continuação não correspondem "
                "às raízes da simulação."
            )

        if second_error - best_error <= tolerance:
            raise RuntimeError(
                "A identificação física dos modos é ambígua."
            )

        return {
            name: best_order[index]
            for index, name in enumerate(mode_names)
        }

    def run_identified(
        self,
        grid: FrequencyGrid,
        num_steps: int = 51
    ) -> dict[str, tuple[float, ...]]:
        """
        Executa a simulação e organiza as velocidades
        de fase pelos modos P1, P2 e T.
        """

        if type(num_steps) is not int or num_steps < 2:
            raise ValueError(
                "A continuação exige ao menos dois pontos."
            )

        # Executa a simulação convencional.
        result = self.run(grid)

        # Prepara a identificação por continuação térmica.
        continuation = CouplingContinuation(
            model=self.model,
            residual_tolerance=self.residual_tolerance
        )

        # Verifica a identificação em três posições da malha.
        check_indices = sorted({
            0,
            len(result.omega) // 2,
            len(result.omega) - 1,
        })

        initial_mapping = None

        for frequency_index in check_indices:

            omega = result.omega[frequency_index]

            reference = continuation.identify(
                omega=omega,
                num_steps=num_steps
            )

            mapping = self._match_mode_indices(
                reference=reference,
                states=result.states_by_frequency[
                    frequency_index
                ]
            )

            if initial_mapping is None:
                initial_mapping = mapping

            elif mapping != initial_mapping:
                raise RuntimeError(
                    "A correspondência dos modos mudou "
                    "ao longo da malha de frequências. "
                    "É necessária uma análise adicional."
                )

        # Extrai as velocidades dos ramos identificados.
        return {
            "omega": result.omega,
            "P1": result.values_for_branch(
                initial_mapping["P1"],
                "phase_velocity"
            ),
            "P2": result.values_for_branch(
                initial_mapping["P2"],
                "phase_velocity"
            ),
            "T": result.values_for_branch(
                initial_mapping["T"],
                "phase_velocity"
            ),
        }
