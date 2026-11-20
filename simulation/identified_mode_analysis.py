
from dataclasses import dataclass

import numpy as np

from models.simulation_result import ModeState

from numerics.coupling_continuation import CouplingContinuation
from numerics.frequency_grid import FrequencyGrid
from numerics.mode_polarization import (
    ModePolarization,
    ModePolarizationAnalyzer,
)

from simulation.thermo_poroelastic_wave_simulator import (
    ThermoPoroelasticWaveSimulator,
)


@dataclass(frozen=True)
class IdentifiedModeSample:
    """Reúne o estado e a polarização de um modo."""

    state: ModeState
    polarization: ModePolarization
    overlap_previous: float | None


class IdentifiedModeAnalyzer:
    """Analisa os modos identificados ao longo das frequências."""

    MODE_NAMES = ("P1", "P2", "T")

    def __init__(
        self,
        simulator: ThermoPoroelasticWaveSimulator,
    ) -> None:

        if not isinstance(
            simulator,
            ThermoPoroelasticWaveSimulator
        ):
            raise TypeError(
                "O simulador deve ser ThermoPoroelasticWaveSimulator."
            )

        self.simulator = simulator

    @staticmethod
    def _overlap(
        previous: ModePolarization,
        current: ModePolarization,
    ) -> float:
        """Calcula a similaridade entre dois autovetores."""

        a = np.asarray(
            previous.amplitudes,
            dtype=complex,
        )

        b = np.asarray(
            current.amplitudes,
            dtype=complex,
        )

        # Produto interno complexo, independente da fase global.
        numerator = abs(np.vdot(a, b))

        denominator = (
            np.linalg.norm(a)
            * np.linalg.norm(b)
        )

        if (
            not np.isfinite(denominator)
            or denominator <= 0.0
        ):
            raise ArithmeticError(
                "Não foi possível comparar os autovetores."
            )

        overlap = float(numerator / denominator)

        if not np.isfinite(overlap):
            raise ArithmeticError(
                "A similaridade calculada não é finita."
            )

        return min(1.0, max(0.0, overlap))

    def run(
        self,
        grid: FrequencyGrid,
        num_steps: int = 51,
    ) -> dict:
        """Integra identificação, velocidades e autovetores."""

        if type(num_steps) is not int or num_steps < 2:
            raise ValueError(
                "A continuação exige ao menos dois pontos."
            )

        # Executa a simulação convencional.
        result = self.simulator.run(grid)

        # Identificação por continuação termomecânica.
        continuation = CouplingContinuation(
            model=self.simulator.model,
            residual_tolerance=self.simulator.residual_tolerance,
        )

        # Análise das amplitudes dos modos.
        polarization_analyzer = ModePolarizationAnalyzer(
            model=self.simulator.model,
            residual_tolerance=self.simulator.residual_tolerance,
        )

        mode_results = {
            name: []
            for name in self.MODE_NAMES
        }

        previous_mapping = None
        previous_polarizations = {}

        for frequency_index, omega in enumerate(result.omega):

            states = result.states_by_frequency[
                frequency_index
            ]

            # Identifica os modos nesta frequência.
            reference = continuation.identify(
                omega=omega,
                num_steps=num_steps,
            )

            mapping = self.simulator._match_mode_indices(
                reference=reference,
                states=states,
            )

            # Interrompe se a correspondência dos
            # ramos mudar entre frequências.
            if previous_mapping is not None:
                if mapping != previous_mapping:
                    raise RuntimeError(
                        "A identificação dos modos mudou "
                        f"em omega={omega:.6g}. "
                        "É necessária uma análise adicional."
                    )

            for name in self.MODE_NAMES:

                state = states[mapping[name]]

                polarization = polarization_analyzer.analyze(
                    state
                )

                # A primeira frequência não possui
                # autovetor anterior para comparação.
                overlap = None

                if name in previous_polarizations:
                    overlap = self._overlap(
                        previous_polarizations[name],
                        polarization,
                    )

                mode_results[name].append(
                    IdentifiedModeSample(
                        state=state,
                        polarization=polarization,
                        overlap_previous=overlap,
                    )
                )

                previous_polarizations[name] = polarization

            previous_mapping = mapping

        return {
            "omega": result.omega,
            "P1": tuple(mode_results["P1"]),
            "P2": tuple(mode_results["P2"]),
            "T": tuple(mode_results["T"]),
        }
