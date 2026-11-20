
from dataclasses import dataclass
from math import isfinite, sqrt

import numpy as np

from models.simulation_result import ModeState
from physics.reciprocal_r_model import ReciprocalRModel


@dataclass(frozen=True)
class ModePolarization:
    """Armazena a composição escalonada de um modo."""

    omega: float
    z: complex

    amplitudes: tuple[complex, complex, complex]

    solid_fraction: float
    fluid_fraction: float
    thermal_fraction: float

    temperature_scale: float
    residual: float


class ModePolarizationAnalyzer:
    """Analisa os autovetores da matriz de dispersão."""

    def __init__(
        self,
        model: ReciprocalRModel,
        residual_tolerance: float = 1e-8,
        ambiguity_tolerance: float = 1e-8,
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

        if (
            not isfinite(ambiguity_tolerance)
            or not 0.0 < ambiguity_tolerance < 1.0
        ):
            raise ValueError(
                "A tolerância de ambiguidade deve estar entre 0 e 1."
            )

        self.model = model
        self.residual_tolerance = residual_tolerance
        self.ambiguity_tolerance = ambiguity_tolerance

    def analyze(
        self,
        state: ModeState
    ) -> ModePolarization:
        """Calcula a composição escalonada de um modo."""

        if not isinstance(state, ModeState):
            raise TypeError(
                "O estado deve ser ModeState."
            )

        medium = self.model.medium
        thermal = medium.thermal

        # Escala térmica de referência.
        temperature_scale = sqrt(
            medium.A() * thermal.theta0 / thermal.C
        )

        if (
            not isfinite(temperature_scale)
            or temperature_scale <= 0.0
        ):
            raise ArithmeticError(
                "A escala térmica calculada é inválida."
            )

        # Matriz espectral correspondente ao estado.
        matrix = self.model.dispersion_matrix(
            omega=state.omega,
            k=state.k,
        )

        if (
            matrix.shape != (3, 3)
            or not np.all(np.isfinite(matrix))
        ):
            raise ArithmeticError(
                "A matriz de dispersão é inválida."
            )

        # Escalonamento das três componentes.
        column_scales = np.array(
            [1.0, 1.0, 1j * temperature_scale],
            dtype=complex,
        )

        scaled_matrix = matrix * column_scales

        # Normalização das linhas.
        row_norms = np.linalg.norm(
            scaled_matrix,
            axis=1,
        )

        # Evita amplificar erros de arredondamento
        # quando a equação térmica está quase anulada.
        t_omega = self.model.t_omega(state.omega)

        thermal_reference_norm = (
            temperature_scale
            * (
                abs(thermal.kappa)
                + abs(
                    thermal.C
                    * t_omega
                    * state.z
                )
            )
        )

        row_norms[2] = max(
            row_norms[2],
            thermal_reference_norm,
        )

        if (
            not np.all(np.isfinite(row_norms))
            or np.any(row_norms <= 0.0)
        ):
            raise ArithmeticError(
                "Não foi possível normalizar a matriz."
            )

        scaled_matrix = (
            scaled_matrix / row_norms[:, None]
        )

        # Decomposição em valores singulares.
        _, singular_values, vh = np.linalg.svd(
            scaled_matrix
        )

        # Detecta direções aproximadamente nulas
        # que não podem ser distinguidas.
        if (
            singular_values[-2]
            <= self.ambiguity_tolerance
            * singular_values[0]
        ):
            raise RuntimeError(
                "Autovetor ambíguo: existem direções "
                "aproximadamente nulas não distinguíveis."
            )

        vector = vh[-1, :].conj()

        vector_norm = np.linalg.norm(vector)

        if (
            not np.isfinite(vector_norm)
            or vector_norm <= 0.0
        ):
            raise ArithmeticError(
                "Não foi possível normalizar o autovetor."
            )

        vector = vector / vector_norm

        # Convenção de fase:
        # maior componente real e positiva.
        largest_index = int(
            np.argmax(np.abs(vector))
        )

        largest_component = vector[largest_index]

        vector *= (
            largest_component.conjugate()
            / abs(largest_component)
        )

        # Resíduo do sistema matricial escalonado.
        residual = (
            np.linalg.norm(scaled_matrix @ vector)
            / np.linalg.norm(
                scaled_matrix,
                ord="fro"
            )
        )

        if (
            not np.isfinite(residual)
            or residual > self.residual_tolerance
        ):
            raise ArithmeticError(
                "O autovetor não satisfaz a matriz "
                "de dispersão dentro da tolerância."
            )

        # Indicadores relativos das amplitudes.
        fractions = np.abs(vector) ** 2

        amplitudes = tuple(
            complex(value)
            for value in vector
        )

        return ModePolarization(
            omega=state.omega,
            z=state.z,
            amplitudes=amplitudes,
            solid_fraction=float(fractions[0]),
            fluid_fraction=float(fractions[1]),
            thermal_fraction=float(fractions[2]),
            temperature_scale=temperature_scale,
            residual=float(residual),
        )
