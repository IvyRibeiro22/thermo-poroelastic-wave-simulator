
from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.polynomial import Polynomial

from models.porous_medium import PorousMedium
from physics.resistance_model import ResistanceModel


@dataclass(frozen=True)
class ReciprocalRModel:
    """
    Modelo termo-poroelástico recíproco R em 1D.

    Constrói a equação característica cúbica em z = v²,
    considerando o acoplamento mecânico, hidráulico e térmico.
    """

    medium: PorousMedium
    resistance: ResistanceModel

    def __post_init__(self) -> None:
        """Verifica os objetos utilizados pelo modelo."""

        if not isinstance(self.medium, PorousMedium):
            raise TypeError(
                "medium deve ser um objeto PorousMedium."
            )

        if not isinstance(self.resistance, ResistanceModel):
            raise TypeError(
                "resistance deve herdar de ResistanceModel."
            )

    def t_omega(self, omega: float) -> complex:
        """
        Calcula o parâmetro térmico complexo.

        t(omega) = tau0 + i/omega
        Unidade: segundos.
        """

        if not isfinite(omega) or omega <= 0.0:
            raise ValueError(
                "A frequência angular deve ser finita e positiva."
            )

        return self.medium.thermal.tau0 + 1j / omega

    def polynomial_coefficients(
        self,
        omega: float
    ) -> np.ndarray:
        """
        Calcula os coeficientes do polinômio P_R(z; omega).

        Retorna [c3, c2, c1, c0], correspondente a:

            c3*z³ + c2*z² + c1*z + c0 = 0
        """

        p = self.medium.mechanical
        t = self.medium.thermal

        rho = self.medium.rho()
        A = self.medium.A()
        B = self.medium.B()
        beta = self.medium.beta()

        # Coeficientes complexos dependentes da frequência.
        t_omega = self.t_omega(omega)
        R = self.resistance.R_omega(omega, self.medium)

        # Polinômios lineares em z.
        a = Polynomial([A, -rho])
        d = Polynomial([p.m, -R])
        n = Polynomial([B, -p.rho_f])

        # Termo térmico: kappa - C*t_omega*z.
        thermal = Polynomial([
            t.kappa,
            -t.C * t_omega
        ])

        # Acoplamento termo-poroelástico recíproco.
        coupling = (
            beta ** 2 * d
            + t.beta_f ** 2 * a
            - 2.0 * beta * t.beta_f * n
        )

        # Equação característica P_R(z; omega).
        z = Polynomial([0.0, 1.0])

        characteristic = (
            (a * d - n * n) * thermal
            - t.theta0 * t_omega * z * coupling
        )

        # NumPy Polynomial armazena coeficientes em ordem
        # crescente. Inverteremos para o futuro np.roots().
        coefficients = np.asarray(
            characteristic.coef[::-1],
            dtype=complex
        )

        if len(coefficients) != 4:
            raise ValueError(
                "O polinômio característico deve ser cúbico."
            )

        if not np.all(np.isfinite(coefficients)):
            raise ValueError(
                "Os coeficientes do polinômio não são finitos."
            )

        return coefficients

    def dispersion_matrix(
        self,
        omega: float,
        k: complex
    ) -> np.ndarray:
        """
        Constrói a matriz espectral 3x3 do modelo R.

        A matriz utiliza z = (omega/k)² e uma forma
        escalonada compatível com o polinômio característico.
        """

        t_omega = self.t_omega(omega)

        if k == 0 or not (
            isfinite(k.real) and isfinite(k.imag)
        ):
            raise ValueError(
                "O número de onda deve ser finito e não nulo."
            )

        p = self.medium.mechanical
        t = self.medium.thermal

        z = (omega / k) ** 2
        R = self.resistance.R_omega(omega, self.medium)

        a = self.medium.A() - self.medium.rho() * z
        d = p.m - R * z
        n = self.medium.B() - p.rho_f * z

        beta = self.medium.beta()
        beta_f = t.beta_f

        thermal = t.kappa - t.C * t_omega * z
        thermal_factor = t.theta0 * t_omega * z

        return np.array([
            [a, n, beta],
            [n, d, beta_f],
            [
                thermal_factor * beta,
                thermal_factor * beta_f,
                thermal
            ]
        ], dtype=complex)
