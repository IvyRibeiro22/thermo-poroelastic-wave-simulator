
from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class MechanicalProperties:
    """Propriedades mecânicas e hidráulicas do meio poroso."""

    phi: float
    rho_s: float
    rho_f: float
    a_inf: float
    k0: float
    eta: float
    lambda_: float
    mu: float
    m: float
    alpha: float
    viscous_length: float

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        values = vars(self)

        for name, value in values.items():
            if not isfinite(value):
                raise ValueError(
                    f"{name} deve ser um número finito."
                )

        if not 0 < self.phi < 1:
            raise ValueError("A porosidade deve estar entre 0 e 1.")

        for name in (
            "rho_s", "rho_f", "k0", "eta",
            "mu", "m", "viscous_length"
        ):
            if getattr(self, name) <= 0:
                raise ValueError(
                    f"{name} deve ser positivo."
                )

        if self.a_inf < 1:
            raise ValueError("A tortuosidade deve ser >= 1.")

        if not 0 <= self.alpha <= 1:
            raise ValueError("O coeficiente de Biot deve estar entre 0 e 1.")

        if self.lambda_ + 2 * self.mu <= 0:
            raise ValueError(
                "O módulo compressional deve ser positivo."
            )
