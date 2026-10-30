# ============================================================
# CLASSE: ThermalProperties
# ARQUIVO: models/thermal_properties.py
#
# OBJETIVO:
# Armazenar e validar as propriedades térmicas utilizadas
# no modelo termo-poroelástico recíproco unidimensional.
# Não haverá a resolução de equações, é só para armazenar os parâmetros.
# ===========================================================



from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class ThermalProperties:
    """Propriedades térmicas do meio poroso saturado."""

    # CAPACIDADE TÉRMICA VOLUMÉTRICA EFETIVA
    C: float #capacidade de armazenamento de energia térmica por unidade de volume do meio.

    # COEFICIENTE DE ACOPLAMENTO TÉRMICO DO SÓLIDO
    beta_s: float #coeficiente de acoplamento térmico do sólido,
                  #que descreve a transferência de calor entre o sólido e o fluido.
    
    beta_f: float #coeficiente de acoplamento térmico do fluido,
                  #que descreve a transferência de calor entre o fluido e o sólido.
    
    kappa: float #condutividade térmica do meio poroso saturado,
                 #que descreve a capacidade do meio de conduzir calor.
    
    tau0: float #tempo de relaxação térmica do meio poroso saturado,
                #que descreve a taxa de dissipação de calor no meio.
    
    theta0: float #temperatura de referência do meio poroso saturado,
                  #que é usada como referência para cálculos de transferência de calor.

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Verifica a validade básica dos parâmetros térmicos."""

        for name, value in vars(self).items():
            if not isfinite(value):
                raise ValueError(
                    f"{name} deve ser um número finito."
                )

        if self.C <= 0.0:
            raise ValueError(
                "A capacidade térmica volumétrica deve ser positiva."
            )

        if self.kappa <= 0.0:
            raise ValueError(
                "A condutividade térmica deve ser positiva."
            )

        if self.tau0 < 0.0:
            raise ValueError(
                "O tempo de relaxação térmica não pode ser negativo."
            )

        if self.theta0 <= 0.0:
            raise ValueError(
                "A temperatura de referência deve ser positiva."
            )

