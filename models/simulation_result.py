
from dataclasses import dataclass
from cmath import sqrt
from math import isfinite, isclose


@dataclass(frozen=True)
class ModeState:
    """
    Representa o estado de um ramo em uma frequência.

    Armazena a raiz complexa, o número de onda
    e a velocidade de fase.
    """

    branch_index: int
    omega: float
    z: complex
    k: complex
    phase_velocity: float

    @classmethod
    def from_root(
        cls,
        branch_index: int,
        omega: float,
        z: complex
    ) -> "ModeState":
        """Calcula as propriedades de propagação a partir de z."""

        if (
            type(branch_index) is not int
            or branch_index not in (0, 1, 2)
        ):
            raise ValueError(
                "O ramo deve ser identificado por 0, 1 ou 2."
            )

        if not isfinite(omega) or omega <= 0.0:
            raise ValueError(
                "A frequência angular deve ser finita e positiva."
            )

        try:
            z = complex(z)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "A raiz z deve ser um número complexo."
            ) from exc

        if (
            not isfinite(z.real)
            or not isfinite(z.imag)
            or z == 0
        ):
            raise ValueError(
                "A raiz z deve ser finita e diferente de zero."
            )

        # Calcula o número de onda complexo.
        k = omega / sqrt(z)

        # Seleciona a propagação no sentido positivo de x.
        if k.real < 0.0:
            k = -k

        if not isfinite(k.real) or not isfinite(k.imag):
            raise ValueError(
                "O número de onda calculado não é finito."
            )

        if k.real <= 0.0:
            raise ValueError(
                "O número de onda não representa "
                "propagação no sentido positivo de x."
            )

        if k.imag < 0.0:
            raise ValueError(
                "A raiz é incompatível com a convenção "
                "de propagação adotada."
            )

        # Calcula a velocidade de fase.
        phase_velocity = omega / k.real

        if not isfinite(phase_velocity):
            raise ValueError(
                "A velocidade de fase não é finita."
            )

        return cls(
            branch_index=branch_index,
            omega=omega,
            z=z,
            k=k,
            phase_velocity=phase_velocity,
        )


@dataclass(frozen=True)
class SimulationResult:
    """
    Armazena os resultados dos três ramos
    ao longo das frequências da simulação.
    """

    omega: tuple[float, ...]
    states_by_frequency: tuple[
        tuple[ModeState, ModeState, ModeState], ...
    ]

    def __post_init__(self) -> None:
        """Valida e organiza os resultados da simulação."""

        try:
            frequencies = tuple(
                float(value) for value in self.omega
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(
                "As frequências devem ser numéricas."
            ) from exc

        if len(frequencies) < 2:
            raise ValueError(
                "A simulação deve conter ao menos duas frequências."
            )

        if not all(
            isfinite(value) and value > 0.0
            for value in frequencies
        ):
            raise ValueError(
                "As frequências devem ser finitas e positivas."
            )

        if any(
            current <= previous
            for previous, current in zip(
                frequencies[:-1],
                frequencies[1:]
            )
        ):
            raise ValueError(
                "As frequências devem estar em ordem crescente."
            )

        try:
            rows = tuple(
                tuple(states)
                for states in self.states_by_frequency
            )
        except TypeError as exc:
            raise TypeError(
                "Os estados devem ser organizados por frequência."
            ) from exc

        if len(rows) != len(frequencies):
            raise ValueError(
                "Cada frequência deve possuir um conjunto de estados."
            )

        for frequency, states in zip(frequencies, rows):

            if len(states) != 3:
                raise ValueError(
                    "Cada frequência deve possuir três modos."
                )

            for index, state in enumerate(states):

                if not isinstance(state, ModeState):
                    raise TypeError(
                        "Todos os estados devem ser ModeState."
                    )

                if state.branch_index != index:
                    raise ValueError(
                        "Os ramos devem estar na ordem 0, 1 e 2."
                    )

                if not isclose(
                    state.omega,
                    frequency,
                    rel_tol=1e-12,
                    abs_tol=0.0
                ):
                    raise ValueError(
                        "A frequência do estado não corresponde "
                        "à frequência da simulação."
                    )

        object.__setattr__(
            self, "omega", frequencies
        )

        object.__setattr__(
            self, "states_by_frequency", rows
        )

    def states_for_branch(
        self,
        branch_index: int
    ) -> tuple[ModeState, ...]:
        """Retorna os estados de um ramo em todas as frequências."""

        if (
            type(branch_index) is not int
            or branch_index not in (0, 1, 2)
        ):
            raise ValueError(
                "O ramo deve ser identificado por 0, 1 ou 2."
            )

        return tuple(
            states[branch_index]
            for states in self.states_by_frequency
        )

    def values_for_branch(
        self,
        branch_index: int,
        quantity: str
    ) -> tuple[float | complex, ...]:
        """Extrai uma propriedade de propagação de um ramo."""

        valid_quantities = (
            "z",
            "k",
            "phase_velocity",
        )

        if quantity not in valid_quantities:
            raise ValueError(
                "Propriedade de propagação inválida."
            )

        states = self.states_for_branch(branch_index)

        return tuple(
            getattr(state, quantity)
            for state in states
        )
