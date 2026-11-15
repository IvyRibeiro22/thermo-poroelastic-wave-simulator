
from dataclasses import FrozenInstanceError

import pytest

from models.simulation_result import ModeState, SimulationResult


# Testes da classe ModeState.


def test_real_wavenumber():
    """Verifica uma onda com número de onda real."""

    state = ModeState.from_root(
        branch_index=0,
        omega=1000.0,
        z=4e6 + 0j
    )

    assert state.k == pytest.approx(0.5 + 0j)
    assert state.phase_velocity == pytest.approx(2000.0)


def test_complex_root_phase_velocity():
    """Verifica a velocidade de fase para uma raiz complexa."""

    omega = 1000.0

    expected_k = 0.5 + 0.02j

    z = (omega / expected_k) ** 2

    state = ModeState.from_root(
        branch_index=1,
        omega=omega,
        z=z
    )

    assert state.k == pytest.approx(expected_k)
    assert state.phase_velocity == pytest.approx(2000.0)


@pytest.mark.parametrize("branch_index", [0, 1, 2])
def test_valid_branch_indices(branch_index):
    """Aceita os três identificadores de ramo."""

    state = ModeState.from_root(
        branch_index=branch_index,
        omega=1000.0,
        z=4e6 + 0j
    )

    assert state.branch_index == branch_index


@pytest.mark.parametrize(
    "branch_index",
    [-1, 3, True, 1.0]
)
def test_invalid_branch_indices(branch_index):
    """Rejeita identificadores inválidos."""

    with pytest.raises(ValueError):
        ModeState.from_root(
            branch_index=branch_index,
            omega=1000.0,
            z=4e6 + 0j
        )


@pytest.mark.parametrize(
    "omega",
    [0.0, -100.0, float("inf"), float("nan")]
)
def test_invalid_frequency(omega):
    """Rejeita frequências inválidas."""

    with pytest.raises(ValueError):
        ModeState.from_root(
            branch_index=0,
            omega=omega,
            z=4e6 + 0j
        )


@pytest.mark.parametrize(
    "z",
    [
        0j,
        complex(float("inf"), 0.0),
        complex(float("nan"), 0.0),
        4e6 + 4e5j,
        -4e6 + 0j,
    ]
)
def test_invalid_roots(z):
    """Rejeita raízes incompatíveis com o modelo adotado."""

    with pytest.raises(ValueError):
        ModeState.from_root(
            branch_index=0,
            omega=1000.0,
            z=z
        )


def test_nonnumeric_root():
    """Rejeita raízes não numéricas."""

    with pytest.raises(TypeError):
        ModeState.from_root(
            branch_index=0,
            omega=1000.0,
            z="valor_invalido"
        )


def test_mode_state_immutability():
    """Verifica a imutabilidade de ModeState."""

    state = ModeState.from_root(
        branch_index=0,
        omega=1000.0,
        z=4e6 + 0j
    )

    with pytest.raises(FrozenInstanceError):
        state.phase_velocity = 3000.0


# Dados de referência para SimulationResult.


def make_states(omega):
    """Cria três estados de referência."""

    roots = (4e6, 1e6, 2.5e5)

    return tuple(
        ModeState.from_root(
            branch_index=index,
            omega=omega,
            z=root
        )
        for index, root in enumerate(roots)
    )


def make_result():
    """Cria um resultado com duas frequências."""

    frequencies = (100.0, 200.0)

    return SimulationResult(
        omega=frequencies,
        states_by_frequency=tuple(
            make_states(omega)
            for omega in frequencies
        )
    )


# Testes da classe SimulationResult.


def test_valid_simulation_result():
    """Verifica a organização dos resultados."""

    result = make_result()

    assert result.omega == (100.0, 200.0)
    assert len(result.states_by_frequency) == 2

    for states in result.states_by_frequency:
        assert len(states) == 3
        assert tuple(
            state.branch_index for state in states
        ) == (0, 1, 2)


def test_states_for_branch():
    """Recupera os estados de um ramo."""

    result = make_result()

    states = result.states_for_branch(1)

    assert len(states) == 2
    assert all(state.branch_index == 1 for state in states)

    assert tuple(state.omega for state in states) == (
        100.0, 200.0
    )


@pytest.mark.parametrize(
    "branch_index, expected_velocity",
    [
        (0, 2000.0),
        (1, 1000.0),
        (2, 500.0),
    ]
)
def test_phase_velocities(
    branch_index,
    expected_velocity
):
    """Extrai as velocidades de fase de um ramo."""

    result = make_result()

    velocities = result.values_for_branch(
        branch_index,
        "phase_velocity"
    )

    assert velocities == pytest.approx(
        (expected_velocity, expected_velocity)
    )


def test_complex_wavenumbers():
    """Extrai os números de onda de um ramo."""

    result = make_result()

    values = result.values_for_branch(
        branch_index=0,
        quantity="k"
    )

    assert values == pytest.approx(
        (0.05 + 0j, 0.10 + 0j)
    )


@pytest.mark.parametrize(
    "quantity",
    ["attenuation", "q_inverse"]
)
def test_unavailable_quantities(quantity):
    """Rejeita propriedades fora do escopo atual."""

    result = make_result()

    with pytest.raises(ValueError):
        result.values_for_branch(
            branch_index=0,
            quantity=quantity
        )


def test_simulation_result_immutability():
    """Verifica a imutabilidade da estrutura."""

    result = make_result()

    with pytest.raises(FrozenInstanceError):
        result.omega = (300.0, 400.0)

    with pytest.raises(TypeError):
        result.states_by_frequency[0] = make_states(300.0)


def test_lists_converted_to_tuples():
    """Converte sequências de entrada em tuplas."""

    result = SimulationResult(
        omega=[100.0, 200.0],
        states_by_frequency=[
            list(make_states(100.0)),
            list(make_states(200.0))
        ]
    )

    assert isinstance(result.omega, tuple)
    assert isinstance(result.states_by_frequency, tuple)
    assert isinstance(result.states_by_frequency[0], tuple)


@pytest.mark.parametrize(
    "frequencies",
    [
        (100.0,),
        (100.0, 100.0),
        (200.0, 100.0),
        (0.0, 100.0),
        (100.0, float("nan")),
        (100.0, float("inf")),
    ]
)
def test_invalid_simulation_frequencies(frequencies):
    """Rejeita malhas de frequência inválidas."""

    with pytest.raises(ValueError):
        SimulationResult(
            omega=frequencies,
            states_by_frequency=(
                make_states(100.0),
                make_states(200.0)
            )
        )


def test_incorrect_frequency_count():
    """Rejeita quantidades incompatíveis de resultados."""

    with pytest.raises(ValueError):
        SimulationResult(
            omega=(100.0, 200.0),
            states_by_frequency=(
                make_states(100.0),
            )
        )


@pytest.mark.parametrize("number_of_modes", [2, 4])
def test_incorrect_mode_count(number_of_modes):
    """Exige exatamente três estados por frequência."""

    states = make_states(100.0)

    if number_of_modes == 2:
        invalid_states = states[:2]
    else:
        invalid_states = states + (states[0],)

    with pytest.raises(ValueError):
        SimulationResult(
            omega=(100.0, 200.0),
            states_by_frequency=(
                invalid_states,
                make_states(200.0)
            )
        )


def test_invalid_state_type():
    """Rejeita objetos que não sejam ModeState."""

    with pytest.raises(TypeError):
        SimulationResult(
            omega=(100.0, 200.0),
            states_by_frequency=(
                (1, 2, 3),
                make_states(200.0)
            )
        )


def test_incorrect_branch_order():
    """Rejeita estados fora da ordem dos ramos."""

    states = make_states(100.0)

    with pytest.raises(ValueError):
        SimulationResult(
            omega=(100.0, 200.0),
            states_by_frequency=(
                (states[1], states[0], states[2]),
                make_states(200.0)
            )
        )


def test_mismatched_state_frequency():
    """Rejeita estados associados à frequência errada."""

    with pytest.raises(ValueError):
        SimulationResult(
            omega=(100.0, 200.0),
            states_by_frequency=(
                make_states(100.0),
                make_states(300.0)
            )
        )


@pytest.mark.parametrize(
    "branch_index",
    [-1, 3, True, 1.0]
)
def test_invalid_result_branch(branch_index):
    """Rejeita índices de ramo inválidos."""

    result = make_result()

    with pytest.raises(ValueError):
        result.states_for_branch(branch_index)


@pytest.mark.parametrize(
    "quantity",
    ["invalid", "omega", "branch_index"]
)
def test_invalid_result_quantity(quantity):
    """Rejeita propriedades não disponíveis."""

    result = make_result()

    with pytest.raises(ValueError):
        result.values_for_branch(0, quantity)
