
import pytest

from dataclasses import FrozenInstanceError

from models.simulation_result import ModeState


def test_wave_without_attenuation():
    """Verifica uma onda com número de onda real."""

    state = ModeState.from_root(
        branch_index=0,
        omega=1000.0,
        z=4e6 + 0j
    )

    assert state.k == pytest.approx(0.5 + 0j)
    assert state.phase_velocity == pytest.approx(2000.0)
    assert state.attenuation == pytest.approx(0.0)
    assert state.q_inverse == pytest.approx(0.0)


def test_wave_with_attenuation():
    """Verifica velocidade, atenuação e Q inverso."""

    omega = 1000.0

    # Número de onda de referência: k = 0.5 + 0.02i.
    expected_k = 0.5 + 0.02j

    # Constrói z a partir de um número de onda conhecido.
    z = (omega / expected_k) ** 2

    state = ModeState.from_root(
        branch_index=1,
        omega=omega,
        z=z
    )

    assert state.k == pytest.approx(expected_k)

    # V = 1000 / 0.5 = 2000 m/s.
    assert state.phase_velocity == pytest.approx(2000.0)

    # Atenuação espacial = Im(k) = 0.02 1/m.
    assert state.attenuation == pytest.approx(0.02)

    # Q^-1 = 2 * 0.02 / 0.5 = 0.08.
    assert state.q_inverse == pytest.approx(0.08)


@pytest.mark.parametrize("branch_index", [0, 1, 2])
def test_valid_branch_indices(branch_index):
    """Aceita os três identificadores provisórios."""

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
    """Rejeita identificadores de ramo inválidos."""

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
    """Rejeita frequências não positivas ou não finitas."""

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
    """Rejeita raízes incompatíveis com as condições adotadas."""

    with pytest.raises(ValueError):
        ModeState.from_root(
            branch_index=0,
            omega=1000.0,
            z=z
        )


def test_nonnumeric_root():
    """Rejeita uma raiz que não pode ser convertida em complexo."""

    with pytest.raises(TypeError):
        ModeState.from_root(
            branch_index=0,
            omega=1000.0,
            z="valor_invalido"
        )


def test_mode_state_immutability():
    """Verifica a proteção dos atributos por frozen=True."""

    state = ModeState.from_root(
        branch_index=0,
        omega=1000.0,
        z=4e6 + 0j
    )

    with pytest.raises(FrozenInstanceError):
        state.phase_velocity = 3000.0
