
import numpy as np
import pytest

from dataclasses import FrozenInstanceError

from numerics.frequency_grid import FrequencyGrid


# ============================================================
# TESTES DA MALHA DE FREQUÊNCIAS
# ============================================================


def test_default_frequency_grid():
    """Verifica a configuração padrão da simulação."""

    grid = FrequencyGrid.build()

    assert grid.omega.size == 1601
    assert grid.omega[0] == pytest.approx(1.0)
    assert grid.omega[-1] == pytest.approx(1e8)

    assert np.all(np.isfinite(grid.omega))
    assert np.all(np.diff(grid.omega) > 0.0)


def test_logarithmic_spacing():
    """Verifica o espaçamento uniforme na escala logarítmica."""

    grid = FrequencyGrid.build(
        omega_min=1.0,
        omega_max=1e8,
        num_points=1601
    )

    # Na escala logarítmica, os incrementos são constantes.
    log_frequencies = np.log10(grid.omega)
    intervals = np.diff(log_frequencies)

    np.testing.assert_allclose(
        intervals,
        intervals[0],
        rtol=1e-10,
        atol=1e-12
    )


def test_linear_frequency_grid():
    """Verifica a geração de uma malha linear."""

    grid = FrequencyGrid.build(
        omega_min=1.0,
        omega_max=5.0,
        num_points=5,
        use_logspace=False
    )

    expected = [1.0, 2.0, 3.0, 4.0, 5.0]

    np.testing.assert_allclose(grid.omega, expected)


def test_characteristic_frequency_refinement():
    """Verifica a inclusão de pontos próximos à transição física."""

    grid = FrequencyGrid.build(
        omega_min=1.0,
        omega_max=1000.0,
        num_points=4,
        characteristic_omegas=(50.0,)
    )

    # Pontos adicionais em 0,9; 1,0 e 1,1 vezes omega_c.
    assert grid.omega.size == 7

    for frequency in (45.0, 50.0, 55.0):
        assert np.any(
            np.isclose(grid.omega, frequency)
        )


def test_duplicate_frequencies():
    """Verifica que a malha não contém frequências repetidas."""

    grid = FrequencyGrid.build(
        omega_min=1.0,
        omega_max=1000.0,
        num_points=4,
        characteristic_omegas=(10.0,)
    )

    assert len(grid.omega) == len(np.unique(grid.omega))


def test_grid_immutability():
    """Verifica a proteção dos valores da malha."""

    grid = FrequencyGrid.build()

    # O array não permite alteração direta.
    with pytest.raises(ValueError):
        grid.omega[0] = 500.0

    # frozen=True impede substituir o atributo.
    with pytest.raises(FrozenInstanceError):
        grid.omega = np.array([1.0, 2.0])


# ============================================================
# TESTES DE ENTRADAS INVÁLIDAS
# ============================================================


@pytest.mark.parametrize(
    "omega_min, omega_max",
    [
        (0.0, 100.0),
        (100.0, 100.0),
        (100.0, 1.0),
        (1.0, float("inf")),
    ]
)
def test_invalid_frequency_limits(omega_min, omega_max):
    """Rejeita limites de frequência inválidos."""

    with pytest.raises(ValueError):
        FrequencyGrid.build(
            omega_min=omega_min,
            omega_max=omega_max
        )


@pytest.mark.parametrize(
    "num_points",
    [1, 2.5, True]
)
def test_invalid_number_of_points(num_points):
    """Rejeita quantidades de pontos inválidas."""

    with pytest.raises(ValueError):
        FrequencyGrid.build(num_points=num_points)


@pytest.mark.parametrize(
    "characteristic",
    [0.0, -10.0, float("inf")]
)
def test_invalid_characteristic_frequencies(characteristic):
    """Rejeita frequências características inválidas."""

    with pytest.raises(ValueError):
        FrequencyGrid.build(
            characteristic_omegas=(characteristic,)
        )


@pytest.mark.parametrize(
    "frequencies",
    [
        [0.0, 1.0],
        [3.0, 2.0],
        [1.0, 1.0],
        [1.0, float("nan")],
    ]
)
def test_invalid_manual_grid(frequencies):
    """Rejeita malhas não positivas, repetidas ou desordenadas."""

    with pytest.raises(ValueError):
        FrequencyGrid(
            omega=np.array(frequencies)
        )


def test_invalid_spacing_option():
    """Verifica que a opção logarítmica deve ser booleana."""

    with pytest.raises(TypeError):
        FrequencyGrid.build(use_logspace="sim")
