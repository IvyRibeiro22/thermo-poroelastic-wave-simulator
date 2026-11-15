
import numpy as np
import pytest

from numerics.mode_tracker import ModeTracker


def test_initialize_modes():
    """Organiza as raízes iniciais pelo módulo decrescente."""

    roots = np.array([
        3.0 + 1.0j,
        20.0 + 0.0j,
        100.0 + 5.0j,
    ])

    ordered = ModeTracker.initialize_modes(roots)

    expected = np.array([
        100.0 + 5.0j,
        20.0 + 0.0j,
        3.0 + 1.0j,
    ])

    np.testing.assert_allclose(ordered, expected)


def test_track_reordered_roots():
    """Preserva os ramos quando as raízes mudam de ordem."""

    tracker = ModeTracker()

    previous = np.array([
        100.0 + 2.0j,
        20.0 + 1.0j,
        2.0 + 0.1j,
    ])

    # As novas raízes aparecem em outra ordem.
    current = np.array([
        2.1 + 0.1j,
        101.0 + 2.1j,
        21.0 + 1.2j,
    ])

    tracked = tracker.track(previous, current)

    expected = np.array([
        101.0 + 2.1j,
        21.0 + 1.2j,
        2.1 + 0.1j,
    ])

    np.testing.assert_allclose(tracked, expected)


def test_track_complex_roots():
    """Verifica a continuidade de raízes complexas."""

    tracker = ModeTracker()

    previous = np.array([
        100.0 + 20.0j,
        20.0 - 8.0j,
        1.0 + 0.5j,
    ])

    current = np.array([
        1.1 + 0.55j,
        103.0 + 19.0j,
        21.0 - 8.5j,
    ])

    tracked = tracker.track(previous, current)

    expected = np.array([
        103.0 + 19.0j,
        21.0 - 8.5j,
        1.1 + 0.55j,
    ])

    np.testing.assert_allclose(tracked, expected)


def test_ambiguous_tracking():
    """Detecta associações igualmente prováveis."""

    tracker = ModeTracker()

    previous = np.array([
        1.0 + 0j,
        -1.0 + 0j,
        5.0 + 0j,
    ])

    current = np.array([
        0.0 + 0j,
        0.0 + 0j,
        5.1 + 0j,
    ])

    # Duas raízes coincidentes impedem uma associação única.
    with pytest.raises(RuntimeError, match="ambíguo"):
        tracker.track(previous, current)


@pytest.mark.parametrize(
    "tolerance",
    [-0.1, 1.0, float("nan"), float("inf")]
)
def test_invalid_tolerance(tolerance):
    """Rejeita tolerâncias inválidas."""

    with pytest.raises(ValueError):
        ModeTracker(ambiguity_tolerance=tolerance)


@pytest.mark.parametrize(
    "roots",
    [
        [1.0, 2.0],
        [1.0, 2.0, float("nan")],
        [1.0, 2.0, float("inf")],
    ]
)
def test_invalid_initial_roots(roots):
    """Rejeita conjuntos de raízes incompletos ou não finitos."""

    with pytest.raises(ValueError):
        ModeTracker.initialize_modes(roots)


def test_invalid_tracking_roots():
    """Rejeita entradas incompletas no rastreamento."""

    tracker = ModeTracker()

    valid = [100.0, 20.0, 2.0]
    invalid = [100.0, 20.0]

    with pytest.raises(ValueError):
        tracker.track(invalid, valid)

    with pytest.raises(ValueError):
        tracker.track(valid, invalid)


def test_original_arrays_preserved():
    """Verifica que o rastreamento não altera as entradas."""

    tracker = ModeTracker()

    previous = np.array([
        100.0 + 2.0j,
        20.0 + 1.0j,
        2.0 + 0.1j,
    ])

    current = np.array([
        2.1 + 0.1j,
        101.0 + 2.1j,
        21.0 + 1.2j,
    ])

    previous_copy = previous.copy()
    current_copy = current.copy()

    tracker.track(previous, current)

    np.testing.assert_array_equal(
        previous, previous_copy
    )

    np.testing.assert_array_equal(
        current, current_copy
    )
