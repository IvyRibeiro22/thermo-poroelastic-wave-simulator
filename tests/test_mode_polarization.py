
from dataclasses import replace
from math import sqrt

import numpy as np
import pytest

from models.mechanical_properties import MechanicalProperties
from models.thermal_properties import ThermalProperties
from models.porous_medium import PorousMedium
from models.simulation_result import ModeState

from physics.darcy_resistance import DarcyResistance
from physics.biot_jkd_resistance import BiotJKDResistance
from physics.reciprocal_r_model import ReciprocalRModel

from numerics.mode_classifier import ModeClassifier
from numerics.mode_polarization import (
    ModePolarization,
    ModePolarizationAnalyzer,
)
from numerics.polynomial_root_solver import PolynomialRootSolver


@pytest.fixture
def medium():
    """Cria o meio de referência."""

    mechanical = MechanicalProperties(
        phi=0.25,
        rho_s=2650.0,
        rho_f=1000.0,
        a_inf=2.0,
        k0=1e-12,
        eta=1e-3,
        lambda_=8e9,
        mu=6e9,
        m=1e10,
        alpha=0.8,
        viscous_length=1e-5,
    )

    thermal = ThermalProperties(
        C=2.3e6,
        beta_s=150000.0,
        beta_f=50000.0,
        kappa=2.0,
        tau0=0.001,
        theta0=293.15,
    )

    return PorousMedium(
        mechanical=mechanical,
        thermal=thermal,
    )


@pytest.fixture(
    params=[DarcyResistance, BiotJKDResistance],
    ids=["Darcy", "Biot-JKD"],
)
def resistance(request):
    """Seleciona a resistência viscosa."""

    return request.param()


@pytest.fixture(
    params=[True, False],
    ids=["acoplado", "desacoplado"],
)
def model(medium, resistance, request):
    """Cria modelos acoplados e desacoplados."""

    if request.param:
        selected_medium = medium
    else:
        thermal = replace(
            medium.thermal,
            beta_s=0.0,
            beta_f=0.0,
        )

        selected_medium = PorousMedium(
            mechanical=medium.mechanical,
            thermal=thermal,
        )

    return ReciprocalRModel(
        medium=selected_medium,
        resistance=resistance,
    )


def calculate_states(model, omega):
    """Calcula os três estados de propagação."""

    solver = PolynomialRootSolver()

    coefficients = model.polynomial_coefficients(
        omega
    )

    roots = solver.solve(coefficients)

    return tuple(
        ModeState.from_root(
            branch_index=index,
            omega=omega,
            z=complex(root),
        )
        for index, root in enumerate(roots)
    )


@pytest.mark.parametrize("omega", [1e3, 1e6])
def test_valid_polarizations(model, omega):
    """Verifica as três polarizações e seus resíduos."""

    analyzer = ModePolarizationAnalyzer(model)

    states = calculate_states(model, omega)

    for state in states:
        result = analyzer.analyze(state)

        assert isinstance(result, ModePolarization)
        assert result.omega == pytest.approx(omega)
        assert result.z == pytest.approx(state.z)

        amplitudes = np.asarray(
            result.amplitudes,
            dtype=complex,
        )

        assert amplitudes.shape == (3,)
        assert np.all(np.isfinite(amplitudes))

        np.testing.assert_allclose(
            np.linalg.norm(amplitudes),
            1.0,
            rtol=1e-10,
        )

        fractions = (
            result.solid_fraction,
            result.fluid_fraction,
            result.thermal_fraction,
        )

        assert all(
            0.0 <= value <= 1.0
            for value in fractions
        )

        assert sum(fractions) == pytest.approx(
            1.0,
            abs=1e-10,
        )

        assert np.isfinite(result.residual)
        assert result.residual < 1e-8

        expected_scale = sqrt(
            model.medium.A()
            * model.medium.thermal.theta0
            / model.medium.thermal.C
        )

        assert result.temperature_scale == pytest.approx(
            expected_scale
        )


def test_phase_convention_and_reproducibility(model):
    """Verifica a convenção de fase do autovetor."""

    state = calculate_states(model, 1e3)[0]

    analyzer = ModePolarizationAnalyzer(model)

    result_1 = analyzer.analyze(state)
    result_2 = analyzer.analyze(state)

    amplitudes = np.asarray(
        result_1.amplitudes,
        dtype=complex,
    )

    largest_index = int(
        np.argmax(np.abs(amplitudes))
    )

    largest = amplitudes[largest_index]

    assert largest.real > 0.0
    assert abs(largest.imag) < 1e-12

    np.testing.assert_allclose(
        result_1.amplitudes,
        result_2.amplitudes,
        rtol=1e-10,
        atol=1e-12,
    )


def test_uncoupled_modes(medium, resistance):
    """Verifica a separação térmica e mecânica."""

    thermal = replace(
        medium.thermal,
        beta_s=0.0,
        beta_f=0.0,
    )

    uncoupled_medium = PorousMedium(
        mechanical=medium.mechanical,
        thermal=thermal,
    )

    model = ReciprocalRModel(
        medium=uncoupled_medium,
        resistance=resistance,
    )

    omega = 1e3
    states = calculate_states(model, omega)

    roots = np.array(
        [state.z for state in states],
        dtype=complex,
    )

    mapping = ModeClassifier(
        model
    ).classify_uncoupled(
        omega,
        roots,
    )

    analyzer = ModePolarizationAnalyzer(model)

    # O modo térmico deve ser puramente térmico.
    thermal_state = states[mapping["T"]]
    thermal_result = analyzer.analyze(thermal_state)

    assert thermal_result.thermal_fraction == (
        pytest.approx(1.0, abs=1e-10)
    )

    assert thermal_result.solid_fraction < 1e-10
    assert thermal_result.fluid_fraction < 1e-10

    # Os modos poroelásticos não devem apresentar
    # participação térmica quando o acoplamento é nulo.
    for name in ("P1", "P2"):
        state = states[mapping[name]]
        result = analyzer.analyze(state)

        assert result.thermal_fraction < 1e-10


def test_coupled_modes(medium, resistance):
    """Verifica a mistura das componentes no modelo acoplado."""

    model = ReciprocalRModel(
        medium=medium,
        resistance=resistance,
    )

    analyzer = ModePolarizationAnalyzer(model)

    states = calculate_states(model, 1e3)

    results = [
        analyzer.analyze(state)
        for state in states
    ]

    # Com os parâmetros de referência, todos os
    # modos apresentam participação térmica.
    assert all(
        result.thermal_fraction > 1e-10
        for result in results
    )

    # Existe um modo predominantemente térmico
    # neste conjunto particular de parâmetros.
    assert any(
        result.thermal_fraction > 0.9
        for result in results
    )


def test_original_objects_are_preserved(model):
    """Não modifica o meio nem o estado original."""

    state = calculate_states(model, 1e3)[0]

    original_medium = model.medium
    original_state = state

    analyzer = ModePolarizationAnalyzer(model)

    analyzer.analyze(state)

    assert model.medium is original_medium
    assert state is original_state


def test_invalid_state(model):
    """Rejeita objetos que não são ModeState."""

    analyzer = ModePolarizationAnalyzer(model)

    with pytest.raises(TypeError):
        analyzer.analyze(None)


def test_invalid_model():
    """Rejeita um modelo incompatível."""

    with pytest.raises(TypeError):
        ModePolarizationAnalyzer(model=None)


@pytest.mark.parametrize(
    "tolerance",
    [0.0, -1e-8, float("nan"), float("inf")],
)
def test_invalid_residual_tolerance(
    medium,
    tolerance
):
    """Rejeita tolerâncias residuais inválidas."""

    model = ReciprocalRModel(
        medium=medium,
        resistance=DarcyResistance(),
    )

    with pytest.raises(ValueError):
        ModePolarizationAnalyzer(
            model=model,
            residual_tolerance=tolerance,
        )


@pytest.mark.parametrize(
    "tolerance",
    [
        0.0,
        -0.1,
        1.0,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_ambiguity_tolerance(
    medium,
    tolerance
):
    """Rejeita tolerâncias de ambiguidade inválidas."""

    model = ReciprocalRModel(
        medium=medium,
        resistance=DarcyResistance(),
    )

    with pytest.raises(ValueError):
        ModePolarizationAnalyzer(
            model=model,
            ambiguity_tolerance=tolerance,
        )


@pytest.mark.parametrize(
    "bad_matrix",
    [
        np.eye(2, dtype=complex),
        np.full((3, 3), complex(float("nan"), 0.0)),
        np.zeros((3, 3), dtype=complex),
    ],
    ids=["wrong-shape", "nonfinite", "zero-matrix"],
)
def test_invalid_dispersion_matrix(
    medium,
    monkeypatch,
    bad_matrix
):
    """Rejeita matrizes incompatíveis."""

    model = ReciprocalRModel(
        medium=medium,
        resistance=DarcyResistance(),
    )

    state = ModeState.from_root(
        branch_index=0,
        omega=1e3,
        z=1e6 + 0j,
    )

    def invalid_matrix(self, omega, k):
        return bad_matrix.copy()

    monkeypatch.setattr(
        ReciprocalRModel,
        "dispersion_matrix",
        invalid_matrix,
    )

    analyzer = ModePolarizationAnalyzer(model)

    with pytest.raises(ArithmeticError):
        analyzer.analyze(state)


def test_ambiguous_eigenspace(medium, monkeypatch):
    """Rejeita direções nulas não distinguíveis."""

    model = ReciprocalRModel(
        medium=medium,
        resistance=DarcyResistance(),
    )

    state = calculate_states(model, 1e3)[0]

    def ambiguous_svd(matrix):
        return (
            np.eye(3, dtype=complex),
            np.array([2.0, 1e-12, 0.0]),
            np.eye(3, dtype=complex),
        )

    monkeypatch.setattr(
        np.linalg,
        "svd",
        ambiguous_svd,
    )

    analyzer = ModePolarizationAnalyzer(model)

    with pytest.raises(
        RuntimeError,
        match="ambíguo",
    ):
        analyzer.analyze(state)


def test_excessive_matrix_residual(medium, monkeypatch):
    """Rejeita um vetor que não satisfaz a matriz."""

    model = ReciprocalRModel(
        medium=medium,
        resistance=DarcyResistance(),
    )

    state = calculate_states(model, 1e3)[0]

    def incorrect_svd(matrix):
        return (
            np.eye(3, dtype=complex),
            np.array([2.0, 1.0, 0.0]),
            np.eye(3, dtype=complex),
        )

    monkeypatch.setattr(
        np.linalg,
        "svd",
        incorrect_svd,
    )

    analyzer = ModePolarizationAnalyzer(model)

    with pytest.raises(
        ArithmeticError,
        match="não satisfaz",
    ):
        analyzer.analyze(state)
