
from dataclasses import replace
from math import sqrt

import numpy as np
import pytest

from models.mechanical_properties import MechanicalProperties
from models.thermal_properties import ThermalProperties
from models.porous_medium import PorousMedium

from physics.darcy_resistance import DarcyResistance
from physics.biot_jkd_resistance import BiotJKDResistance
from physics.reciprocal_r_model import ReciprocalRModel

from numerics.coupling_continuation import CouplingContinuation
from numerics.frequency_grid import FrequencyGrid
from numerics.mode_polarization import (
    ModePolarization,
    ModePolarizationAnalyzer,
)

from simulation.identified_mode_analysis import (
    IdentifiedModeAnalyzer,
    IdentifiedModeSample,
)
from simulation.thermo_poroelastic_wave_simulator import (
    ThermoPoroelasticWaveSimulator,
)


@pytest.fixture(
    params=[DarcyResistance, BiotJKDResistance],
    ids=["Darcy", "Biot-JKD"],
)
def simulator(request):
    """Cria o simulador para ambas as resistências."""

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

    medium = PorousMedium(
        mechanical=mechanical,
        thermal=thermal,
    )

    model = ReciprocalRModel(
        medium=medium,
        resistance=request.param(),
    )

    return ThermoPoroelasticWaveSimulator(model)


@pytest.fixture
def grid():
    """Cria uma malha pequena de frequências."""

    return FrequencyGrid.build(
        omega_min=1000.0,
        omega_max=8000.0,
        num_points=4,
        use_logspace=True,
    )


@pytest.fixture
def analyzer(simulator):
    """Cria o analisador integrado."""

    return IdentifiedModeAnalyzer(simulator)


def create_polarization(amplitudes):
    """Cria uma polarização para testar a similaridade."""

    values = tuple(
        complex(value)
        for value in amplitudes
    )

    fractions = [
        abs(value) ** 2
        for value in values
    ]

    return ModePolarization(
        omega=1000.0,
        z=1e6 + 0j,
        amplitudes=values,
        solid_fraction=fractions[0],
        fluid_fraction=fractions[1],
        thermal_fraction=fractions[2],
        temperature_scale=1.0,
        residual=0.0,
    )


def test_result_structure(analyzer, grid):
    """Verifica a estrutura da análise integrada."""

    result = analyzer.run(grid)

    assert set(result) == {
        "omega", "P1", "P2", "T"
    }

    assert isinstance(result["omega"], tuple)

    np.testing.assert_allclose(
        result["omega"],
        grid.omega,
        rtol=1e-12,
    )

    for name in ("P1", "P2", "T"):

        samples = result[name]

        assert isinstance(samples, tuple)
        assert len(samples) == len(grid.omega)

        assert all(
            isinstance(sample, IdentifiedModeSample)
            for sample in samples
        )


def test_velocities_match_identified_simulation(
    analyzer,
    simulator,
    grid,
):
    """Compara as velocidades dos dois procedimentos."""

    analysis = analyzer.run(grid)

    identified = simulator.run_identified(grid)

    for name in ("P1", "P2", "T"):

        velocities = tuple(
            sample.state.phase_velocity
            for sample in analysis[name]
        )

        np.testing.assert_allclose(
            velocities,
            identified[name],
            rtol=1e-10,
        )


def test_polarization_consistency(
    analyzer,
    simulator,
    grid,
):
    """Verifica cada estado e sua polarização."""

    result = analyzer.run(grid)

    direct_analyzer = ModePolarizationAnalyzer(
        simulator.model
    )

    for name in ("P1", "P2", "T"):

        for index, sample in enumerate(result[name]):

            state = sample.state
            polarization = sample.polarization

            assert state.omega == pytest.approx(
                grid.omega[index]
            )

            assert polarization.omega == pytest.approx(
                state.omega
            )

            assert polarization.z == pytest.approx(
                state.z,
                rel=1e-10,
            )

            # Verifica a normalização das amplitudes.
            amplitudes = np.asarray(
                polarization.amplitudes,
                dtype=complex,
            )

            assert amplitudes.shape == (3,)
            assert np.all(np.isfinite(amplitudes))

            assert np.linalg.norm(
                amplitudes
            ) == pytest.approx(
                1.0,
                rel=1e-10,
            )

            fractions = (
                polarization.solid_fraction,
                polarization.fluid_fraction,
                polarization.thermal_fraction,
            )

            assert sum(fractions) == pytest.approx(
                1.0,
                abs=1e-10,
            )

            assert all(
                0.0 <= value <= 1.0
                for value in fractions
            )

            assert polarization.residual < 1e-8

            # Confere com uma análise independente.
            direct = direct_analyzer.analyze(state)

            np.testing.assert_allclose(
                polarization.amplitudes,
                direct.amplitudes,
                rtol=1e-10,
                atol=1e-12,
            )


def test_roots_match_coupling_continuation(
    analyzer,
    simulator,
    grid,
):
    """Compara as raízes com a continuação térmica."""

    result = analyzer.run(grid)

    continuation = CouplingContinuation(
        simulator.model
    )

    for index, omega in enumerate(result["omega"]):

        reference = continuation.identify(
            omega=omega,
            num_steps=51,
        )

        for name in ("P1", "P2", "T"):

            state = result[name][index].state

            assert state.z == pytest.approx(
                reference[name],
                rel=1e-7,
            )


def test_overlap_between_frequencies(
    analyzer,
    grid,
):
    """Verifica a similaridade dos autovetores."""

    result = analyzer.run(grid)

    for name in ("P1", "P2", "T"):

        samples = result[name]

        # A primeira frequência não tem antecessora.
        assert samples[0].overlap_previous is None

        for index in range(1, len(samples)):

            previous = samples[index - 1]
            current = samples[index]

            overlap = current.overlap_previous

            assert overlap is not None
            assert np.isfinite(overlap)
            assert 0.0 <= overlap <= 1.0

            a = np.asarray(
                previous.polarization.amplitudes,
                dtype=complex,
            )

            b = np.asarray(
                current.polarization.amplitudes,
                dtype=complex,
            )

            expected = (
                abs(np.vdot(a, b))
                / (
                    np.linalg.norm(a)
                    * np.linalg.norm(b)
                )
            )

            assert overlap == pytest.approx(
                expected,
                abs=1e-12,
            )


def test_original_objects_are_preserved(
    analyzer,
    simulator,
    grid,
):
    """Garante que a análise não modifique o modelo."""

    original_model = simulator.model
    original_medium = original_model.medium

    original_thermal = original_medium.thermal
    original_mechanical = original_medium.mechanical

    result_before = simulator.run(grid)

    analyzer.run(grid)

    result_after = simulator.run(grid)

    assert simulator.model is original_model
    assert simulator.model.medium is original_medium

    assert (
        simulator.model.medium.thermal
        is original_thermal
    )

    assert (
        simulator.model.medium.mechanical
        is original_mechanical
    )

    assert result_before == result_after


def test_uncoupled_mode_composition(
    simulator,
    grid,
):
    """Verifica as participações no limite desacoplado."""

    original_medium = simulator.model.medium

    thermal = replace(
        original_medium.thermal,
        beta_s=0.0,
        beta_f=0.0,
    )

    medium = PorousMedium(
        mechanical=original_medium.mechanical,
        thermal=thermal,
    )

    model = ReciprocalRModel(
        medium=medium,
        resistance=simulator.model.resistance,
    )

    uncoupled_simulator = (
        ThermoPoroelasticWaveSimulator(model)
    )

    analysis = IdentifiedModeAnalyzer(
        uncoupled_simulator
    )

    result = analysis.run(grid)

    for sample in result["T"]:

        polarization = sample.polarization

        assert polarization.thermal_fraction == pytest.approx(
            1.0,
            abs=1e-10,
        )

        assert polarization.solid_fraction < 1e-10
        assert polarization.fluid_fraction < 1e-10

    for name in ("P1", "P2"):

        for sample in result[name]:

            assert (
                sample.polarization.thermal_fraction
                < 1e-10
            )


def test_invalid_simulator():
    """Rejeita objetos incompatíveis."""

    with pytest.raises(TypeError):
        IdentifiedModeAnalyzer(simulator=None)


def test_invalid_grid(analyzer):
    """Rejeita uma malha inválida."""

    with pytest.raises(TypeError):
        analyzer.run(grid=None)


@pytest.mark.parametrize(
    "num_steps",
    [0, 1, -1, True, 2.5],
)
def test_invalid_step_count(
    analyzer,
    grid,
    num_steps,
):
    """Rejeita quantidades inválidas de passos."""

    with pytest.raises(ValueError):
        analyzer.run(
            grid,
            num_steps=num_steps,
        )


def test_continuation_failure(
    analyzer,
    grid,
    monkeypatch,
):
    """Não oculta erros de continuação."""

    def fail_identification(self, omega, num_steps=51):
        raise RuntimeError(
            "Falha proposital na continuação."
        )

    monkeypatch.setattr(
        CouplingContinuation,
        "identify",
        fail_identification,
    )

    with pytest.raises(
        RuntimeError,
        match="Falha proposital",
    ):
        analyzer.run(grid)


def test_polarization_failure(
    analyzer,
    grid,
    monkeypatch,
):
    """Não oculta erros na análise dos autovetores."""

    def fail_analysis(self, state):
        raise ArithmeticError(
            "Falha proposital na polarização."
        )

    monkeypatch.setattr(
        ModePolarizationAnalyzer,
        "analyze",
        fail_analysis,
    )

    with pytest.raises(
        ArithmeticError,
        match="Falha proposital",
    ):
        analyzer.run(grid)


def test_change_of_mode_mapping(
    analyzer,
    grid,
    monkeypatch,
):
    """Detecta uma troca inesperada entre os modos."""

    original_identify = CouplingContinuation.identify

    call_count = 0

    def changed_identification(self, omega, num_steps=51):

        nonlocal call_count

        call_count += 1

        result = original_identify(
            self,
            omega,
            num_steps=num_steps,
        )

        # Simula a troca de P1 e P2
        # na segunda frequência.
        if call_count == 2:

            result["P1"], result["P2"] = (
                result["P2"],
                result["P1"],
            )

        return result

    monkeypatch.setattr(
        CouplingContinuation,
        "identify",
        changed_identification,
    )

    with pytest.raises(
        RuntimeError,
        match="identificação dos modos mudou",
    ):
        analyzer.run(grid)


@pytest.mark.parametrize(
    "first, second, expected",
    [
        (
            (1, 0, 0),
            (1, 0, 0),
            1.0,
        ),
        (
            (1, 0, 0),
            (0, 1, 0),
            0.0,
        ),
        (
            (1j, 0, 0),
            (1, 0, 0),
            1.0,
        ),
        (
            (1 / sqrt(2), 1 / sqrt(2), 0),
            (1, 0, 0),
            1 / sqrt(2),
        ),
    ],
    ids=[
        "identical",
        "orthogonal",
        "global-phase",
        "partial-overlap",
    ],
)
def test_overlap_formula(
    first,
    second,
    expected,
):
    """Verifica o cálculo matemático da similaridade."""

    previous = create_polarization(first)
    current = create_polarization(second)

    overlap = IdentifiedModeAnalyzer._overlap(
        previous,
        current,
    )

    assert overlap == pytest.approx(
        expected,
        abs=1e-12,
    )


def test_overlap_rejects_zero_vector():
    """Rejeita vetores sem norma definida."""

    previous = create_polarization(
        (0, 0, 0)
    )

    current = create_polarization(
        (1, 0, 0)
    )

    with pytest.raises(ArithmeticError):
        IdentifiedModeAnalyzer._overlap(
            previous,
            current,
        )
