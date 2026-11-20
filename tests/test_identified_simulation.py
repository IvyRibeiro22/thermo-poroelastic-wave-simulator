
import numpy as np
import pytest

from models.mechanical_properties import MechanicalProperties
from models.thermal_properties import ThermalProperties
from models.porous_medium import PorousMedium
from models.simulation_result import ModeState

from physics.darcy_resistance import DarcyResistance
from physics.biot_jkd_resistance import BiotJKDResistance
from physics.reciprocal_r_model import ReciprocalRModel

from numerics.frequency_grid import FrequencyGrid
from numerics.coupling_continuation import CouplingContinuation

from simulation.thermo_poroelastic_wave_simulator import (
    ThermoPoroelasticWaveSimulator,
)


@pytest.fixture(
    params=[DarcyResistance, BiotJKDResistance],
    ids=["Darcy", "Biot-JKD"],
)
def simulator(request):
    """Cria o simulador para as duas resistências."""

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
    """Cria a malha de frequências do teste."""

    return FrequencyGrid.build(
        omega_min=1000.0,
        omega_max=8000.0,
        num_points=4,
        use_logspace=True,
    )


def test_identified_result_structure(simulator, grid):
    """Verifica a estrutura dos resultados identificados."""

    result = simulator.run_identified(grid)

    assert set(result) == {"omega", "P1", "P2", "T"}

    assert isinstance(result["omega"], tuple)

    np.testing.assert_allclose(
        result["omega"],
        grid.omega,
        rtol=1e-12,
    )

    for name in ("P1", "P2", "T"):
        velocities = result[name]

        assert isinstance(velocities, tuple)
        assert len(velocities) == len(grid.omega)

        assert all(
            np.isfinite(value) and value > 0.0
            for value in velocities
        )


def test_identified_velocities_match_raw_simulation(
    simulator,
    grid
):
    """Compara as velocidades identificadas com os três ramos."""

    raw = simulator.run(grid)
    identified = simulator.run_identified(grid)

    for index, states in enumerate(raw.states_by_frequency):

        raw_velocities = np.array([
            state.phase_velocity
            for state in states
        ])

        identified_velocities = np.array([
            identified[name][index]
            for name in ("P1", "P2", "T")
        ])

        np.testing.assert_allclose(
            np.sort(raw_velocities),
            np.sort(identified_velocities),
            rtol=1e-12,
        )


def test_each_frequency_matches_continuation(
    simulator,
    grid
):
    """Confere a identificação em todos os pontos da malha."""

    identified = simulator.run_identified(grid)

    continuation = CouplingContinuation(
        simulator.model
    )

    for frequency_index, omega in enumerate(
        identified["omega"]
    ):

        reference = continuation.identify(
            omega=omega,
            num_steps=51,
        )

        for branch_index, name in enumerate(
            ("P1", "P2", "T")
        ):

            expected_state = ModeState.from_root(
                branch_index=branch_index,
                omega=omega,
                z=reference[name],
            )

            assert identified[name][frequency_index] == (
                pytest.approx(
                    expected_state.phase_velocity,
                    rel=1e-9,
                )
            )


def test_identification_stable_with_more_steps(
    simulator,
    grid
):
    """Compara duas discretizações do acoplamento."""

    result_11 = simulator.run_identified(
        grid,
        num_steps=11,
    )

    result_51 = simulator.run_identified(
        grid,
        num_steps=51,
    )

    for name in ("P1", "P2", "T"):

        np.testing.assert_allclose(
            result_11[name],
            result_51[name],
            rtol=1e-7,
        )


def test_original_run_is_preserved(simulator, grid):
    """Verifica que a identificação não altera o método run."""

    result_before = simulator.run(grid)

    simulator.run_identified(grid)

    result_after = simulator.run(grid)

    assert result_before == result_after


def test_original_parameters_are_not_modified(
    simulator,
    grid
):
    """Preserva o material fornecido ao simulador."""

    original_medium = simulator.model.medium
    original_thermal = original_medium.thermal
    original_mechanical = original_medium.mechanical

    simulator.run_identified(grid)

    assert simulator.model.medium is original_medium

    assert (
        simulator.model.medium.thermal
        is original_thermal
    )

    assert (
        simulator.model.medium.mechanical
        is original_mechanical
    )

    assert original_thermal.beta_s == 150000.0
    assert original_thermal.beta_f == 50000.0


def test_invalid_grid(simulator):
    """Rejeita malhas incompatíveis."""

    with pytest.raises(TypeError):
        simulator.run_identified(grid=None)


@pytest.mark.parametrize(
    "num_steps",
    [0, 1, -1, True, 2.5],
)
def test_invalid_step_count(
    simulator,
    grid,
    num_steps
):
    """Rejeita quantidades inválidas de passos."""

    with pytest.raises(ValueError):
        simulator.run_identified(
            grid,
            num_steps=num_steps,
        )


def test_continuation_error_is_propagated(
    simulator,
    grid,
    monkeypatch
):
    """Não oculta falhas de continuação."""

    def fail_identification(self, omega, num_steps=51):
        raise RuntimeError(
            "Continuação interrompida."
        )

    monkeypatch.setattr(
        CouplingContinuation,
        "identify",
        fail_identification,
    )

    with pytest.raises(
        RuntimeError,
        match="Continuação interrompida",
    ):
        simulator.run_identified(grid)


def test_change_of_mode_mapping_is_detected(
    simulator,
    grid,
    monkeypatch
):
    """Detecta uma troca de correspondência entre os modos."""

    original_identify = CouplingContinuation.identify

    call_count = 0

    def altered_identification(self, omega, num_steps=51):
        nonlocal call_count

        call_count += 1

        result = original_identify(
            self,
            omega,
            num_steps=num_steps,
        )

        # Simula uma troca de P1 e P2 na segunda verificação.
        if call_count == 2:
            result["P1"], result["P2"] = (
                result["P2"],
                result["P1"],
            )

        return result

    monkeypatch.setattr(
        CouplingContinuation,
        "identify",
        altered_identification,
    )

    with pytest.raises(
        RuntimeError,
        match="correspondência dos modos mudou",
    ):
        simulator.run_identified(grid)


def test_matching_detects_ambiguity():
    """Rejeita correspondências não distinguíveis."""

    omega = 1000.0

    roots = (
        4e6 + 0j,
        4e6 + 0j,
        1e6 + 0j,
    )

    states = tuple(
        ModeState.from_root(
            branch_index=index,
            omega=omega,
            z=root,
        )
        for index, root in enumerate(roots)
    )

    reference = {
        "P1": 4e6 + 0j,
        "P2": 4e6 + 0j,
        "T": 1e6 + 0j,
    }

    with pytest.raises(
        RuntimeError,
        match="ambígua",
    ):
        ThermoPoroelasticWaveSimulator._match_mode_indices(
            reference,
            states,
        )


def test_matching_rejects_wrong_roots():
    """Rejeita referências incompatíveis com a simulação."""

    omega = 1000.0

    roots = (
        4e6 + 0j,
        1e6 + 0j,
        100.0 + 0j,
    )

    states = tuple(
        ModeState.from_root(
            branch_index=index,
            omega=omega,
            z=root,
        )
        for index, root in enumerate(roots)
    )

    reference = {
        "P1": 1e9 + 0j,
        "P2": 1e8 + 0j,
        "T": 1e7 + 0j,
    }

    with pytest.raises(
        RuntimeError,
        match="não correspondem",
    ):
        ThermoPoroelasticWaveSimulator._match_mode_indices(
            reference,
            states,
        )


def test_matching_independent_of_root_order():
    """Identifica os modos mesmo com raízes reordenadas."""

    omega = 1000.0

    roots = (
        2e6 + 0j,
        1e6 + 0j,
        4e6 + 0j,
    )

    states = tuple(
        ModeState.from_root(
            branch_index=index,
            omega=omega,
            z=root,
        )
        for index, root in enumerate(roots)
    )

    reference = {
        "P1": 4e6 + 0j,
        "P2": 2e6 + 0j,
        "T": 1e6 + 0j,
    }

    mapping = (
        ThermoPoroelasticWaveSimulator._match_mode_indices(
            reference,
            states,
        )
    )

    assert mapping == {
        "P1": 2,
        "P2": 0,
        "T": 1,
    }
