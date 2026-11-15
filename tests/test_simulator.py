
import numpy as np
import pytest

from models.mechanical_properties import MechanicalProperties
from models.thermal_properties import ThermalProperties
from models.porous_medium import PorousMedium
from models.simulation_result import SimulationResult

from physics.darcy_resistance import DarcyResistance
from physics.biot_jkd_resistance import BiotJKDResistance
from physics.reciprocal_r_model import ReciprocalRModel

from numerics.frequency_grid import FrequencyGrid
from numerics.polynomial_root_solver import PolynomialRootSolver

from simulation.thermo_poroelastic_wave_simulator import (
    ThermoPoroelasticWaveSimulator,
)


@pytest.fixture
def medium():
    """Cria o meio poroso de referência."""

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
def simulator(medium, request):
    """Cria o simulador para os dois modelos."""

    model = ReciprocalRModel(
        medium=medium,
        resistance=request.param(),
    )

    return ThermoPoroelasticWaveSimulator(model)


@pytest.fixture
def grid():
    """Cria uma malha de quatro frequências."""

    return FrequencyGrid.build(
        omega_min=1000.0,
        omega_max=8000.0,
        num_points=4,
        use_logspace=True,
    )


def test_complete_simulation(simulator, grid):
    """Verifica a execução integrada do simulador."""

    result = simulator.run(grid)

    assert isinstance(result, SimulationResult)
    assert len(result.omega) == 4
    assert len(result.states_by_frequency) == 4

    for omega, states in zip(
        result.omega,
        result.states_by_frequency,
    ):
        assert len(states) == 3

        for index, state in enumerate(states):
            assert state.branch_index == index
            assert state.omega == pytest.approx(omega)


def test_simulation_root_residuals(simulator, grid):
    """Verifica os resíduos das raízes calculadas."""

    result = simulator.run(grid)
    solver = PolynomialRootSolver()

    for omega, states in zip(
        result.omega,
        result.states_by_frequency,
    ):
        coefficients = (
            simulator.model.polynomial_coefficients(omega)
        )

        roots = np.array(
            [state.z for state in states],
            dtype=complex,
        )

        residuals = solver.check_residuals(
            coefficients,
            roots,
        )

        assert np.all(np.isfinite(residuals))
        assert np.max(residuals) < 1e-8


def test_wavenumber_consistency(simulator, grid):
    """Verifica a relação entre z e k."""

    result = simulator.run(grid)

    for states in result.states_by_frequency:
        for state in states:

            reconstructed_z = (
                state.omega / state.k
            ) ** 2

            assert reconstructed_z == pytest.approx(
                state.z,
                rel=1e-10,
            )


def test_phase_velocity_consistency(simulator, grid):
    """Verifica as velocidades de fase calculadas."""

    result = simulator.run(grid)

    for states in result.states_by_frequency:
        for state in states:

            expected_velocity = (
                state.omega / state.k.real
            )

            assert state.phase_velocity == pytest.approx(
                expected_velocity
            )

            assert np.isfinite(state.phase_velocity)
            assert state.phase_velocity > 0.0


def test_positive_propagation_direction(simulator, grid):
    """Verifica a convenção dos números de onda."""

    result = simulator.run(grid)

    for states in result.states_by_frequency:
        for state in states:

            assert state.k.real > 0.0
            assert state.k.imag >= 0.0


def test_phase_velocity_extraction(simulator, grid):
    """Recupera as velocidades dos três ramos."""

    result = simulator.run(grid)

    for branch_index in range(3):

        velocities = result.values_for_branch(
            branch_index=branch_index,
            quantity="phase_velocity",
        )

        assert len(velocities) == len(result.omega)
        assert all(
            np.isfinite(value) and value > 0.0
            for value in velocities
        )


def test_darcy_and_jkd_are_distinct(medium):
    """Compara as duas resistências em alta frequência."""

    grid = FrequencyGrid.build(
        omega_min=1e6,
        omega_max=2e6,
        num_points=2,
        use_logspace=False,
    )

    darcy_model = ReciprocalRModel(
        medium=medium,
        resistance=DarcyResistance(),
    )

    jkd_model = ReciprocalRModel(
        medium=medium,
        resistance=BiotJKDResistance(),
    )

    darcy_result = ThermoPoroelasticWaveSimulator(
        darcy_model
    ).run(grid)

    jkd_result = ThermoPoroelasticWaveSimulator(
        jkd_model
    ).run(grid)

    darcy_roots = np.array([
        state.z
        for state in darcy_result.states_by_frequency[0]
    ])

    jkd_roots = np.array([
        state.z
        for state in jkd_result.states_by_frequency[0]
    ])

    assert not np.allclose(
        darcy_roots,
        jkd_roots,
        rtol=1e-5,
        atol=0.0,
    )


def test_simulation_without_attenuation_outputs(
    simulator,
    grid,
):
    """Verifica as grandezas disponíveis nos resultados."""

    result = simulator.run(grid)

    for states in result.states_by_frequency:
        for state in states:
            assert not hasattr(state, "attenuation")
            assert not hasattr(state, "q_inverse")

    with pytest.raises(ValueError):
        result.values_for_branch(0, "attenuation")

    with pytest.raises(ValueError):
        result.values_for_branch(0, "q_inverse")


def test_invalid_model():
    """Rejeita um modelo físico inválido."""

    with pytest.raises(TypeError):
        ThermoPoroelasticWaveSimulator(model=None)


def test_invalid_grid(simulator):
    """Rejeita uma malha de frequências inválida."""

    with pytest.raises(TypeError):
        simulator.run(grid=None)


@pytest.mark.parametrize(
    "tolerance",
    [
        0.0,
        -1e-8,
        float("nan"),
        float("inf"),
    ],
)
def test_invalid_residual_tolerance(
    medium,
    tolerance,
):
    """Rejeita tolerâncias residuais inválidas."""

    model = ReciprocalRModel(
        medium=medium,
        resistance=DarcyResistance(),
    )

    with pytest.raises(ValueError):
        ThermoPoroelasticWaveSimulator(
            model=model,
            residual_tolerance=tolerance,
        )
