
import pytest
from dataclasses import FrozenInstanceError

from models.mechanical_properties import MechanicalProperties


def create_valid_properties():
    """Cria um conjunto ilustrativo de propriedades válidas."""

    return MechanicalProperties(
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


def test_valid_properties():
    """Verifica a criação de um meio com parâmetros válidos."""

    properties = create_valid_properties()

    assert properties.phi == 0.25
    assert properties.rho_s == 2650.0
    assert properties.k0 == 1e-12


def test_invalid_porosity():
    """Verifica a rejeição de uma porosidade impossível."""

    properties = create_valid_properties()

    with pytest.raises(ValueError):
        MechanicalProperties(
            **{
                **vars(properties),
                "phi": 1.5,
            }
        )


def test_negative_viscosity():
    """Verifica a rejeição de uma viscosidade negativa."""

    properties = create_valid_properties()

    with pytest.raises(ValueError):
        MechanicalProperties(
            
            **{
                **vars(properties),
                "eta": -0.001,
            }
        )


def test_immutable_properties():
    """Verifica que as propriedades não podem ser alteradas."""

    properties = create_valid_properties()

    with pytest.raises(FrozenInstanceError):
        properties.phi = 0.40

# replace permite criar uma nova instância de uma dataclass,
# substituindo apenas os atributos que desejamos modificar.
#
# Isso é útil porque nossas classes utilizam frozen=True,
# impedindo a alteração direta dos atributos.
from dataclasses import replace

# Importa a classe que acabamos de implementar.
# O ponto indica o caminho:
# pacote models -> arquivo thermal_properties.py
from models.thermal_properties import ThermalProperties


# ============================================================
# TESTES DA CLASSE ThermalProperties
#
# OBJETIVO:
# Verificar se as propriedades térmicas são armazenadas
# corretamente e se valores inválidos são rejeitados.
#
# Os valores utilizados são apenas exemplos numéricos.
# Eles não representam os parâmetros definitivos do TCC.
# ============================================================


def create_valid_thermal_properties() -> ThermalProperties:
    """
    Cria um objeto ThermalProperties com parâmetros válidos.

    Utilizaremos este objeto como referência para os testes,
    evitando repetir todos os parâmetros em cada função.
    """

    return ThermalProperties(

        # Capacidade térmica volumétrica: J/(m³·K).
        C=2.3e6,

        # Acoplamentos térmicos: Pa/K.
        beta_s=150000.0,
        beta_f=50000.0,

        # Condutividade térmica: W/(m·K).
        kappa=2.0,

        # Tempo de relaxação térmica: segundos.
        tau0=0.001,

        # Temperatura absoluta de referência: kelvin.
        theta0=293.15
    )


# ------------------------------------------------------------
# TESTE 1: CRIAÇÃO COM VALORES VÁLIDOS
# ------------------------------------------------------------

def test_valid_thermal_properties():
    """
    Verifica se a classe aceita parâmetros térmicos válidos.
    """

    # Cria um objeto utilizando os valores definidos acima.
    properties = create_valid_thermal_properties()

    # assert verifica se uma condição é verdadeira.
    #
    # Caso a condição seja falsa, o Pytest registra
    # uma falha no teste.

    assert properties.C == 2.3e6
    assert properties.kappa == 2.0
    assert properties.theta0 == 293.15


# ------------------------------------------------------------
# TESTE 2: CAPACIDADE TÉRMICA NEGATIVA
# ------------------------------------------------------------

def test_negative_thermal_capacity():
    """
    Verifica se uma capacidade térmica negativa é rejeitada.
    """

    properties = create_valid_thermal_properties()

    # pytest.raises informa que esperamos uma exceção.
    #
    # Neste caso, esperamos ValueError.
    #
    # replace cria um NOVO objeto, substituindo C por -1000.
    # Durante a criação, __post_init__ executa validate().

    with pytest.raises(ValueError):
        replace(properties, C=-1000.0)


# ------------------------------------------------------------
# TESTE 3: VALOR INFINITO
# ------------------------------------------------------------

def test_infinite_thermal_capacity():
    """
    Verifica se a função isfinite impede valores infinitos.
    """

    properties = create_valid_thermal_properties()

    # float("inf") representa o infinito positivo.
    #
    # Como esse número não é finito, validate()
    # deve lançar um ValueError.

    with pytest.raises(ValueError):
        replace(properties, C=float("inf"))


# ------------------------------------------------------------
# TESTE 4: TEMPO DE RELAXAÇÃO NEGATIVO
# ------------------------------------------------------------

def test_negative_relaxation_time():
    """
    Verifica se tempos de relaxação negativos são rejeitados.
    """

    properties = create_valid_thermal_properties()

    # O modelo permite tau0 = 0 ou tau0 > 0.
    # Portanto, tau0 negativo deve provocar uma exceção.

    with pytest.raises(ValueError):
        replace(properties, tau0=-0.001)


# ------------------------------------------------------------
# TESTE 5: IMUTABILIDADE DOS PARÂMETROS
# ------------------------------------------------------------

def test_thermal_properties_are_immutable():
    """
    Verifica o comportamento de frozen=True.
    """

    properties = create_valid_thermal_properties()

    # FrozenInstanceError é a exceção esperada ao tentar
    # modificar um atributo de uma dataclass congelada.

    with pytest.raises(FrozenInstanceError):

        # Essa operação deve ser impedida pelo Python.
        properties.C = 1000.0
