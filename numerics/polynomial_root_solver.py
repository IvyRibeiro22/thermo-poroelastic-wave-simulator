
import numpy as np


class PolynomialRootSolver:
    """
    Resolve polinômios cúbicos com coeficientes complexos.

    Utilizado para calcular as raízes da equação
    característica do modelo termo-poroelástico R.
    """

    @staticmethod
    def _validate_coefficients(coefficients) -> np.ndarray:
        """Verifica os coeficientes do polinômio cúbico."""

        try:
            c = np.asarray(coefficients, dtype=np.complex128)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "Os coeficientes devem ser numéricos."
            ) from exc

        if c.shape != (4,):
            raise ValueError(
                "O polinômio cúbico deve ter quatro coeficientes."
            )

        if not np.all(np.isfinite(c)):
            raise ValueError(
                "Os coeficientes devem ser finitos."
            )

        if c[0] == 0:
            raise ValueError(
                "O coeficiente cúbico não pode ser zero."
            )

        return c

    def solve(self, coefficients) -> np.ndarray:
        """
        Calcula as três raízes complexas do polinômio.

        Entrada: [c3, c2, c1, c0].
        Retorno: array com três raízes complexas.

        O escalonamento melhora o condicionamento numérico
        quando os coeficientes têm magnitudes diferentes.
        """

        c = self._validate_coefficients(coefficients)

        # Normaliza os coeficientes pelo termo cúbico.
        relative = c / c[0]

        # Escala característica da variável z.
        scale = max(
            1.0,
            abs(relative[1]),
            np.sqrt(abs(relative[2])),
            np.cbrt(abs(relative[3])),
        )

        if not np.isfinite(scale):
            raise ValueError(
                "Não foi possível calcular uma escala finita."
            )

        # Substituição z = scale * y.
        scaled_coefficients = np.array([
            1.0,
            relative[1] / scale,
            relative[2] / scale**2,
            relative[3] / scale**3,
        ], dtype=complex)

        # Resolve o polinômio na variável escalonada y.
        roots_scaled = np.roots(scaled_coefficients)

        # Recupera as raízes na variável original z.
        roots = roots_scaled * scale

        if roots.shape != (3,) or not np.all(np.isfinite(roots)):
            raise ArithmeticError(
                "Não foi possível obter três raízes finitas."
            )

        return roots

    def check_residuals(
        self,
        coefficients,
        roots
    ) -> np.ndarray:
        """
        Calcula o resíduo relativo de cada raiz.

        Quanto menor o resíduo, melhor a satisfação
        numérica da equação característica.
        """

        c = self._validate_coefficients(coefficients)
        z = np.asarray(roots, dtype=complex)

        if z.shape != (3,) or not np.all(np.isfinite(z)):
            raise ValueError(
                "São necessárias três raízes complexas finitas."
            )

        # Avalia P(z) nas raízes calculadas.
        numerator = np.abs(np.polyval(c, z))

        # Normalização pela soma dos módulos dos termos.
        magnitude = np.abs(z)

        denominator = (
            abs(c[0]) * magnitude**3
            + abs(c[1]) * magnitude**2
            + abs(c[2]) * magnitude
            + abs(c[3])
        )

        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator),
            where=denominator > 0
        )
