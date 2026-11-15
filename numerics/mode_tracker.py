
from itertools import permutations

import numpy as np


class ModeTracker:
    """
    Acompanha três raízes complexas ao longo das frequências.

    Associa as raízes de frequências consecutivas pela
    proximidade relativa, preservando a continuidade.

    Os ramos ainda não são classificados como P1, P2 ou T.
    """

    def __init__(self, ambiguity_tolerance: float = 1e-3):
        """
        Define a tolerância para detectar associações ambíguas.
        """

        if (
            not np.isfinite(ambiguity_tolerance)
            or not 0.0 <= ambiguity_tolerance < 1.0
        ):
            raise ValueError(
                "A tolerância deve ser finita e estar entre 0 e 1."
            )

        self.ambiguity_tolerance = ambiguity_tolerance

    @staticmethod
    def _validate_roots(roots) -> np.ndarray:
        """Verifica se foram fornecidas três raízes finitas."""

        try:
            values = np.asarray(roots, dtype=complex)
        except (TypeError, ValueError) as exc:
            raise TypeError(
                "As raízes devem ser numéricas."
            ) from exc

        if values.shape != (3,):
            raise ValueError(
                "São necessárias exatamente três raízes."
            )

        if not np.all(np.isfinite(values)):
            raise ValueError(
                "As raízes devem ser finitas."
            )

        return values

    @classmethod
    def initialize_modes(cls, roots) -> np.ndarray:
        """
        Organiza as raízes da primeira frequência.

        A ordenação inicial usa o módulo de z apenas
        para estabelecer identificadores de ramos.
        """

        values = cls._validate_roots(roots)

        order = np.argsort(
            -np.abs(values),
            kind="stable"
        )

        return values[order].copy()

    @staticmethod
    def _relative_distance(
        previous: complex,
        current: complex
    ) -> float:
        """Calcula a distância relativa entre duas raízes."""

        scale = max(
            abs(previous),
            abs(current),
            np.finfo(float).tiny
        )

        return abs(current - previous) / scale

    def track(
        self,
        previous_roots,
        current_roots
    ) -> np.ndarray:
        """
        Associa as raízes atuais às da frequência anterior.

        Avalia as seis permutações possíveis e seleciona
        a associação com menor distância relativa total.
        """

        previous = self._validate_roots(previous_roots)
        current = self._validate_roots(current_roots)

        candidates = []

        # Três raízes possuem apenas 3! = 6 associações.
        for order in permutations(range(3)):

            total_distance = sum(
                self._relative_distance(
                    previous[i],
                    current[order[i]]
                )
                for i in range(3)
            )

            candidates.append(
                (total_distance, order)
            )

        candidates.sort(key=lambda item: item[0])

        best_score, best_order = candidates[0]
        second_score, _ = candidates[1]

        # Interrompe quando duas associações são muito próximas.
        difference = second_score - best_score

        if difference <= (
            self.ambiguity_tolerance
            * max(1.0, best_score)
        ):
            raise RuntimeError(
                "Rastreamento ambíguo: refine a malha "
                "de frequências ou analise os modos físicos."
            )

        return current[list(best_order)].copy()
