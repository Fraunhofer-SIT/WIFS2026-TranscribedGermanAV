from methods.coav.coav_av import CoavInterpretable
from methods.meta.score_method import ScoreMethod


class Coav(ScoreMethod):
    """Compression based verification: a PPM model of one document gives the
    compression cost of the other, and the pair is scored by that distance.

    Order 5 is the reported setting; the implementation itself defaults to 4.
    """

    def __init__(
        self,
        name: str = "COAV",
        order: int = 5,
        variant: str = "d",
        use_exclusion: bool = False,
        dist_measure: str = "cbc",
        symmetric: bool = True,
    ) -> None:
        super().__init__(name=name)
        self.coav = CoavInterpretable(
            order=order,
            variant=variant,
            use_exclusion=use_exclusion,
            dist_measure=dist_measure,
            symmetric=symmetric,
        )

    def get_similarity_score(self, doc1: str, doc2: str) -> float:
        return -self.coav.dist(doc1, doc2)
