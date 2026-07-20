from methods.meta.hf_transformer_method import HFTransformer

# Qiu et al. 2025, https://huggingface.co/StyleDistance/mstyledistance
MODEL = "StyleDistance/mstyledistance"


class MStyleDistance(HFTransformer):
    def __init__(
        self,
        model_name: str = MODEL,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        name: str = "MStyleDistance",
    ):
        super().__init__(
            name=name,
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
        )
