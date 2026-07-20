from methods.meta.hf_transformer_method import HFTransformer

# Kim et al. 2025, https://github.com/junghwanjkim/multilingual_aa
# German is an unseen language for this model.
MODEL = "Blablablab/multilingual-style-representation"


class MultilingualStyleRepresentation(HFTransformer):
    def __init__(
        self,
        model_name: str = MODEL,
        normalize_embeddings: bool = True,
        batch_size: int = 32,
        name: str = "MultilingualStyleRepresentation",
    ):
        super().__init__(
            name=name,
            model_name=model_name,
            normalize_embeddings=normalize_embeddings,
            batch_size=batch_size,
        )
