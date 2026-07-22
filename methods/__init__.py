"""Method registry.

Imports are lazy so that a missing optional dependency only affects the method
that needs it. The settings below are the ones reported in the paper. German is
the default; lang="en" selects the English variant where a method has one.
"""

# Chronological order, as in the paper.
MAIN = [
    "coav",
    "siambert",
    "fevec",
    "dv",
    "css",
    "mlsr",
    "mstyledistance",
    "stylospeaker",
    "rsp",
    "lambdag",
]

# Selectable by name, but not part of "all": the paper has no LoRA row. The run
# that was meant to produce one silently fell back to a frozen backbone, because
# peft was missing and the old code swallowed the ImportError.
EXTRA = ["rsp_lora"]

# No stochastic component, so a single run is exact and reproducible. The two
# style models are frozen encoders scored against an EER threshold, which leaves
# nothing to vary between runs.
DETERMINISTIC = {"coav", "stylospeaker", "mlsr", "mstyledistance"}


def build(name, seed=None, lang="de"):
    if name == "coav":
        from methods.coav.coav_method import Coav

        return Coav(order=5)

    if name == "siambert":
        from methods.siambert.siambert_method import SiamBERT

        backbone = "deepset/gbert-large" if lang == "de" else "bert-large-cased"
        return SiamBERT(model_name=backbone, freeze_backbone=False, seed=seed)

    if name == "fevec":
        if lang == "de":
            from methods.fevec.fevec_de import FeVecDE

            return FeVecDE(seed=seed)
        from methods.fevec.fevec_method import FeVec

        return FeVec(seed=seed)

    if name == "dv":
        if lang == "de":
            from methods.dv.dv_de import DVDE

            return DVDE(seed=seed)
        from methods.dv.dv_method import DV

        return DV(seed=seed)

    if name == "css":
        from methods.css.css_method import CSS

        if lang == "de":
            encoder, spacy_model = "deepset/gbert-base", "de_core_news_sm"
        else:
            # van Leeuwen et al. run the original on roberta-base.
            encoder, spacy_model = "roberta-base", "en_core_web_sm"
        return CSS(model_name=encoder, spacy_model=spacy_model, lang=lang, seed=seed)

    if name == "mlsr":
        from methods.mlsr.mlsr_method import MultilingualStyleRepresentation

        return MultilingualStyleRepresentation()

    if name == "mstyledistance":
        from methods.mstyledistance.mstyledistance_method import MStyleDistance

        return MStyleDistance()

    if name == "stylospeaker":
        from methods.stylospeaker.stylospeaker_method import StyloSpeaker

        # The original tags with Stanza; German goes through spaCy.
        backend = "stanza" if lang == "en" else "spacy"
        return StyloSpeaker(lang=lang, nlp_backend=backend)

    if name in ("rsp", "rsp_lora"):
        from methods.rsp.rsp_method import RSP

        if lang == "de":
            encoder, spacy_model, groups = "deepset/gbert-large", "de_core_news_sm", None
        else:
            # Paper table 3: LUAR plus ELFEN. The English lexicons ship with
            # ELFEN, so no feature area has to be dropped.
            encoder, spacy_model, groups = "rrivera1849/LUAR-MUD", "en_core_web_lg", "all"
        return RSP(
            model_name=encoder,
            backbone="freeze" if name == "rsp" else "lora",
            interp_lang=lang,
            elfen_model=spacy_model,
            elfen_groups=groups,
            seed=seed,
        )

    if name == "lambdag":
        from methods.lambdag.lambdag_method import LambdaG

        return LambdaG(lowercasing=True, random_seed=seed)

    raise ValueError(f"unknown method: {name}")
