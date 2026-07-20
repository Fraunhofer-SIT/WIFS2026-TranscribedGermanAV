from functools import lru_cache


@lru_cache(maxsize=None)
def get_nlp(model="de_core_news_md"):
    import spacy

    return spacy.load(model)
