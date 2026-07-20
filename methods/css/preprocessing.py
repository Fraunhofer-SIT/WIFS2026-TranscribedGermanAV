from functools import lru_cache

import numpy as np
import spacy
import textstat
import torch
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer


@lru_cache(maxsize=None)
def _encoder(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


@lru_cache(maxsize=None)
def _spacy(model_name):
    return spacy.load(model_name)


def get_embedding(text, model_name, device="cpu"):
    """Mean of the token embeddings. Texts beyond the encoder limit are cut off,
    which is the behaviour van Leeuwen et al. describe."""
    tokenizer, model = _encoder(model_name)
    model.to(device)
    with torch.no_grad():
        inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze(0)


def extract_style_features(text, spacy_model, lang="de"):
    textstat.set_lang(lang)
    doc = _spacy(spacy_model)(text)

    sentences = list(doc.sents)
    words = [token.text for token in doc if token.is_alpha]
    return np.array(
        [
            textstat.flesch_reading_ease(text),
            sum(len(sent) for sent in sentences) / len(sentences),
            sum(1 for token in doc if token.pos_ == "NOUN"),
            sum(1 for token in doc if token.pos_ == "VERB"),
            sum(1 for token in doc if token.is_stop) / len(doc),
            sum(len(word) for word in words) / len(words),
        ]
    )


def build_scaler(texts, spacy_model, lang="de"):
    features = np.vstack([extract_style_features(t, spacy_model, lang) for t in texts])
    scaler = StandardScaler()
    scaler.fit(features)
    return scaler


def get_model_input(text, model_name, scaler=None, spacy_model=None, lang="de", device="cpu"):
    emb = get_embedding(text, model_name, device)
    if scaler is None:
        return emb

    features = extract_style_features(text, spacy_model, lang).reshape(1, -1)
    normalized = torch.tensor(
        scaler.transform(features), dtype=torch.float32, device=device
    ).flatten()
    return torch.cat((emb, normalized))
