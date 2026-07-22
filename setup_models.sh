#!/usr/bin/env bash
# Models and corpora that pip does not install. Run once after pip install.
set -euo pipefail

# de_core_news_lg for FeVec, sm for StyloSpeaker, RSP, CSS and DV. FeVec tags
# and parses with the large model, a smaller one changes its features.
python -m spacy download de_core_news_lg   # 3.7.0 for the reported runs
python -m spacy download de_core_news_sm   # 3.7.0

# Only needed for --lang en: en_core_web_lg for RSP, en_core_web_sm for CSS and DV.
if [ "${WITH_ENGLISH:-0}" = "1" ]; then
    python -m spacy download en_core_web_sm  # 3.7.1
    python -m spacy download en_core_web_lg  # 3.7.1
fi

python - <<'EOF'
import nltk

# punkt splits sentences and words for DV, stopwords back the English function
# word list, the rest is only needed for FeVec with --lang en.
for package in ("punkt", "punkt_tab", "stopwords"):
    nltk.download(package, quiet=True)

import os

if os.environ.get("WITH_ENGLISH") == "1":
    for package in ("conll2000", "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"):
        nltk.download(package, quiet=True)
EOF