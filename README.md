# Authorship Verification on German YouTube Transcripts

Ten authorship verification methods, evaluated on German YouTube transcripts with
and without POSNoise masking. This is the code behind the paper under review.

`SETTINGS.md` documents the configuration of every method, the German
adaptations and the deviations from the original implementations.

## Install

    pip install -r requirements.txt
    ./setup_models.sh                  # spaCy models and NLTK data
    WITH_ENGLISH=1 ./setup_models.sh   # additionally for --lang en

DV wraps an implementation that is not shipped here, see NOTICE. It is only
needed for `--methods dv`:

    git clone https://github.com/AlexMoreo/diff-vectors \
        methods/dv/external/diff-vectors

## Run

    python run.py                                                  # all ten methods
    python run.py --methods coav,lambdag --topics math
    python run.py --methods css --masking posnoised --runs 3

The defaults reproduce what the paper reports: German, both masking conditions,
three runs, no fixed seed. Results are appended to `results/train_runs.csv` and
`results/test_runs.csv`, one row per run. Selecting the median run is left to the
caller, see `SETTINGS.md`.

| flag | default |
|---|---|
| `--corpora` | `corpora` |
| `--results` | `results` |
| `--methods` | all of coav, siambert, fevec, dv, css, mlsr, mstyledistance, stylospeaker, rsp, lambdag. Also available: rsp_lora |
| `--masking` | `original,posnoised` |
| `--topics` | all |
| `--lang` | `de`, `en` selects the English variant where a method has one |
| `--runs` | 3, ignored for the deterministic methods |
| `--seed` | unset |

## Corpora

Not part of this repository. `CORPORA.md` describes the layout and how the
corpora behind the paper were composed, `preprocessing/` documents the steps and
ships the scripts for the publishable ones.

## Layout

    run.py              entry point
    common.py           AVMethod, corpus reading, metrics
    methods/            one directory per method, plus meta/ for the shared base
    preprocessing/      raw transcript to corpus
    SETTINGS.md         settings, adaptations, deviations
    CORPORA.md          corpus layout and composition
    NOTICE              third party code and licenses
