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


## Robustness analyses

`analysis/` holds the scripts behind the confidence intervals, repeated
splits, cross-domain runs and significance tests of the paper. The approximate
randomization test is the implementation of Van Asch used by the PAN 2014
authorship verification overview. It is not part of this repository;
`analysis/port_art.py` fetches the three files from
https://github.com/mikekestemont/ruzicka, verifies their checksums and applies
the Python 3 port in `analysis/art3.patch`.

    python analysis/port_art.py

`patched_run.py` is a drop-in replacement for `run.py` that also writes
`predictions.csv` with one row per case (topic, masking, model, run, split,
pair, y_true, y_pred, score). The corpus builders write symbolic links in the
`run.py` layout. The analyses of the paper were produced as follows, with
`<corpora>` the structured corpus tree of `CORPORA.md`.

    A=analysis
    python $A/make_cv_folds.py --src <corpora> --dst corpora_cv --seeds 42,43,44,45,46 --train-frac 0.4
    python $A/make_cross_domain.py --src <corpora> --dst corpora_cross
    python $A/patched_run.py --corpora corpora       --results results_main  --methods all --runs 1
    python $A/patched_run.py --corpora corpora_cv    --results results_cv    --methods all --runs 1
    python $A/patched_run.py --corpora corpora_cross --results results_cross --methods all --runs 1

    python $A/bootstrap_ci.py --predictions results_main/predictions.csv --out bootstrap_ci.csv --B 1000 --seed 1
    for base in dyi finanzen lehr-mathe; do for mask in original posnoised; do
      for m in COAV CSS DV-Bin-DE FeVec-DE LambdaG MStyleDistance MultilingualStyleRepresentation RSP SiamBERT StyloSpeaker; do
        python $A/run_art_table.py --predictions results_cv/predictions.csv --mode folds \
            --base $base --model "$m" --masking $mask --r 1000 --seed 7 --out art_splits
      done
    done; done
    for t in dyi finanzen lehr-mathe; do for mask in original posnoised; do
      python $A/run_art_table.py --predictions results_main/predictions.csv --mode methods \
          --corpus $t --masking $mask --r 1000 --seed 7 --exact-threshold 10 --out art_methods
    done; done
    python $A/summarize_splits.py --test-runs results_cv/test_runs.csv --art-dir art_splits --out splits_summary
    python $A/make_cross_table.py --cross results_cross/test_runs.csv --main results_main/test_runs.csv --out cross_table

Splits follow the construction of the published corpora: the speakers of a
domain are shuffled with a fixed seed, one ring over the shuffled order defines
the N-cases, and the train/test split cuts that order at 40 % of the speakers,
so one N-case per side contains a document of a speaker from the other side.
`--seeds 42 --train-frac 0.4` reproduces the published split. `--seed` fixes
the shuffles of the test, `--r` their number, and `--exact-threshold` the
number of differing predictions up to which the paired test enumerates all
sign assignments (20 in the original, 10 keeps the all-method comparisons
fast). Significance marks follow PAN 2014: `***` p < 0.001, `**` p < 0.01,
`*` p < 0.05, `=` otherwise.

Add to the "Layout" section:

    analysis/           robustness analyses and significance tests

Add to .gitignore:

    analysis/art3/

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
