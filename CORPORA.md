# Corpora

The corpora are not part of this repository. This describes what `run.py`
expects and how the corpora behind the paper were built.

## Layout

    corpora/<masking>/<topic>/<split>/<case>/
        known <author> - <video id>.txt
        unknown <author> - <video id>.txt

- `<masking>` is `original` or `posnoised`
- `<topic>` is `diy`, `finance` or `math`
- `<split>` is `train` or `test`
- `<case>` is `[Y] <author> vs. <author>` or `[N] <author> vs. <other author>`

The label sits in the directory name and is read from there. A `truth.txt` with
one `<case> <Y|N>` line per case is written into each split on the first run if
it is missing.

## Composition

Three domains, 50 speakers each, two videos per speaker, 300 videos in total.
Each speaker contributes exactly two documents.

**Verification cases.** Pairing a speaker's two documents gives a same-author
case, which yields 50 Y cases per domain. For the different-author cases the
speakers are randomly permuted and arranged in a ring. Each adjacent pair
contributes one case, built from the first speaker's second document and the
second speaker's first document, which yields 50 N cases. A domain therefore
holds 100 cases, balanced between Y and N.

**Splits.** The speakers are split 40/60 between train and test, so 20 speakers
give 40 training cases and 30 speakers give 60 test cases, each balanced. The
test set is the larger one, which leaves little training data and makes the task
harder.


## Building them

    corpora/<masking>/<topic>/<split>/

is produced by `preprocessing/build_corpus.py` from the trimmed texts, see
`preprocessing/README.md` for the steps before that. Text files are named
`<author> - <video id>.txt` with an eleven character video id, and the author is
read from that name. Speakers with fewer than two texts are dropped.

The `posnoised` tree mirrors `original` with every text masked by POSNoise,
https://github.com/Halvani/POSNoise, German, spaCy `de_core_news_lg`. The
`truth.txt` files are copied rather than masked.
