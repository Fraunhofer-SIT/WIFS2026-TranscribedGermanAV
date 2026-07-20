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

**Splits.** The speakers of a domain are split 40/60 between train and test
before any case is formed, so that no speaker appears in both. The verification
cases are built within each split afterwards, which keeps the two corpora author
disjoint. The 20 train speakers give 40 training cases and the 30 test speakers
give 60 test cases. The larger test set leaves little training data and makes the
task harder.

**Verification cases.** Within a split, pairing a speaker's two documents gives a
same-author case, one Y case per speaker. For the different-author cases the
speakers of the split are randomly permuted and arranged in a ring; each adjacent
pair contributes one case, built from the first speaker's first document and the
second speaker's second document. Each split is therefore balanced between Y and
N, 20 Y and 20 N in train and 30 Y and 30 N in test.


## Building them

    corpora/<masking>/<topic>/<split>/

is produced by `preprocessing/build_corpus.py` from the trimmed texts, see
`preprocessing/README.md` for the steps before that. Text files are named
`<author> - <video id>.txt` with an eleven character video id, and the author is
read from that name. Speakers with fewer than two texts are dropped.

The `posnoised` tree mirrors `original` with every text masked by POSNoise,
https://github.com/Halvani/POSNoise, German, spaCy `de_core_news_lg`. The
`truth.txt` files are copied rather than masked.
