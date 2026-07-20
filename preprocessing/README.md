# Preprocessing

How the corpora were built from the raw transcripts. The corpora themselves are
not part of this repository.


### 1. prepare.py: raw transcript to corpus text

    python prepare.py <raw transcripts> --out prepared

Middle excerpt, anonymization, normalization, filler removal.

**Excerpt.** Transcripts of at most 5000 characters are taken whole. Otherwise
the text is split into sentences, the sentence at the character midpoint becomes
the center, and the excerpt grows from there, alternating one sentence back and
one forward, until it reaches 5000 characters.

**Anonymization.** Names are detected on the *full* transcript and redacted in
the excerpt. Three sources are combined: self-introductions matching fixed
patterns (ich bin, ich heisse, mein name ist, hier ist, hier spricht), person
entities from spaCy `de_core_news_md`, and the channel name. Every name is also
broken into tokens of at least three characters, so that a first name alone is
caught too. Matches are replaced on word boundaries, case-insensitively:
channels become `[KANAL]`, persons `[NAME]`.

**Normalization.** Ellipses and runs of three or more dots become a space, a
missing space after punctuation is inserted, repeated whitespace is collapsed,
and the first letter is capitalized.

**Fillers.** Removed on word boundaries, case-insensitively: aehm, aehs, aeh,
oehm, oeh, ehm, hmm, hm, mhm, mmh, mm, mh.

### 2. trim.py: mechanical cleaning and shared phrase trimming

    python trim.py prepared --out trimmed

**Foreign characters.** Anything outside Latin letters, digits, whitespace and
common punctuation is dropped.

**Whisper loops.** When the recognizer gets stuck it repeats the same word
sequence. Immediate repetitions collapse to one instance: from 3 repetitions for
sequences of several words, from 5 for a single word, so that ordinary doubling
like "ja ja" survives. A sentence identical to the one directly before it is
dropped.

**Shared phrases.** Both texts of an author are compared sentence by sentence
from the start and from the end. For the comparison sentences are lowercased and
reduced to alphanumerics plus single spaces. A boundary sentence counts as a
shared phrase when the character based similarity (sequence matcher ratio) is at
least 0.80. The first pair below that stops the respective end. Differing total
lengths do not matter, since the end is counted backwards. Phrases sitting at
different depths in the two texts are not caught, and neither are heavily
reworded ones below the threshold. The trim is capped so that at least 3
sentences survive in each text.

### 3. Masking

The POSNoise condition is produced with the original implementation, not with
code from this repository: https://github.com/Halvani/POSNoise

    from posnoise import POSNoise, SpacyLanguage, SpacyModelSize
    pn = POSNoise(language=SpacyLanguage.German, spacy_model_size=SpacyModelSize.Large)
    masked = pn.pos_noise(text)

The unmasked texts form the `original` condition, the masked ones `posnoised`.

### 4. build_corpus.py: verification cases and splits

    python build_corpus.py trimmed --out corpora/original

Pairs each speaker's two documents into a same-author case, rings the speakers
and pairs the neighbours into different-author cases, and splits both 40/60 by
speaker. 

## Result

    corpora/<masking>/<topic>/{train,test}/

which is what run.py expects. Run step 4 once on the unmasked texts into
`corpora/original` and once on the masked ones into `corpora/posnoised`, with the
same seed, so that both conditions hold the same cases.
