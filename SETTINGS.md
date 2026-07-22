# Settings and Adaptations

Records the configuration behind the reported results, one section per method.
Each section names the publication the method comes from and, where it exists,
the implementation it is based on.

---

## Runs and seeds

- The reported results were produced **without a fixed seed**, so that the three
  runs are genuine independent repetitions. Reported is the **median-accuracy
  run**, a real row rather than a column-wise average. Ties are broken by the
  lower AUC. Where a method is stochastic, the section below names the source.
- **Deterministic methods, one run is sufficient:** COAV, StyloSpeaker, MLSR,
  MStyleDistance. The two style models are frozen encoders whose threshold is set
  by EER, which leaves nothing to vary between runs.
- Two threshold conventions are in use. COAV, MLSR and MStyleDistance set the
  decision threshold by **EER** on the training scores. The remaining methods
  calibrate on training accuracy or use the classifier's own 0.5.
- Text length seen per method, relevant for comparability. COAV, LambdaG, DV,
  FeVec and StyloSpeaker see the full text. SiamBERT and RSP see 512-token
  windows, averaged. CSS, MLSR and MStyleDistance see only the first 512 tokens.
  In all three cases the truncation is the design decision of the respective
  authors, not an adaptation.

---

## COAV

Oren Halvani, Christian Winter and Lukas Graner. On the Usefulness of Compression
Models for Authorship Verification. ARES 2017.

- Approach: compression-based verification. A PPM model is built on the known
  document. The cost of compressing the questioned document under that model
  gives a distance, which is thresholded. No training, no random component.
- Hyperparameters: order 5, variant d, exclusion disabled, distance measure cbc,
  symmetric. Threshold by EER on the training distances.
- Choice of model order: 5
- German adaptation: none needed. The method works on the character level and is
  language neutral.

---

## SIAMBERT

Nils Reimers and Iryna Gurevych. Sentence-BERT: Sentence Embeddings using Siamese
BERT-Networks. EMNLP 2019.

The architecture follows Sentence-BERT. The baseline as configured here is
described in an earlier arXiv version of the LambdaG paper, section 6.6, which
the published version no longer contains. 
- Approach: siamese BERT, both texts through the same encoder, compared by the
  cosine similarity of the averaged embeddings, then thresholded.
- Hyperparameters: text in 512-token chunks, mean pooling, averaged into a
  document vector. Loss MSE between cosine and label. Optimizer AdamW, batch size
  4, learning rate 5e-5, 100 epochs with 10 warmup and cosine decay. Epoch chosen
  by AUC on the validation split (20 percent), threshold by accuracy on train.
  The backbone is set up fresh before every training run, so that runs are
  independent and topics do not carry over. Stochastic through initialization and
  batch order.
- Reported variant: **the fully fine-tuned encoder**, that is the whole
  gbert-large is trainable.
- German adaptation: backbone gbert-large (cased) instead of bert-large-cased.
  The remaining processing is language neutral. Cased was chosen because German
  noun capitalization carries information.
- Caveat: full fine-tuning on about 40 training pairs is prone to overfitting and
  memory-intensive.

---

## FEVEC-DE

Janith Weerasinghe, Rhia Singh and Rachel Greenstadt. Feature Vector Difference
based Authorship Verification for Open-World Settings. Working Notes of CLEF
2021.

- Approach: absolute difference of the feature vectors of two documents,
  followed by a linear classifier.
- Hyperparameters: SGD classifier with log loss. Alpha tuned by
  RandomizedSearchCV over loguniform(1e-4, 1.0), 15 iterations. Train/validation
  split 80/20.
  Stochastic through the split, the classifier and the search, all bound to the
  same seed.
- Feature union: character 3-grams (char_wb, tf-idf), POS 3-grams, special
  characters, function word frequencies, character count, distribution and
  average of characters per word, vocabulary richness, masked stop words
  (3-grams), POS tag statistics. With chunking enabled additionally POS chunk
  3-grams and POS chunk subtree 1-grams.
- German adaptation: German spaCy model instead of the English tagger: de_core_news_lg, md, sm. Cross-lingual UPOS tags instead of
  the English Penn Treebank tagset. Function words from
  `spacy.lang.de.stop_words.STOP_WORDS` rather than a manual list. The
  misspelling feature is dropped, since the English spelling and British-spelling
  lists have no German counterpart.
- Note: the German variant deliberately does not load the English NLTK taggers
  (`treebank_brill_aubt.pickle`, PerceptronTagger). They are only needed by the
  English base class.

---

## DV-BIN-DE

Silvia Corbara, Alejandro Moreo and Fabrizio Sebastiani. Same or Different?
Diff-Vectors for Authorship Analysis. ACM Transactions on Knowledge Discovery
from Data 2023. https://github.com/AlexMoreo/diff-vectors

- Approach: a feature vector represents a document pair, its value being the
  absolute difference of the feature frequencies. Character and word n-grams,
  function words, POS. Feature selection by chi-square (top 50000). Logistic
  regression, balanced pair sampling.
- Implementation: the unmodified upstream repository is wrapped, not reimplemented.
- Hyperparameters: binary variant (SAV) only. All same-author pairs and as many
  different-author pairs, capped at 50000. Logistic regression with at most 1000
  iterations. Stochastic through the pair sampling.
- German adaptation: German function words from spaCy stop words. German word and
  sentence tokenization (NLTK, language german). German POS model
  de_core_news_sm. Character and word n-grams, length distributions and feature
  selection stay language neutral.

---

## CSS

Britt van Leeuwen, Sandjai Bhulai and Rob van der Mei. Combining Style and
Semantics for Robust Authorship Verification. Machine Learning with Applications
2025.

- Approach: siamese network on frozen encoder embeddings. The base network is
  three dense layers (256, 128, 64), compared by cosine similarity. Optionally,
  hand-made style features (sentence length, POS counts, stop word ratio, word
  length, Flesch readability) are concatenated to the embedding at input level.
- Hyperparameters: encoder frozen, only the head is trained. BCE loss on the
  cosine similarity, optimizer Adam, batch size 16, learning rate 1e-3, 10
  epochs. Texts are truncated at the encoder limit of 512 tokens. Stochastic
  through initialization and batch order.
- Reported variant: **without style features**.
- German adaptation: the English roberta-base is replaced by the German encoder gbert-base. spaCy and the Flesch readability switched to German, which only
  matters when the style features are active.
- Deviation from the original: the frozen encoder matches the paper, as does the
  architecture (256/128/64, cosine, style at input level) and the 512-token
  truncation. Four differences remain and are not corrected, since the numbers
  were produced with them. No L2 regularization per dense layer. Plain BCE
  instead of weighted BCE with a false-negative weight of 10 against 1. Ten
  epochs with style features instead of 15. A sigmoid instead of the paper's
  lambda layer for mapping the cosine to [0,1]. The original also selects the best epoch on the test set, which is not done here.
- Caveat on the style features: on masked text they are of limited value, because
  the POS-based masking already abstracts the content. spaCy and the Flesch
  readability then operate on non-natural text, and the POS counts duplicate the
  masking.

---

## MLSR

Junghwan Kim, Haotian Zhang and David Jurgens. Leveraging Multilingual Training
for Authorship Representation: Enhancing Generalization across Languages and
Domains. EMNLP 2025.
https://huggingface.co/Blablablab/multilingual-style-representation

- Approach: a pre-trained multilingual style embedding model. Both documents are
  encoded by the same frozen encoder, compared by cosine similarity, then
  thresholded. No fine-tuning and no trainable head.
- Model: `Blablablab/multilingual-style-representation`, XLM-R-large, 1024
  dimensions. Chosen because the model generalizes to unseen languages, German
  among them, which makes it a sensible German AV candidate.
- Hyperparameters: used as a sentence transformer, mean pooling, normalized
  embeddings, batch size 32. The encoder truncates at its `max_seq_length` of
  512, which is architectural (XLM-R has 514 positions).
- Threshold: **EER**. This is our own choice. The paper specifies none and
  supplies embeddings plus cosine only.
- German adaptation: none. The multilinguality sits in the pre-trained encoder.
- Determinism: deterministic. A frozen encoder and an EER threshold leave nothing
  stochastic.

---

## MSTYLEDISTANCE

Justin Qiu, Jiacheng Zhu, Ajay Patel, Marianna Apidianaki and Chris
Callison-Burch. mStyleDistance: Multilingual Style Embeddings and their
Evaluation. Findings of ACL 2025.
https://huggingface.co/StyleDistance/mstyledistance

- Approach: a pre-trained multilingual style embedding model, trained
  contrastively on synthetic near-duplicates that differ in exactly one style
  feature. Both documents are encoded by the same frozen encoder, compared by
  cosine similarity, then thresholded. No fine-tuning and no trainable head.
- Model: `StyleDistance/mstyledistance`, XLM-R-base, 768 dimensions.
- Hyperparameters: used as a sentence transformer, mean pooling, normalized
  embeddings, batch size 32. The encoder truncates at its `max_seq_length` of
  512, which is architectural.
- Threshold: **EER**, our own choice, as for MLSR.
- German adaptation: none. The multilinguality sits in the pre-trained encoder.
- Determinism: deterministic, as for MLSR.

---

## STYLOSPEAKER

Cristina Aggazzotti and Elizabeth Allyn Smith. A Stylometric Analysis of Speaker
Attribution from Speech Transcripts. arXiv 2512.13667.
https://github.com/caggazzotti/styloSpeaker

- Approach: interpretable stylometry for verifying speakers in transcripts. Per
  document, features on five linguistic levels are collected: character, word,
  token, syntax and discourse. The two documents of a pair are joined by the
  absolute difference of their feature vectors. A logistic regression decides.
- Hyperparameters: tf-idf over character n-grams of length 3 to 6, over token
  n-grams of length 1 to 3 and over POS n-grams of length 1 to 3. Each n-gram
  block is capped at 2000 features and requires occurrence in at least ten
  percent of the training documents. The remaining style features are
  z-standardized. The logistic regression runs with at most 1000 iterations.
- Choice of feature combination: the original compares concatenation, difference
  and product of the feature vectors. Reported is the absolute difference, which
  reaches the best AUC in the original. 
- Determinism: the method has no stochastic component. Tf-idf, StandardScaler and
  the lbfgs solver are deterministic, and the pairs come from the corpus
  unchanged rather than being sampled. One run is exact and reproducible. The
  seed only reaches `random_state` of the logistic regression, which lbfgs
  ignores.
- German adaptation: tokenization and POS tagging via the German spaCy model
  de_core_news_sm instead of the English Stanza model. The UPOS tags used are
  language neutral, so the POS features stay unchanged. The 390 English function
  words and 69 function phrases are replaced by 1602 German single words and 391
  German multi-word phrases. Both lists come from the German POSNoise pattern
  list (v3.0) and match exactly the patterns our masking preserves, so the
  function word block is computable under both conditions. The English
  contractions are replaced by 34 German pairs, being fusions of preposition and
  article as well as clitic short forms. Phrases and contractions are counted on
  word boundaries, so that short function words are not counted inside longer
  ones.
- Readability: seven of the original's nine measures are kept unchanged. They are
  computed with the German language setting, so that syllable counting follows
  German hyphenation and the Flesch value corresponds to the German Amstad
  adaptation. Dale-Chall and the count of difficult words are dropped, because
  both require an English list of familiar words. textstat warns "There is no
  easy words vocabulary for de, using english" and falls back to the English
  3000-word list, so Dale-Chall does not react to the language setting at all and
  reduces to sentence length, which is a feature in its own right anyway. The
  syllable-based measures (SMOG, Flesch-Kincaid, Gunning-Fog, Linsear Write) are
  kept: syllables are counted through Pyphen in the configured language, so the
  measurement is correctly German. Their English constants do not matter for this
  model, because an additive offset cancels exactly under the absolute difference
  and the scale factor is absorbed by the StandardScaler and the regression
  coefficient. Gunning-Fog internally consults the same word list, but it filters
  no word at all on our texts.
- Caveat: at about 8000 features and 40 training pairs the logistic regression
  reaches a training accuracy of 1.000 on all corpora, while the test values lie
  clearly below. The method overfits substantially.

---

## RSP

Peter Zeng, Pegah Alipoormolabashi, Jihu Mun, Gourab Dey, Nikita Soni, Niranjan
Balasubramanian, Owen Rambow and H. Andrew Schwartz. Residualized Similarity for
Faithfully Explainable Authorship Verification. Findings of EMNLP 2025.
https://github.com/peterzeng/rsp

- Approach: an interpretable feature system supplies an explainable similarity, a
  neural network predicts the residual to the true label. The final score is
  similarity plus residual, then thresholded.
- Hyperparameters: z-standardization fitted on train. Loss MSE, optimizer AdamW,
  learning rate 5e-5, at most 10 epochs with early stopping (patience 3),
  batch size 8, weight decay 1e-2, validation fraction 0.2. Text in 512-token
  windows, averaged.
- Reported variant: **the encoder is frozen and only the residual head is
  trained**, because there are about 40 training pairs.
- Determinism: with a frozen backbone RSP is practically deterministic. The
  small remaining spread comes from the head initialization and the validation
  split.
- Deviation from the original: the interpretable base Gram2vec is replaced by
  ELFEN. The original adapts the encoder via LoRA on large amounts of data. At
  about 40 training pairs the frozen variant is the one used here.
- German adaptation: neural encoder gbert-large (cased) instead of LUAR.
  Interpretable base ELFEN in German with the German spaCy model
  de_core_news_sm. ELFEN restricted to lexicon-free, cross-lingually robust
  feature groups (surface, POS, morphology, dependency, information, lexical
  richness), since emotion, semantics and psycholinguistics would require
  language-bound lexica.

---

## LAMBDAG

Andrea Nini, Oren Halvani, Lukas Graner, Sophie Titze, Valerio Gherardi and
Shunichi Ishihara. Grammar as a Behavioral Biometric: Using Cognitively Motivated
Grammar Models for Authorship Verification. Humanities and Social Sciences
Communications 2026. LambdaG 0.1.1, MIT,
https://pypi.org/project/LambdaG/#files

- Approach: cognitively motivated grammar model. Kneser-Ney smoothed language
  models are built over POS sequences. The score is a likelihood ratio between
  the model of the known author and a reference model, then thresholded.
- Implementation: the Kneser-Ney language model is imported from the package
  unchanged. The scoring class is a copy of `lambdag.lambdag.LambdaGMethod`,
  extended by an optional process pool and by `fit_predict`.
- Hyperparameters: basis tokens, model order 10 (the implementation default),
  Kneser-Ney smoothing with discount 0.75 and special handling of the
  begin-of-sequence element, lowercasing enabled, sentence splitting enabled, 100
  sampled reference sentences per pair. The reference population is drawn from the
  other authors of the same corpus. The raw score is calibrated into a
  probability by a class-balanced logistic regression, threshold 0.5. Stochastic
  through the reference sampling.
- German adaptation: no language-specific special handling within the procedure
  itself. The language enters through the upstream tokenization and the POS level.
- Note: with a fixed seed, serial and parallel execution produce different scores,
  because the serial path uses one shared generator while the parallel path draws
  an independent stream per pair. The reported runs were unseeded and are
  therefore unaffected.
