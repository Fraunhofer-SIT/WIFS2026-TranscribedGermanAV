"""Residualized similarity (Zeng et al., EMNLP Findings 2025).

An interpretable feature system produces a per-document vector; the vectors are
z-scored per feature and the base similarity is s = cos(f(d1), f(d2)). A neural
model predicts the residual res = y - s with y in {-1, +1}, and the final score
is s + res, thresholded.

The paper pairs LUAR with Gram2vec; its table 3 shows ELFEN works as the
interpretable base just as well, which is the configuration used here.

The reported runs freeze the backbone and train the residual head only, because
there are about 40 training pairs. backbone="lora" adapts the backbone with
low-rank adapters, which is closer to the original but needs peft installed.
"""

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from common import AVMethod, calc_metrics, read_corpus

# Lexicon-free areas, robust across languages. Emotion, semantics and the like
# would pull in language-bound resources.
ELFEN_LEXICON_FREE = ["surface", "morphology", "dependency", "information", "lexical_richness", "pos"]


class RSP(AVMethod):
    def __init__(
        self,
        model_name: str = "deepset/gbert-large",
        backbone: str = "freeze",
        max_length: int = 512,
        d_model: int = 128,
        attn_heads: int = 4,
        head_hidden: int = 128,
        dropout: float = 0.3,
        lr: float = 5e-5,
        epochs: int = 10,
        batch_size: int = 8,
        weight_decay: float = 1e-2,
        val_frac: float = 0.2,
        patience: int = 3,
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        interp_lang: str = "de",
        elfen_model: Optional[str] = None,
        elfen_groups=None,
        seed: int | None = None,
        device=None,
        name: str = "RSP",
    ) -> None:
        super().__init__(name=name)
        self.model_name = model_name
        self.backbone = backbone.lower().strip()
        self.max_length = max_length
        self.d_model = d_model
        self.attn_heads = attn_heads
        self.head_hidden = head_hidden
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.val_frac = val_frac
        self.patience = patience
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.interp_lang = interp_lang.lower().strip()
        self.elfen_model = elfen_model
        if elfen_groups is None:
            self.elfen_groups = list(ELFEN_LEXICON_FREE)
        elif isinstance(elfen_groups, str):
            self.elfen_groups = [g.strip() for g in elfen_groups.split(",") if g.strip()]
        else:
            self.elfen_groups = list(elfen_groups)
        self.seed = seed
        self._device = device
        self._elfen_cols = None
        self._scaler = None
        self._tok = None
        self._model = None
        self._head = None
        self._interp_cache: Dict[str, np.ndarray] = {}
        self._emb_cache: Dict[str, np.ndarray] = {}
        self.threshold = 0.5
        self.train_metrics = None
        self.test_metrics = None
        self.last_intconf = None

    @staticmethod
    def _to_pm1(labels) -> np.ndarray:
        return np.where(np.asarray(labels, float) >= 0.5, 1.0, -1.0)

    @staticmethod
    def _residual_targets(labels, s) -> np.ndarray:
        return RSP._to_pm1(labels) - np.asarray(s, float)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        a = np.asarray(a, float).ravel()
        b = np.asarray(b, float).ravel()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) or 1e-9
        return float(np.dot(a, b) / denom)

    @staticmethod
    def _pick_threshold(scores, labels) -> float:
        scores = np.asarray(scores, float)
        labels = np.asarray(labels, int)
        if scores.size == 0:
            return 0.5
        lo, hi = float(scores.min()), float(scores.max())
        cands = np.unique(np.concatenate([[lo - 1e-3, hi + 1e-3], scores]))
        mids = (cands[:-1] + cands[1:]) / 2.0
        thresholds = np.concatenate([[lo - 1e-3], mids, [hi + 1e-3]])
        best_t, best_acc = 0.5, -1.0
        for t in thresholds:
            acc = np.mean((scores > t).astype(int) == labels)
            if acc > best_acc:
                best_acc, best_t = acc, float(t)
        return best_t

    @staticmethod
    def _intconf(s_interp, res_pred, pred_same) -> np.ndarray:
        """How far the decision rests on the interpretable features (paper 3.3)."""
        s = np.asarray(s_interp, float)
        r = np.abs(np.asarray(res_pred, float))
        num = np.where(np.asarray(pred_same, bool), 1.0 + s, 1.0 - s)
        den = num + r
        return np.divide(num, den, out=np.zeros_like(num), where=den != 0)

    def _elfen_default_model(self) -> str:
        return "de_core_news_sm" if self.interp_lang == "de" else "en_core_web_sm"

    def _elfen_matrix(self, docs: List[str]):
        import copy

        import polars as pl
        from elfen.configs.extractor_config import CONFIG_ALL
        from elfen.extractor import Extractor

        df = pl.DataFrame(
            {"text": [t if isinstance(t, str) and t.strip() else " " for t in docs]}
        )
        cfg = copy.deepcopy(CONFIG_ALL)
        cfg["language"] = self.interp_lang
        cfg["model"] = self.elfen_model or self._elfen_default_model()
        if self.elfen_groups and "all" not in self.elfen_groups:
            cfg["features"] = {
                k: v for k, v in cfg["features"].items() if k in self.elfen_groups
            }
        ex = Extractor(data=df, config=cfg)
        feats = ex.extract_features()
        data = feats if hasattr(feats, "columns") else ex.data

        num_cols = []
        for c in (c for c in data.columns if c != "text"):
            try:
                if data.schema[c].is_numeric():
                    num_cols.append(c)
            except (KeyError, AttributeError):
                pass
        X = np.asarray(data.select(num_cols).to_numpy(), float)
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0), num_cols

    @staticmethod
    def _align_cols(X: np.ndarray, cols: List[str], ref: List[str]) -> np.ndarray:
        """Map a feature matrix onto the training column set; ELFEN can vary."""
        if cols == ref:
            return X
        idx = {c: i for i, c in enumerate(cols)}
        out = np.zeros((X.shape[0], len(ref)), float)
        for j, c in enumerate(ref):
            if c in idx:
                out[:, j] = X[:, idx[c]]
        return out

    def _build_interp(self, train_docs: List[str]) -> None:
        from sklearn.preprocessing import StandardScaler

        X, cols = self._elfen_matrix(train_docs)
        self._elfen_cols = cols
        self._scaler = StandardScaler().fit(X)
        for doc, v in zip(train_docs, self._scaler.transform(X)):
            self._interp_cache[doc] = v

    def _prewarm_interp(self, docs: List[str]) -> None:
        todo = [d for d in dict.fromkeys(docs) if d not in self._interp_cache]
        if not todo:
            return
        X, cols = self._elfen_matrix(todo)
        X = self._align_cols(X, cols, self._elfen_cols)
        for d, v in zip(todo, self._scaler.transform(X)):
            self._interp_cache[d] = v

    def _interp_vec(self, text: str) -> np.ndarray:
        if text not in self._interp_cache:
            X, cols = self._elfen_matrix([text])
            X = self._align_cols(X, cols, self._elfen_cols)
            self._interp_cache[text] = self._scaler.transform(X)[0]
        return self._interp_cache[text]

    def _interp_sim(self, d1: str, d2: str) -> float:
        return self._cosine(self._interp_vec(d1), self._interp_vec(d2))

    def _setup_neural(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self._model is None:
            self._model = AutoModel.from_pretrained(
                self.model_name, trust_remote_code=True
            ).to(self._device)
            self._configure_backbone()
        return torch

    def _configure_backbone(self):
        if self.backbone == "freeze":
            for p in self._model.parameters():
                p.requires_grad = False
        elif self.backbone == "lora":
            from peft import LoraConfig, get_peft_model

            self._model = get_peft_model(
                self._model,
                LoraConfig(
                    r=self.lora_r,
                    lora_alpha=self.lora_alpha,
                    lora_dropout=self.lora_dropout,
                    bias="none",
                    target_modules=["query", "key", "value", "dense"],
                ),
            )
        elif self.backbone != "full":
            raise ValueError(f"unknown backbone: {self.backbone}")

    @property
    def _backbone_trainable(self) -> bool:
        return self.backbone in ("lora", "full")

    def _embed_torch(self, text: str):
        import torch

        ids = self._tok(text, add_special_tokens=True, truncation=False)["input_ids"]
        if len(ids) == 0:
            ids = [self._tok.cls_token_id or 0]
        chunks = [ids[i:i + self.max_length] for i in range(0, len(ids), self.max_length)]
        maxlen = max(len(c) for c in chunks)
        pad_id = self._tok.pad_token_id or 0
        input_ids, attn = [], []
        for c in chunks:
            input_ids.append(c + [pad_id] * (maxlen - len(c)))
            attn.append([1] * len(c) + [0] * (maxlen - len(c)))
        input_ids = torch.tensor(input_ids, device=self._device)
        attn = torch.tensor(attn, device=self._device, dtype=torch.float)
        out = self._model(input_ids=input_ids, attention_mask=attn).last_hidden_state
        mask = attn.unsqueeze(-1)
        chunk_vecs = (out * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
        return chunk_vecs.mean(0)

    def _emb_np(self, text: str) -> np.ndarray:
        """Only valid while the backbone is frozen, otherwise the cache would
        hand out embeddings of an encoder that has since been adapted."""
        if self._backbone_trainable:
            raise RuntimeError("the embedding cache is not valid for a trainable backbone")
        if text not in self._emb_cache:
            import torch

            self._model.eval()
            with torch.no_grad():
                self._emb_cache[text] = self._embed_torch(text).detach().cpu().numpy()
        return self._emb_cache[text]

    def _build_head(self, interp_dim: int, neural_dim: int):
        import torch
        import torch.nn as nn

        d, heads, hid, drop = self.d_model, self.attn_heads, self.head_hidden, self.dropout

        class ResidualHead(nn.Module):
            def __init__(self):
                super().__init__()
                self.proj_interp = nn.Linear(interp_dim, d)
                self.proj_neural = nn.Linear(neural_dim, d)
                self.slot_emb = nn.Parameter(torch.zeros(1, 4, d))
                nn.init.normal_(self.slot_emb, std=0.02)
                self.attn = nn.MultiheadAttention(d, heads, dropout=drop, batch_first=True)
                self.mlp = nn.Sequential(
                    nn.Linear(4 * d, hid), nn.ReLU(), nn.Dropout(drop),
                    nn.Linear(hid, hid), nn.ReLU(), nn.Dropout(drop),
                    nn.Linear(hid, 1),
                )

            def forward(self, i1, i2, n1, n2):
                seq = torch.stack(
                    [self.proj_interp(i1), self.proj_interp(i2),
                     self.proj_neural(n1), self.proj_neural(n2)],
                    dim=1,
                ) + self.slot_emb
                a, _ = self.attn(seq, seq, seq)
                return torch.tanh(self.mlp(a.reshape(a.size(0), -1)).squeeze(-1))

        self._head = ResidualHead().to(self._device)
        return self._head

    def _train_residual(self, pairs, labels):
        import torch
        import torch.nn as nn
        from sklearn.model_selection import train_test_split

        train_docs = sorted({d for (k, u) in pairs for d in (k, u)})
        self._build_interp(train_docs)
        s = np.array([self._interp_sim(k, u) for (k, u) in pairs], float)
        res_actual = self._residual_targets(labels, s)

        torch = self._setup_neural()
        self._build_head(self._interp_vec(pairs[0][0]).shape[0], self._model.config.hidden_size)
        if not self._backbone_trainable:
            for d in train_docs:
                self._emb_np(d)

        idx = np.arange(len(pairs))
        do_val = len(pairs) >= 10 and len(set(labels)) > 1
        if do_val:
            tr_i, val_i = train_test_split(
                idx, test_size=self.val_frac, random_state=self.seed,
                stratify=np.asarray(labels),
            )
        else:
            tr_i, val_i = idx, np.array([], dtype=int)

        params = list(self._head.parameters())
        if self._backbone_trainable:
            params += [p for p in self._model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr, weight_decay=self.weight_decay)
        mse = nn.MSELoss()

        def _tens(v):
            return torch.tensor(np.asarray(v, np.float32), device=self._device)

        def _forward_batch(batch_idx, live):
            i1 = torch.stack([_tens(self._interp_vec(pairs[j][0])) for j in batch_idx])
            i2 = torch.stack([_tens(self._interp_vec(pairs[j][1])) for j in batch_idx])
            # A trainable backbone has to be run, a frozen one is cached.
            emb = self._embed_torch if live else (lambda t: _tens(self._emb_np(t)))
            n1 = torch.stack([emb(pairs[j][0]) for j in batch_idx])
            n2 = torch.stack([emb(pairs[j][1]) for j in batch_idx])
            return self._head(i1, i2, n1, n2)

        def _snapshot():
            st = {"head": {k: v.detach().cpu().clone()
                           for k, v in self._head.state_dict().items()}}
            if self._backbone_trainable:
                st["backbone"] = {n: p.detach().cpu().clone()
                                  for n, p in self._model.named_parameters() if p.requires_grad}
            return st

        def _restore(st):
            self._head.load_state_dict(st["head"])
            if "backbone" in st:
                with torch.no_grad():
                    for n, p in self._model.named_parameters():
                        if n in st["backbone"]:
                            p.copy_(st["backbone"][n].to(p.device))

        best_val, best_state, bad = math.inf, None, 0
        rng = np.random.RandomState(self.seed)
        for _ in range(self.epochs):
            self._head.train()
            if self._backbone_trainable:
                self._model.train()
            order = list(tr_i)
            rng.shuffle(order)
            for b in range(0, len(order), self.batch_size):
                bi = order[b:b + self.batch_size]
                opt.zero_grad()
                loss = mse(_forward_batch(bi, self._backbone_trainable), _tens(res_actual[bi]))
                loss.backward()
                opt.step()

            if not do_val:
                continue
            self._head.eval()
            if self._backbone_trainable:
                self._model.eval()
            with torch.no_grad():
                vp = _forward_batch(list(val_i), self._backbone_trainable).detach().cpu().numpy()
            vloss = float(np.mean((vp - res_actual[val_i]) ** 2))
            if vloss < best_val - 1e-5:
                best_val, bad = vloss, 0
                best_state = _snapshot()
            else:
                bad += 1
                if bad >= self.patience:
                    break
        if best_state is not None:
            _restore(best_state)

    def _res_pred(self, pairs) -> np.ndarray:
        import torch

        self._head.eval()
        if self._backbone_trainable:
            self._model.eval()
        out = []
        with torch.no_grad():
            for (k, u) in pairs:
                interp = [torch.tensor(v.astype(np.float32), device=self._device)[None]
                          for v in (self._interp_vec(k), self._interp_vec(u))]
                if self._backbone_trainable:
                    neural = [self._embed_torch(t)[None] for t in (k, u)]
                else:
                    neural = [torch.tensor(self._emb_np(t).astype(np.float32),
                                           device=self._device)[None] for t in (k, u)]
                out.append(float(self._head(*interp, *neural).item()))
        return np.asarray(out, float)

    def _scores(self, pairs) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        s = np.array([self._interp_sim(k, u) for (k, u) in pairs], float)
        res = self._res_pred(pairs)
        return s + res, s, res

    @staticmethod
    def _pairs_and_labels(corpus_path):
        problems, labels, _ = read_corpus(corpus_path)
        return [(p[0], p[2]) for p in problems], [int(round(float(v))) for v in labels]

    def train(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        pairs, labels = self._pairs_and_labels(corpus_path)
        # A trainable backbone would otherwise keep training across corpora and
        # runs. A frozen one is never updated, so reloading it would only cost
        # time.
        if self.backbone != "freeze" and self._model is not None:
            import torch

            del self._model
            self._model = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        self._interp_cache, self._emb_cache = {}, {}

        self._train_residual(pairs, labels)
        final, _, _ = self._scores(pairs)
        self.threshold = self._pick_threshold(final, labels)
        preds = (final > self.threshold).astype(int)
        self.train_metrics = calc_metrics(corpus_name, self.name, final, preds, labels)
        return self.train_metrics

    def eval(self, corpus_name: str, corpus_path: str, **kwargs) -> Dict[str, Any]:
        pairs, labels = self._pairs_and_labels(corpus_path)
        # Scaler, columns and weights stay the ones fitted on train.
        self._interp_cache, self._emb_cache = {}, {}
        self._prewarm_interp([d for (k, u) in pairs for d in (k, u)])

        final, s, res = self._scores(pairs)
        preds = (final > self.threshold).astype(int)
        ic = self._intconf(s, res, preds.astype(bool))
        self.last_intconf = {
            "mean": float(np.mean(ic)) if ic.size else 0.0,
            "median": float(np.median(ic)) if ic.size else 0.0,
            "frac_ge_0.5": float(np.mean(ic >= 0.5)) if ic.size else 0.0,
        }
        self.test_metrics = calc_metrics(corpus_name, self.name, final, preds, labels)
        return self.test_metrics