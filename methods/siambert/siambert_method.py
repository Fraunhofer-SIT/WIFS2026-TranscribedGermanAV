"""Siamese BERT baseline (Nini et al. 2025, LambdaG paper, section 6.6), without
the forensic LR/PLDA/Cllr machinery.

A document is cut into chunks of max_length tokens, every chunk is mean-pooled
over the attention mask, and the chunk vectors are averaged into one document
vector. A pair is scored by the cosine of its two vectors and thresholded.
"""

from typing import Any, Dict, List, Tuple

import numpy as np

from common import AVMethod, calc_metrics, read_corpus


class SiamBERT(AVMethod):
    def __init__(
        self,
        model_name: str = "deepset/gbert-large",
        max_length: int = 512,
        batch_size: int = 4,
        lr: float = 5e-5,
        epochs: int = 100,
        warmup_epochs: int = 10,
        val_frac: float = 0.2,
        freeze_backbone: bool = False,
        unfrozen_layers: int = 2,
        seed: int | None = None,
        device=None,
        name: str = "SiamBERT",
    ) -> None:
        super().__init__(name=name)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.warmup_epochs = warmup_epochs
        self.val_frac = val_frac
        # The reported runs fine-tune the whole backbone. Setting this keeps
        # all but the top unfrozen_layers fixed, which was tried because
        # gbert-large has about 40 training pairs to work with.
        self.freeze_backbone = freeze_backbone
        self.unfrozen_layers = unfrozen_layers
        self.seed = seed
        self._device = device
        self._tok = None
        self._model = None
        self.threshold = 0.5
        self.train_metrics = None
        self.test_metrics = None

    @staticmethod
    def _pick_threshold(cosines: np.ndarray, labels: np.ndarray) -> float:
        cosines = np.asarray(cosines, float)
        labels = np.asarray(labels, int)
        cands = np.unique(np.concatenate([[-1.0, 1.0], cosines]))
        mids = (cands[:-1] + cands[1:]) / 2.0
        thresholds = np.concatenate([[-1.0], mids, [1.0]])
        best_t, best_acc = 0.0, -1.0
        for t in thresholds:
            acc = np.mean((cosines > t).astype(int) == labels)
            if acc > best_acc:
                best_acc, best_t = acc, float(t)
        return best_t

    def _reset_model(self):
        """Drop the backbone so that train() rebuilds it from_pretrained.

        Without this the same weights would keep training across corpora and
        runs, which would make runs dependent and topics leak into each other.
        """
        import torch

        if self._model is not None:
            del self._model
        self._model = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _setup(self):
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._device is None:
            self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.seed is not None:
            torch.manual_seed(self.seed)
            np.random.seed(self.seed)
        if self._tok is None:
            self._tok = AutoTokenizer.from_pretrained(self.model_name)
        if self._model is None:
            self._model = AutoModel.from_pretrained(self.model_name).to(self._device)
            if self.freeze_backbone:
                self._apply_freeze()
        return torch

    def _apply_freeze(self):
        m = self._model
        for p in m.parameters():
            p.requires_grad = False
        try:
            for layer in m.encoder.layer[-self.unfrozen_layers:]:
                for p in layer.parameters():
                    p.requires_grad = True
        except AttributeError:
            for p in list(m.parameters())[-2 * self.unfrozen_layers:]:
                p.requires_grad = True

    def _embed_text_torch(self, text: str):
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

    def _cosine_pair_torch(self, known: str, unknown: str):
        import torch

        f_a = self._embed_text_torch(known)
        f_u = self._embed_text_torch(unknown)
        return torch.nn.functional.cosine_similarity(
            f_u.unsqueeze(0), f_a.unsqueeze(0)
        ).squeeze(0)

    def _cosines_eval(self, pairs) -> np.ndarray:
        import torch

        self._model.eval()
        out = []
        with torch.no_grad():
            for known, unknown, _ in pairs:
                out.append(float(self._cosine_pair_torch(known, unknown).item()))
        return np.asarray(out, float)

    def _finetune(self, pairs, labels):
        import torch
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
        from transformers import get_cosine_schedule_with_warmup

        idx = np.arange(len(pairs))
        strat = labels if len(set(labels)) > 1 else None
        tr_i, val_i = train_test_split(
            idx, test_size=self.val_frac, random_state=self.seed, stratify=strat
        )
        params = [p for p in self._model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=self.lr)
        steps_per_epoch = max(1, int(np.ceil(len(tr_i) / self.batch_size)))
        sched = get_cosine_schedule_with_warmup(
            opt, steps_per_epoch * self.warmup_epochs, steps_per_epoch * self.epochs
        )
        # The paper regresses the cosine, which lives in [-1, 1], onto a binary
        # label. The scale mismatch is kept on purpose.
        mse = torch.nn.MSELoss()

        best_auc, best_state = -1.0, None
        rng = np.random.RandomState(self.seed)
        for _ in range(self.epochs):
            self._model.train()
            order = list(tr_i)
            rng.shuffle(order)
            for b in range(0, len(order), self.batch_size):
                batch = order[b:b + self.batch_size]
                opt.zero_grad()
                loss = 0.0
                for j in batch:
                    known, unknown, _ = pairs[j]
                    cos = self._cosine_pair_torch(known, unknown)
                    target = torch.tensor(float(labels[j]), device=self._device)
                    loss = loss + mse(cos, target)
                (loss / len(batch)).backward()
                opt.step()
                sched.step()

            val_cos = self._cosines_eval([pairs[i] for i in val_i])
            val_lab = [labels[i] for i in val_i]
            try:
                auc = roc_auc_score(val_lab, val_cos) if len(set(val_lab)) > 1 else 0.5
            except ValueError:
                auc = 0.5
            if auc > best_auc:
                best_auc = auc
                best_state = {
                    k: v.detach().cpu().clone() for k, v in self._model.state_dict().items()
                }
        if best_state is not None:
            self._model.load_state_dict(best_state)

    @staticmethod
    def _pairs_and_labels(corpus_path) -> Tuple[List[Tuple[str, str, None]], List[int]]:
        problems, labels, _ = read_corpus(corpus_path)
        labels = [int(round(float(v))) for v in labels]
        pairs = [(known, unknown, None) for known, _, unknown, _ in problems]
        return pairs, labels

    def train(self, corpus_name: str, corpus_path: str, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        pairs, labels = self._pairs_and_labels(corpus_path)
        self._reset_model()
        self._setup()
        self._finetune(pairs, labels)
        cos = self._cosines_eval(pairs)
        self.threshold = self._pick_threshold(cos, labels)
        preds = (cos > self.threshold).astype(int)
        self.train_metrics = calc_metrics(corpus_name, self.name, cos, preds, labels)
        return self.train_metrics

    def eval(self, corpus_name: str, corpus_path: str, **kwargs: Dict[str, Any]) -> Dict[str, Any]:
        pairs, labels = self._pairs_and_labels(corpus_path)
        cos = self._cosines_eval(pairs)
        preds = (cos > self.threshold).astype(int)
        self.test_metrics = calc_metrics(corpus_name, self.name, cos, preds, labels)
        return self.test_metrics
