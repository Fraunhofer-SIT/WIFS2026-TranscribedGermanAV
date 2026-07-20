import numpy as np

from methods.coav.ppm import PPM


class CoavInterpretable:
    def __init__(
        self,
        order=4,
        variant="d",
        use_exclusion=False,
        dist_measure="cbc",
        symmetric=True,
    ):
        self.order = order
        self.variant = variant
        self.use_exclusion = use_exclusion
        self.symmetric = symmetric
        self.dist_measure = dist_measure
        self._log_probs_cache = {}

    def log_probs(self, text, conditional_text=None):
        cache_key = (text, conditional_text)
        if cache_key in self._log_probs_cache:
            return self._log_probs_cache[cache_key]

        ppm = PPM(order=self.order, variant=self.variant, use_exclusion=self.use_exclusion)
        if conditional_text is not None:
            ppm.encode_text(conditional_text)
        probs = np.array(ppm.encode_text(text))
        result = -np.log(probs)

        self._log_probs_cache[cache_key] = result
        return result

    def symbol_dists(self, x, y):
        y_log_probs = self.log_probs(y)
        x_log_probs = self.log_probs(x)
        y_after_x_log_probs = self.log_probs(y, x)

        if self.dist_measure == "cbc":
            dist_normalize = np.sqrt(np.sum(x_log_probs) * np.sum(y_log_probs))
        elif self.dist_measure == "ncd":
            dist_normalize = max(np.sum(x_log_probs), np.sum(y_log_probs))
        elif self.dist_measure == "cdm":
            dist_normalize = np.sum(x_log_probs) + np.sum(y_log_probs)

        n = len(y_log_probs)
        symbol_dists = (1 / n) - (y_log_probs - y_after_x_log_probs) / dist_normalize
        dist = 1 - (np.sum(y_log_probs) - np.sum(y_after_x_log_probs)) / dist_normalize
        assert abs(symbol_dists.sum() - dist) < 1e-10
        return symbol_dists

    def dist(self, x, y):
        if not self.symmetric:
            return self.symbol_dists(x, y).sum()
        return (self.symbol_dists(x, y).sum() + self.symbol_dists(y, x).sum()) / 2
