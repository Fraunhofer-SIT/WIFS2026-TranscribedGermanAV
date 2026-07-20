from collections import defaultdict


class PPM:
    def __init__(self, order, variant="d", use_exclusion=False, vocabulary_size=256):
        self.table = defaultdict(lambda: defaultdict(int))
        self.order = order
        self.variant = variant
        self.use_exclusion = use_exclusion
        self.vocabulary_size = vocabulary_size

        def ns(context, symbol):
            return self.table[context][symbol]

        if use_exclusion:

            def n(context, excluded_symbols, **kwargs):
                return sum(
                    count
                    for symbol, count in self.table[context].items()
                    if symbol not in excluded_symbols
                )

            def u(context, excluded_symbols, **kwargs):
                return sum(
                    (1 if count > 0 else 0)
                    for symbol, count in self.table[context].items()
                    if symbol not in excluded_symbols
                )

            self.context_seen = lambda context, excluded_symbols, **kwargs: any(
                count > 0
                for symbol, count in self.table[context].items()
                if symbol not in excluded_symbols
            )
        else:

            def n(context, **kwargs):
                return sum(self.table[context].values())

            def u(context, **kwargs):
                return sum(
                    (1 if count > 0 else 0)
                    for symbol, count in self.table[context].items()
                )

            self.context_seen = lambda context, **kwargs: any(
                count > 0 for symbol, count in self.table[context].items()
            )

        if variant == "a":
            self.symbol_p_condition = lambda context, symbol: (
                ns(context, symbol) >= 1
            )  # condition: ns >= 1
            self.symbol_p = lambda context, symbol, **kwargs: (
                ns(context, symbol) / (1 + n(context, **kwargs))
            )  # symbol_p: ns/(1+n)
            self.escape_p = lambda context, symbol, **kwargs: (
                1 / (1 + n(context, **kwargs))
            )  # escape_p: 1/(1+n)
        elif variant == "b":
            self.symbol_p_condition = lambda context, symbol: (
                ns(context, symbol) >= 2
            )  # condition: ns >= 2
            self.symbol_p = lambda context, symbol, **kwargs: (
                (ns(context, symbol) - 1) / n(context, **kwargs)
            )  # symbol_p: (ns-1)/n
            self.escape_p = lambda context, symbol, **kwargs: (
                u(context, **kwargs) / n(context, **kwargs)
            )  # escape_p:     u/n
        elif variant == "c":
            self.symbol_p_condition = lambda context, symbol: (
                ns(context, symbol) >= 2
            )  # condition: ns >= 2
            self.symbol_p = lambda context, symbol, **kwargs: (
                ns(context, symbol) / (n(context, **kwargs) + u(context, **kwargs))
            )  # symbol_p: ns/(n+u)
            self.escape_p = lambda context, symbol, **kwargs: (
                u(context, **kwargs) / (n(context, **kwargs) + u(context, **kwargs))
            )  # escape_p: u/(n+u)
        elif variant == "d":
            self.symbol_p_condition = lambda context, symbol: (
                ns(context, symbol) >= 1
            )  # condition: ns >= 1
            self.symbol_p = lambda context, symbol, **kwargs: (
                (2 * ns(context, symbol) - 1) / (2 * n(context, **kwargs))
            )  # symbol_p: (2*ns-1)/(2*n)
            self.escape_p = lambda context, symbol, **kwargs: (
                u(context, **kwargs) / (2 * n(context, **kwargs))
            )  # escape_p:       u/(2*n)
        elif variant == "e":
            self.symbol_p_condition = lambda context, symbol: (
                ns(context, symbol) >= 1
            )  # condition: ns >= 1
            self.symbol_p = lambda context, symbol, **kwargs: (
                (4 * ns(context, symbol) - 2) / (4 * n(context, **kwargs) - 1)
            )  # symbol_p: (4*ns-2)/(4*n-1)
            self.escape_p = lambda context, symbol, **kwargs: (
                (2 * u(context, **kwargs) - 1) / (4 * n(context, **kwargs) - 1)
            )  # escape_p: (2*u-1)/(4*n-1)

    def calc_probability(self, context, symbol):
        p = 1.0
        kwargs = {"excluded_symbols": set()} if self.use_exclusion else {}
        # kwargs is used for "addons", such as exclusion
        while True:
            if self.context_seen(context, **kwargs):
                if self.symbol_p_condition(context=context, symbol=symbol):
                    # found qualifying symbol for context => p(symbol | context)
                    return p * self.symbol_p(context=context, symbol=symbol, **kwargs)
                else:
                    # found context but symbol does not qualify => escape
                    p *= self.escape_p(context=context, symbol=symbol, **kwargs)
            if len(context) > 0:
                if self.use_exclusion:
                    kwargs["excluded_symbols"] = {
                        symbol
                        for symbol in self.table[context].keys()
                        if self.symbol_p_condition(context=context, symbol=symbol)
                    }
                context = context[1:]
            else:
                # context is already empty => fallback
                return p / self.vocabulary_size

    def encode_symbol(self, context, symbol, update_tables=True):
        p = self.calc_probability(context, symbol)
        if update_tables:
            for i in range(len(context) + 1):
                shortened_context = context[i:]
                self.table[shortened_context][symbol] += 1
        return p

    def encode_text(self, text, update_tables=True):
        probs = []
        for i in range(0, len(text)):
            context = text[max(i - self.order, 0) : i]
            symbol = text[i]
            prob = self.encode_symbol(context, symbol, update_tables=update_tables)
            probs.append(prob)
        return probs

    ### old recursive version of calc_probability
    # def calc_probability(self, context, symbol, total_symbol_count=256):
    #     if context is None:
    #         return 1.0 / total_symbol_count
    #     elif context not in self.table:
    #         # escape (context not seen yet)
    #         shorter_context = context[1:] if len(context) > 0 else None
    #         return self.calc_probability(shorter_context, symbol)
    #     elif self.symbol_p_condition(context, symbol):
    #         # p(symbol | context)
    #         return self.symbol_p(context, symbol)
    #     else:
    #         # escape
    #         shorter_context = context[1:] if len(context) > 0 else None
    #         return self.escape_p(context, symbol) * self.calc_probability(shorter_context, symbol)

    # def calc_probability_chain(self, context, symbol):
    #     probabilities = []
    #     while True:
    #         if context is None:
    #             probabilities.append((context, 1.0 / self.vocabulary_size))
    #             return probabilities
    #         elif context not in self.table:
    #             context = context[1:] if len(context) > 0 else None
    #         elif self.symbol_p_condition(context, symbol):
    #             probabilities.append((context, self.symbol_p(context, symbol)))
    #             return probabilities
    #         else:
    #             # escape
    #             probabilities.append((context, self.escape_p(context, symbol)))
    #             context = context[1:] if len(context) > 0 else None
