import pynini as pn

import time

def _size_to_pow_2(n):

    for i in range(1, 65):
        if n < 2**i:
            return i - 1

    raise ValueError(f"{n} is larger than 2^64 = {2**64}")


class ImplicitBPEAutomaton:

    def __init__(self):
        self._initialized = False

    @classmethod
    def build_from_pynini(cls, automaton, syms):

        syms_set = set(sym for (_, sym) in syms)

        unigram_info = {}
        explicit_adj = {}
        banned_arcs = {}

        start = automaton.start()

        for arc in automaton.arcs(start):
            if arc.ilabel != 0:
                unigram_info[syms.find(arc.ilabel)] = arc.nextstate

        for state in automaton.states():
            if state != 0:
                seen = []
                explicit_adj[state] = {}
                for arc in automaton.arcs(state):
                    sym = syms.find(arc.ilabel)

                    if arc.nextstate != unigram_info[sym]:
                        explicit_adj[state][sym] = arc.nextstate
                    seen.append(sym)

                banned_arcs[state] = syms_set - set(seen)

        out = cls()
        out._unigram_info = unigram_info
        out._explicit_adj = explicit_adj
        out._banned_arcs = banned_arcs
        out._syms = syms_set
        out._states = automaton.num_states()
        out._initialized = True

        return out

    def estimate_size(self):
        if not self._initialized:
            raise RuntimeError("NOT YET INITIALIZED")

        V, Q = len(self._syms), len(self._banned_arcs)
        V_b, Q_b = _size_to_pow_2(V), _size_to_pow_2(Q)

        unigram_info_bytes = V_b * Q_b
        explicit_adj_bytes = 0
        banned_arcs_bytes = 0
        for state in self._explicit_adj:
            if len(self._explicit_adj[state]) > 0:
                explicit_adj_bytes += Q_b
                explicit_adj_bytes += (V_b + Q_b) * len(self._explicit_adj[state])

        for state in self._banned_arcs:
            if len(self._banned_arcs[state]) > 0:
                banned_arcs_bytes += Q_b
                banned_arcs_bytes += V_b * len(self._banned_arcs[state])
        return unigram_info_bytes + explicit_adj_bytes + banned_arcs_bytes

    def lookup_arc(self, state, sym):
        if not self._initialized:
            raise RuntimeError("NOT YET INITIALIZED")
        if state == 0:
            return (sym, self._unigram_info[sym])
        if sym in self._explicit_arcs[state]:
            return (sym, self._explicit_arcs[state][sym])
        if sym in self._banned_arcs[state]:
            return None
        return (sym, self._unigram_info[sym])

    def enumerate_arcs(self, state):
        if not self._initialized:
            raise RuntimeError("NOT YET INITIALIZED")
        if state == 0:
            return list(self._unigram_info.items())
        out = list(self._explicit_adj[state].items())
        for sym in (
            self._syms - set(self._explicit_adj[state]) - set(self._banned_arcs[state])
        ):
            out.append((sym, self._unigram_info[sym]))

        return out

    def num_states(self):
        if not self._initialized:
            raise RuntimeError("NOT YET INITIALIZED")
        return self._states

    def compare_to_pynini(self, automaton, syms):
        if not self._initialized:
            raise RuntimeError("NOT YET INITIALIZED")
        assert (
            self._states == automaton.num_states()
        ), f"num states mismatch {self._states} vs {automaton.num_states()}"
        for state in automaton.states():
            implicit = sorted(self.enumerate_arcs(state))
            explicit = []
            for arc in automaton.arcs(state):
                explicit.append((syms.find(arc.ilabel), arc.nextstate))
            implicit.sort()
            explicit.sort()

            assert implicit == explicit, f"arcs fail to match up for state {state}"

        return True


def estimate_pynini_size(automaton, syms):
    V_b, Q_b = _size_to_pow_2(syms.num_symbols() - 1), _size_to_pow_2(
        automaton.num_states() - 1
    )
    tot = 0
    for state in automaton.states():
        tot += (V_b + Q_b) * automaton.num_arcs(state)
    return tot + 8 * automaton.num_states()


if __name__ == "__main__":
    for BPE_SIZE in [4000, 8000]:  #, 16000, 32000]:
        print("\n\n")
        print(f"BPE_SIZE {BPE_SIZE}")
        automaton = pn.Fst.read(
            f"precomputed_canonical_automata/wiki_tokenizer_sigma_final_{BPE_SIZE}_merges.fst"
        )
        syms = automaton.input_symbols()

        print("starting implicit conversion")
        start = time.time()
        implicit = ImplicitBPEAutomaton.build_from_pynini(automaton, syms)
        print(f"finished conversion in {time.time() - start:0.2f} seconds")
        print(f"VERIFY EQUALITY: {implicit.compare_to_pynini(automaton, syms)}")
        i_size, e_size = implicit.estimate_size(), estimate_pynini_size(automaton, syms)
        print(
            f"SIZES: implicit: {i_size}, fst: {e_size}, reduction in size: {100 - i_size * 100.0 / e_size :0.4f}%"
        )
        print(f"NUM STATES: {automaton.num_states()}")
        print(
            f"EXPLICIT ARCS: implicit: {len(implicit._unigram_info) + sum(len(implicit._banned_arcs[s]) for s in implicit._banned_arcs) + sum(len(implicit._explicit_adj[s]) for s in implicit._explicit_adj)}, fst: {sum(automaton.num_arcs(s) for s in automaton.states())}"
        )
        print("\n\n")
