import argparse
import json
import time

import pynini as pn
import pywrapfst as fst


def _create_sigma(alphabet, syms):

    f = pn.Fst()
    one = fst.Weight.one(f.weight_type())

    cur = f.add_state()
    f.set_start(cur)

    for c in alphabet:
        label = syms.find(c)
        if label == -1:
            print(f"NOT FOUND {label} {c}")
        f.add_arc(cur, fst.Arc(label, label, one, cur))

    f.set_final(cur)

    f.optimize()
    return f


def convert_wiki_tokenizer_unconstrained(path):
    j = json.load(open(path, "r"))
    vocab = list(j["model"]["vocab"])

    isyms = pn.SymbolTable()
    osyms = pn.SymbolTable()

    # OpenFst reserves id 0 for epsilon
    isyms.add_symbol("<epsilon>", 0)
    osyms.add_symbol("<epsilon>", 0)

    for w in vocab:
        isyms.add_symbol(w)
        osyms.add_symbol(w)

    lexicon = pn.Fst()
    one = fst.Weight.one(lexicon.weight_type())

    start = lexicon.add_state()
    lexicon.set_start(start)

    for w in vocab:
        w = list(w)
        if w[-1] == "▁":
            w.pop()
            w[-1] = w[-1] + "▁"

        cur = start
        for c in w:
            next = lexicon.add_state()
            lexicon.add_arc(cur, fst.Arc(isyms.find(c), 0, one, next))
            cur = next
        next = lexicon.add_state()
        lexicon.add_arc(cur, fst.Arc(0, osyms.find("".join(w)), one, next))
        lexicon.set_final(next, one)

    lexicon.optimize()

    # close the lexicon, so that we can spell out sequences of subwords
    for s in lexicon.states():
        if lexicon.final(s) == fst.Weight.one(lexicon.weight_type()):
            lexicon.add_arc(s, fst.Arc(0, 0, one, start))

    return lexicon, isyms, osyms


def _convert_merge_to_token(merge):
    l, r = merge

    return l + r


def _construct_merge_iter(merges, syms, vocab):
    for idx, merge in enumerate(merges):
        try:
            l, r = merge
        except:
            print("MERGE FAILED", merge)
            return
        if idx % 100 == 0:
            print(
                f"MERGE {idx}/{len(merges)}: {merge, _convert_merge_to_token(merge), syms.find(l), syms.find(r), syms.find(_convert_merge_to_token(merge))}"
            )

        yield _merge_to_transducer(merge, syms, vocab)


def convert_wiki_tokenizer_merge_transducers_iter(path):

    j = json.load(open(path, "r"))
    vocab = list(j["model"]["vocab"])
    merges = list(j["model"]["merges"])

    syms = pn.SymbolTable()

    # OpenFst reserves id 0 for epsilon
    syms.add_symbol("<epsilon>", 0)

    for w in vocab:
        syms.add_symbol(w)
    return syms, _construct_merge_iter(merges, syms, vocab)


def convert_wiki_tokenizer_merge_transducers(path):

    j = json.load(open(path, "r"))
    vocab = list(j["model"]["vocab"])
    merges = list(j["model"]["merges"])

    syms = pn.SymbolTable()

    # OpenFst reserves id 0 for epsilon
    syms.add_symbol("<epsilon>", 0)

    for w in vocab:
        syms.add_symbol(w)

    merge_transducers = []
    for idx, merge in enumerate(merges):
        try:
            l, r = merge
        except:
            print("MERGE FAILED", merge)
            return
        if idx % 100 == 0:
            print(
                f"MERGE {idx}/{len(merges)}: {merge, _convert_merge_to_token(merge), syms.find(l), syms.find(r), syms.find(_convert_merge_to_token(merge))}"
            )

        merge_transducers.append(_merge_to_transducer(merge, syms, vocab))

    return syms, merge_transducers


def get_wiki_syms(path):

    j = json.load(open(path, "r"))
    vocab = list(j["model"]["vocab"])
    merges = j["model"]["merges"]

    syms = pn.SymbolTable()

    # OpenFst reserves id 0 for epsilon
    syms.add_symbol("<epsilon>", 0)

    for w in vocab:
        syms.add_symbol(w)

    return syms, merges, vocab


def get_wiki_transducers_generator(syms, merges):
    seen_merges = set()
    for idx, merge in enumerate(merges):
        try:
            l, r = merge
        except:
            print("MERGE FAILED", merge)
            return
        if idx % 100 == 0:
            print(
                f"MERGE {idx}/{len(merges)}: {merge, _convert_merge_to_token(merge), syms.find(l), syms.find(r), syms.find(_convert_merge_to_token(merge))}"
            )

        if _convert_merge_to_token(merge) not in seen_merges:
            yield merge
        seen_merges.add(_convert_merge_to_token(merge))


def _merge_to_transducer(merge, syms, vocab):

    l, r = merge
    w = _convert_merge_to_token(merge)
    if l == r:
        return create_double_bpe_kmp_transducer(merge, w, vocab, syms)
    else:
        return create_bpe_kmp_transducer(merge, w, vocab, syms)


def create_double_bpe_kmp_transducer(pair, w, vocab, syms):

    l, r = pair

    f = pn.Fst()
    one = fst.Weight.one(f.weight_type())
    start = f.add_state()
    f.set_start(start)
    f.set_final(start, one)

    for v in vocab:
        if v == w:
            break
        if v != l and v != "<epsilon>":
            f.add_arc(start, fst.Arc(syms.find(v), syms.find(v), one, start))

    cur = f.add_state()
    f.add_arc(start, fst.Arc(syms.find(l), 0, one, cur))

    dead = f.add_state()
    f.add_arc(cur, fst.Arc(0, syms.find(l), one, dead))
    f.set_final(dead)

    next = f.add_state()
    f.set_final(next)
    f.add_arc(cur, fst.Arc(syms.find(r), syms.find(w), one, next))
    f.add_arc(next, fst.Arc(0, 0, one, start))

    for v in vocab:
        if v == w:
            break
        if v != l and v != r and v != "<epsilon>":
            f.add_arc(dead, fst.Arc(syms.find(v), syms.find(v), one, start))
            f.add_arc(next, fst.Arc(syms.find(v), syms.find(v), one, start))

    f.optimize()
    f.set_input_symbols(syms)
    f.set_output_symbols(syms)
    return f


def create_bpe_kmp_transducer(pair, w, vocab, syms):

    l, r = pair

    if l == r:
        return create_double_bpe_kmp_transducer(pair, w, vocab, syms)

    f = pn.Fst()
    one = fst.Weight.one(f.weight_type())
    start = f.add_state()
    f.set_start(start)
    f.set_final(start)
    for v in vocab:
        if v == w:
            break
        if v != l and v != "<epsilon>":
            f.add_arc(start, fst.Arc(syms.find(v), syms.find(v), one, start))

    cur = f.add_state()
    f.add_arc(start, fst.Arc(syms.find(l), 0, one, cur))
    f.add_arc(cur, fst.Arc(syms.find(l), syms.find(l), one, cur))

    dead = f.add_state()
    f.add_arc(cur, fst.Arc(0, syms.find(l), one, dead))
    f.set_final(dead)

    for v in vocab:
        if v == w:
            break
        if v != l and v != r and v != "<epsilon>":
            f.add_arc(dead, fst.Arc(syms.find(v), syms.find(v), one, start))

    next = f.add_state()
    f.add_arc(cur, fst.Arc(syms.find(r), syms.find(w), one, next))

    f.add_arc(next, fst.Arc(0, 0, one, start))

    f.optimize()
    f.set_input_symbols(syms)
    f.set_output_symbols(syms)
    return f


def compile_tokenizer_to_fst(path):
    lexicon, _, osyms = convert_wiki_tokenizer_unconstrained(path)

    syms, merges, vocab = get_wiki_syms(path)
    merge_transducers = get_wiki_transducers_generator(syms, merges)

    alphabet = sorted(
        [c for (_, c) in syms if len(c) == 1 or len(c) == 2 and c.endswith("▁")]
    )
    linear_osyms = _create_sigma(alphabet, syms)

    unconstrained = (linear_osyms @ lexicon).project("output").optimize()

    unconstrained.set_input_symbols(osyms)
    unconstrained.set_output_symbols(osyms)

    constrained = linear_osyms

    BATCH = 5
    OPTIMIZE_BATCH = 20

    for idx, merge in enumerate(merge_transducers):

        merge = _merge_to_transducer(merge, syms, vocab)
        if idx > 500:
            OPTIMIZE_BATCH = 50
        if idx % BATCH == 0:
            intermediate = merge

            if idx % OPTIMIZE_BATCH == 0:
                s = time.time()
                constrained = pn.determinize(constrained)
                constrained.minimize()
                print(f"opt time {time.time() - s:0.4f}")
            if constrained.num_states() > 10000:
                print(
                    f"CONSTRAINED IS TOO LARGE, OPTIMIZING time: {time.ctime(time.time())} states: {constrained.num_states()}"
                )
                constrained = pn.determinize(constrained)
                constrained.minimize()
            print(
                f"MERGE {idx}/{len(merges)} time: {time.ctime(time.time())} states: {constrained.num_states()}"
            )
        else:
            intermediate = intermediate @ merge
            if idx % BATCH == BATCH - 1:
                constrained = (constrained @ intermediate).project("output").rmepsilon()

    constrained.set_input_symbols(syms)
    constrained.set_output_symbols(syms)
    constrained = (constrained @ intermediate).project("output").rmepsilon()
    constrained = pn.determinize(constrained)
    constrained.minimize()
    return constrained


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tokenizer-json-file", type=str, help="tokenizer json file path"
    )
    parser.add_argument(
        "--output-path", type=str, help="the output path for the compiled automaton"
    )
    args = parser.parse_args()

    bpe = compile_tokenizer_to_fst(args.tokenizer_json_file)

    bpe.write(args.output_path)
