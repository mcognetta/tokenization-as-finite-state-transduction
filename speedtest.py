import argparse, time

import pynini as pn
import pywrapfst as fst

from tokenizers import Tokenizer
from compiler import (
    convert_wiki_tokenizer_merge_transducers_iter,
    convert_wiki_tokenizer_unconstrained,
)

ALPHABET = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
UNMARKED_WORDS = [
    "depth",
    "Boris",
    "kilowatts",
    "trades",
    "Aub",
    "airstrip",
    "surprisingly",
    "Soulfly",
    "whereupon",
    "Gan",
    "Grammy",
    "grammar",
    "fans",
    "trim",
    "Goldsmith",
    "Mass",
    "dim",
    "enquiry",
    "cook",
    "riddled",
    "Chun",
    "Gruppo",
    "styles",
    "beats",
    "reassured",
    "Pantanal",
    "defeat",
    "reshaping",
    "Nominated",
    "Janney",
    "Figures",
    "does",
    "Bakula",
    "Lockport",
    "stunned",
    "anytime",
    "fulfilled",
    "Clayton",
    "charg",
    "saccharine",
    "Formossa",
    "fever",
    "statements",
    "successive",
    "Filoil",
    "swayed",
    "Mastodon",
    "infatuated",
    "interior",
    "Chalton",
    "Magilligan",
    "street",
    "masturbates",
    "wading",
    "Urgestein",
    "Nangka",
    "Field",
    "historically",
    "sorties",
    "Vespasian",
    "Rolling",
    "Gomez",
    "hundred",
    "fault",
    "outset",
    "simplicity",
    "leak",
    "Statistical",
    "reduction",
    "infinitesimally",
    "Unforgiven",
    "subjectivity",
    "tunefully",
    "System",
    "earning",
    "renovated",
    "halfway",
    "worth",
    "satire",
    "Suria",
    "Kid",
    "twins",
    "Precipitation",
    "Nigeria",
    "waterfronts",
    "sort",
    "apparent",
    "forbade",
    "noted",
    "clawed",
    "struggles",
    "diction",
    "appointments",
    "operations",
    "Group",
    "reverts",
    "impressed",
    "Ser",
    "glands",
    "Rim",
]
WORDS = [w + "▁" for w in UNMARKED_WORDS]


def benchmark_preconstructed(bpe, automaton, agnostic_transducer, TRIALS=5):
    tot_time = 0.0
    start = time.time()
    base_automaton = automaton

    for _ in range(TRIALS):
        print(f"TRIAL {_}")
        a = base_automaton.copy()

        """
        here we benchmark the entire pipeline of
        linear -> unconstrained lattice -> bpe_constrained_lattice
        """
        start = time.time()
        out = (a @ agnostic_transducer).project("output").rmepsilon()
        out.minimize()
        out = out @ bpe
        tot_time += time.time() - start

        if _ != TRIALS - 1:
            del out

    return tot_time / TRIALS, out.optimize()


def _construct_edit_distance(syms):

    f = pn.Fst()
    one = fst.Weight.one(f.weight_type())
    start = f.add_state()
    f.set_start(start)
    f.set_final(start)

    chars_no_marker, chars_marker = [], []
    for label, sym in syms:
        if len(sym) == 1 and sym in ALPHABET:
            chars_no_marker.append(label)
        if len(sym) == 2 and sym[0] in ALPHABET and sym[-1] == "▁":
            chars_marker.append(label)

    delete, insert, sub_marker_int, sub_marker, sub_no_marker_int, sub_no_marker = (
        f.add_state(),
        f.add_state(),
        f.add_state(),
        f.add_state(),
        f.add_state(),
        f.add_state(),
    )

    f.set_final(delete)
    f.set_final(insert)
    f.set_final(sub_no_marker)
    f.set_final(sub_marker)

    for label in chars_no_marker:
        if label != 0:
            f.add_arc(start, fst.Arc(label, 0, one, delete))
            f.add_arc(start, fst.Arc(0, label, one, insert))
            f.add_arc(start, fst.Arc(label, 0, one, sub_no_marker_int))
            f.add_arc(sub_no_marker_int, fst.Arc(0, label, one, sub_no_marker))

            f.add_arc(start, fst.Arc(label, label, one, start))
            f.add_arc(insert, fst.Arc(label, label, one, insert))
            f.add_arc(delete, fst.Arc(label, label, one, delete))
            f.add_arc(sub_no_marker, fst.Arc(label, label, one, sub_no_marker))

    for label in chars_marker:
        if label != 0:
            f.add_arc(start, fst.Arc(label, 0, one, delete))
            f.add_arc(start, fst.Arc(0, label, one, insert))
            f.add_arc(start, fst.Arc(label, 0, one, sub_marker_int))
            f.add_arc(sub_marker_int, fst.Arc(0, label, one, sub_marker))

            f.add_arc(start, fst.Arc(label, label, one, start))
            f.add_arc(insert, fst.Arc(label, label, one, insert))
            f.add_arc(delete, fst.Arc(label, label, one, delete))
            f.add_arc(sub_marker, fst.Arc(label, label, one, sub_marker))

    f.optimize()
    return f


def benchmark_full_composition(merge_iterators, automaton, BATCH_SIZE=10, TRIALS=5):
    tot_time = 0.0
    base_automaton = automaton

    for trial in range(TRIALS):
        print(f"TRIAL {trial}")
        automaton = base_automaton.copy()
        for idx, merge in enumerate(merge_iterators[trial]):

            """
            here we only benchmark merge compositions

            we specifically don't measure the time taken to construct a
            merge gadget or the time required to minimize the resulting automaton
            as those can both be optimized way more than what we have done here
            """
            temp_start = time.time()
            automaton = automaton @ merge
            tot_time += time.time() - temp_start

            if idx > 0 and idx % BATCH_SIZE == 0:

                automaton = pn.determinize(automaton.project("output").rmepsilon())
                automaton.minimize()

        automaton = pn.determinize(automaton.project("output").rmepsilon())
        automaton.minimize()
    return tot_time / TRIALS, automaton


def _create_linear_fst(word, syms):

    f = pn.Fst()
    one = fst.Weight.one(f.weight_type())

    cur = f.add_state()
    f.set_start(cur)
    word = list(word)
    if word[-1] == "▁":
        word.pop()
        word[-1] = word[-1] + "▁"

    for c in word:
        label = syms.find(c)
        if label == -1:
            print(f"NOT FOUND {label} {c}")
        f.add_arc(cur, fst.Arc(label, label, one, cur := f.add_state()))

    f.set_final(cur)

    f.optimize()
    return f


def _create_linear_fst_from_list(lst, syms):

    f = pn.Fst()
    one = fst.Weight.one(f.weight_type())

    cur = f.add_state()
    f.set_start(cur)

    for token in lst:
        label = syms.find(token)
        if label == -1:
            print(f"NOT FOUND {label} {token}")
        f.add_arc(cur, fst.Arc(label, label, one, cur := f.add_state()))

    f.set_final(cur)

    f.optimize()
    return f


def _fuse_spaces(chars):
    """
    this fuses space tokens in a list of characters. needed as hf pretokenization

    i.e., [a, b, ▁, d, ▁, e] -> [a, b▁, d▁, e]
    """
    result = []
    i = 0
    while i < len(chars):
        if i < len(chars) - 1 and chars[i + 1] == "▁":
            result.append(chars[i] + chars[i + 1])
            i += 2
        else:
            result.append(chars[i])
            i += 1
    return result


def compare_single_tokenization(BPE_SIZE, TRIALS=100):
    s = "As part of its promotion , One Direction performed the song on televised programmes and during their worldwide Take Me Home Tour ( 2013 ) . One Direction performed the track on The Today Show at the Rockefeller Center on 13 November 2012 , to a record crowd estimated at 15 @,@ 000 . ' Kiss You ' was included in the set list of the group ' s 3 December 2012 sold @-@ out show at New York City ' s Madison Square Garden . One Direction delivered a performance of ' Kiss You ' , in front of a video game @-@ themed set , on the final of the ninth series of The X Factor UK on 10 December 2012 . According to the Daily Mail , their ' energetic rendition ' of ' Kiss You ' proved that the group have an elusive quality . On 12 December 2012 , the group also performed the number on the final of the second season of The X Factor USA . Considering One Direction the ' franchise ' s biggest success story ' , an editor for The Huffington Post opined that the boy band ' s prominent presence on both the US and UK versions of The X Factor seemed fitting . Not only Take Me Home Tour , they also performance in Where We Are Tour ( 2014 ) & On the Road Again Tour ( 2015 ) "
    fused = _fuse_spaces(list(s.replace(" ", "▁")))

    tokenizer = Tokenizer.from_file(f"tokenizers/tokenizer_{BPE_SIZE}.json")
    bpe = pn.Fst.read(
        f"precomputed_canonical_automata/wiki_tokenizer_sigma_FINAL_{BPE_SIZE}_merges.fst"
    )
    syms = bpe.input_symbols()
    linear = _create_linear_fst(fused, syms)

    unconstrained, _, _ = convert_wiki_tokenizer_unconstrained(
        f"tokenizers/tokenizer_{BPE_SIZE}.json"
    )

    temp_lattice = pn.compose(linear, unconstrained, connect=False)
    temp_lattice.set_input_symbols(syms)
    temp_lattice.set_output_symbols(syms)
    lattice = (linear @ unconstrained).project("output").optimize()
    canon = (lattice @ bpe).optimize()
    encoding = tokenizer.encode(s).tokens

    linear_syms = _create_linear_fst_from_list(encoding, syms)

    print("verifying canonical tokenizations are equal", linear_syms == canon)

    hf_time, pynini_time = 0.0, 0.0

    for _ in range(TRIALS):
        start = time.time()
        tokenizer.encode(s).tokens
        hf_time += time.time() - start

    print(f"HuggingFace: {hf_time/TRIALS}")

    for _ in range(TRIALS):
        start = time.time()
        lattice = (linear @ unconstrained).project("output").rmepsilon()
        canon = lattice @ bpe
        pynini_time += time.time() - start

    print(f"Transduction: {pynini_time/TRIALS}")


def composition_benchmark(BPE_SIZE=4000, TRIALS=5):

    bpe = pn.Fst.read(
        f"precomputed_canonical_automata/wiki_tokenizer_sigma_FINAL_{BPE_SIZE}_merges.fst"
    )
    syms = bpe.input_symbols()

    unconstrained, _, _ = convert_wiki_tokenizer_unconstrained(
        f"tokenizers/tokenizer_{BPE_SIZE}.json"
    )

    automata = [_create_linear_fst(w, syms) for w in WORDS]

    for a in automata:
        a.set_input_symbols(syms)
        a.set_output_symbols(syms)

    automaton = automata[0]
    for a in automata[1:]:
        automaton = automaton | a

    automaton.optimize()

    automaton = (
        (automaton @ _construct_edit_distance(syms)).project("output").optimize()
    )

    pre_time, pre_auto = benchmark_preconstructed(
        bpe, automaton.copy(), unconstrained, TRIALS=TRIALS
    )
    full_time, full_auto = benchmark_full_composition(
        [
            merge_iter
            for (_, merge_iter) in [
                convert_wiki_tokenizer_merge_transducers_iter(
                    f"tokenizers/tokenizer_{BPE_SIZE}.json"
                )
                for _ in range(TRIALS)
            ]
        ],
        automaton.copy(),
        TRIALS=TRIALS,
    )

    print("\n\n")
    print(f"PRECOMPOSED TIME: {pre_time}")
    print(f"FULL COMPOSITION TIME: {full_time}")

    print(
        f"precomposed is equivalent to transduction? {pn.equivalent(pre_auto.optimize(), full_auto.optimize())}"
    )
    return pre_auto.optimize(), full_auto.optimize()


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment",
        help="the experiment you want to run",
        default="edit_distance",
        choices=["single_tokenization", "edit_distance"],
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        choices=[4000, 8000],
        default=4000,
        help="the bpe vocab size to use",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=5,
        help="the number of trials for benchmarking",
    )

    args = parser.parse_args()
    BPE = args.vocab_size
    TRIALS = args.trials
    if args.experiment == "edit_distance":
        composition_benchmark(BPE, TRIALS=TRIALS)
    elif args.experiment == "single_tokenization":
        compare_single_tokenization(BPE, TRIALS=TRIALS)
