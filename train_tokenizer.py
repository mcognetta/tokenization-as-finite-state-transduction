from tokenizers import Tokenizer
from tokenizers.pre_tokenizers import Whitespace

from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

import compiler

import argparse
from pathlib import Path

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--vocab-size",
        type=int,
        help="the vocab size for help with producing the final automaton naming",
    )
    parser.add_argument(
        "--corpus-path",
        help="the training corpus path",
    )
    parser.add_argument(
        "--save-path",
        help="the path to save the tokenizer json config at",
    )

    parser.add_argument(
        "--compile-to-fst", action="store_true", help="convert the tokenizer to a compiled fst"
    )
    args = parser.parse_args()

    tokenizer = Tokenizer(BPE())
    tokenizer.pre_tokenizer = Whitespace()
    trainer = BpeTrainer(vocab_size=args.vocab_size, end_of_word_suffix="▁")
    tokenizer.train([args.corpus_path], trainer)

    tokenizer.save(args.save_path)

    if args.compile_to_fst:
        automaton = compiler.compile_tokenizer_to_fst(args.save_path)

        fst_path = Path(args.save_path).stem + ".fst"
        automaton.write(fst_path)