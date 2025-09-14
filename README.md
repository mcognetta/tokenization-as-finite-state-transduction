**DISCLAIMER:** This is experimental, academic code. Consider it to be an alpha-version with bugs present, especially outside of the implemented use-case.

# Introduction

This is the reference implementation for *Tokenization as Finite-State Transduction* --- particularly the sections about BPE-pattern promotion.

The paper can be found [here](https://direct.mit.edu/coli/article/doi/10.1162/coli.a.23/132855/Tokenization-as-Finite-State-Transduction).

This repo contains several files/directories which are necessary to run the experiments. Some are referenced directly in the paper, but others are just the underlying implementations.

- `compiler.py`
  - The base implementation of BPE merge gadgets that compiles HuggingFace tokenizers to FSTs
- `train_tokenizer.py`
  - A helper file for training new HuggingFace BPE tokenizers with the required structure
- `speedtest.py`
  - Implementation of the full composition vs precomputed composition benchmark (Table 1)
  - Implementation of the HuggingFace vs FST tokenization experiment (Section 6.4)
- `compression.py`
  - Implementation of the implicit encoding of the BPE automaton (Section 6.5)
- `wiki.train.raw`
  - A copy of the WikiText2 training corpus used to train our pretrained tokenizers
- `tokenizers`
  - A set of precomputed tokenizers of different sizes
  - Each of these used the same configuration and training set
- `precomputed_canonical_automata`
  - **NOTE**: This directory is compressed due to the size, you must uncompress it first.
    -  I.e., `tar -xvf precomputed_canonical_automata.tar.gz`
  - A set of precomputed canonical BPE automata for our precomputed tokenizers
  - We omit the 16k and 32k tokenizer automata, as it is too large to upload to git
    - You can regenerate these from `compiler.py` but it takes a very long time
    - We are exploring other ways to include this file

# Installation

This implementation relies on HuggingFace's tokenizers library and Pynini, a Python wrapper around OpenFST. These can be installed with `pip install tokenizers pynini`.

If `pynini` does not properly install, it can be built with conda: `conda install -c conda-forge pynini`.

# Running

- `speedtest.py`
  - `python speedtest.py --experiment <experiment> --vocab-size <size> --trials <default = 5>`
  - `experiment` is either `single_tokenization` or `edit_distance` depending on which result you want to replicate (Table 1 or Section 6.4)
  - `vocab-size` is one of `[4000, 8000]` depending on the BPE vocabulary size you want to test
    - We omit `16000` and `32000` since its canonical automaton is too large to package together with this repo
  - `trials` is the number of trials to average the timing over
    - The default is 5
  - The output is timing information for the experiment
    - HuggingFace vs precomposed canonical automaton for `single_tokenization`
    - Full composition vs precomposed canonical automaton for `edit_distance`
- `compression.py`
  - `python compression.py`
  - This just runs the compression algorithm for each of `[4000, 8000]` sized BPE tokenizers
  - The output is the number of implicitly encoded arcs for the compressed version as well as the total on-disk space requirements for each representation
- `train_tokenizer.py`
  - `python train_tokenizer.py --corpus-path <corpus file path> --vocab-size <desired vocab size> --save-path <desired output path> [--compile-to-fst]`
  - Output path should be a `*.json` file
  - `--compile-to-fst` is optional and will produce an FST at the same location as the json file
    - **NOTE:** This is *extremely* slow. For 32k it will take more than one day to complete
- `compiler.py`
  - `python compiler.py --tokenizer-json-file <file path> --output-path <compiled fst path>`
  - This will compile an existing tokenizer (from a json representation) to an FST
  - **NOTE:** Like `train_tokenizer.py`'s `--compile-to-fst` flag, this is very slow