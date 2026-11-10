# Third-party material

This page lists the third-party material that is in this repository: what it is, where it came from and, where the entry gives them, its copyright line and its licence. This list rests on the files' own comments, on NOTICE and on the project's records. NOTICE has, for the works that have an entry there, the notices and licence texts that their licences ask for; docs/sources-notes.tsv has most of the same matter file by file; for the works whose entry here points to lean/CREDITS.md, that page and lean/CREDITS-FILES.tsv list the files under lean/. The licence texts are in licenses/. Published papers, books, standards and manuals whose methods the code implements or cites are listed in PUBLISHED-WORKS.md. Under "What", the words "taken", "adapted", "quoted", "values" and "embedded" say what of the work is in a file; "whose comment says" gives the file's own words on the work; "named in" lists files that name the work; "listed for" lists the files for which the entry is carried.

## The paper that this protocol builds on

- What: the library `blocks/`, with the files of `prover/`, `verifier/`, `tools/` and `tests/` that run on it, implements and restates its constructions. PUBLISHED-WORKS.md, "The paper that this protocol builds on", says file by file which.
- From: A. Chiesa, G. Fenzi, G. Weissenberg, "Zero-Knowledge IOPPs for Constrained Interleaved Codes", revised version, 2026; an earlier version: Cryptology ePrint Archive 2026/391.

"Further work that this protocol builds on", in comments of the sources, is further work of the three researchers whom README.md's acknowledgments name first.

Where the programs print that rounds rest on "a lemma of the paper that this protocol builds on (see THIRD_PARTY.md), proved there", they mean its generic bound on mutual correlated agreement for additive codes (`blocks/numbers/codes.h`; `blocks/numbers/blocks.h`).

## ggml

- What:
  - adapted: `prover/tests/weights_blocks/make_model.cpp` (wider(): the encoder to Q8_0); `prover/weights_blocks/decode.cuh` (scale_min(); decoding of Q4_K, Q5_K: sub_of(), k_decode_planes, k_cells); `prover/weights_blocks/fold_rule.h` (sub_scale(); decoding of Q4_K, Q5_K, Q6_K: half_bits(), code(), value6(), value(); value8() (Q8_0)); `prover/weights_commitment/fold_weights.cuh` (scale_min(); decoding of Q4_K, Q5_K, Q6_K: half_at(), value_q45(), value_q6(), value()); `weight_sums/tests/oracle.py` (decoder of Q4_K, Q5_K, Q6_K: scale_and_minimum(), values_q4_k_or_q5_k(), values_q6_k()); `weight_sums/weight_sums.cpp` (scale_and_minimum(); decoder of Q4_K, Q5_K, Q6_K: decode_q4_or_q5(), decode_q6(), their cases of decode()).
  - quoted: `prover/weights_blocks/fold_rule.h` (comment at value8(): the definition of a Q8_0 block); `scales/gguf_weights.py` (comment above _decode_q8_0: the definition of a Q8_0 block); `weight_sums/weight_sums.cpp` (comment at decode_q8_0(): the definition of a Q8_0 block).
  - values: `weight_sums/tests/oracle.py` (the value types; the storage types; BLOCK: the sizes of Q4_K, Q5_K, Q6_K); `weight_sums/tests/refusals.py` (numbers of the GGUF format: value types, storage types, BYTES_OF_256, places of d and dmin); `weight_sums/weight_sums.cpp` (Storage: the sizes of Q4_K, Q5_K, Q6_K).
  - `docs/model.md` (Sources: the GGUF container, the block formats and their decoding), whose comment says: “The GGUF container, the block formats and their decoding are ggml's (github.com/ggml-org/llama.cpp, MIT licence; notice in `THIRD_PARTY.md`), as in its Python package gguf 0.19.0.”.
  - `integrations/vllm/integer_model/integer_model/block_weights.py` (decoded_values), whose comment says: “Four expressions of this file are those of ggml (llama.cpp), which is under the MIT License.”.
  - `integrations/vllm/integer_model/tests/test_block_decoding.py` (exact_values), whose comment says: “Four expressions of this file are those of ggml (llama.cpp), which is under the MIT License.”.
  - `prover/engine/tiled.cuh` (tiled_mma()), whose comment says: “The asm template and the order of its operands are the same as in ggml, ggml/src/ggml-cuda/mma.cuh (llama.cpp), MIT; that project's notice is in NOTICE.”.
  - `prover/hash_commitment/hf_range_windows.cuh` (k_block_keys), whose comment says: “k_block_keys repeats statements of that file's k_decode_planes, which are adapted from ggml's dequantize_block_q4_K, _q5_K (ggml/src/ggml-cuda/convert.cu; MIT licence, "Copyright (c) 2023-2024 The ggml authors"; the notice is in NOTICE and THIRD_PARTY.md).”.
  - `prover/hash_commitment/hf_range_windows.cuh` (k_block_keys_wide), whose comment says: “The routine follows k_block_keys above. Its loop over the tasks, its way from a weight row and a block of columns to the block's address …”.
  - `prover/hiding/forest_fused.cuh` (multiply()), whose comment says: “The asm template and the order of its operands are the same as in ggml, ggml/src/ggml-cuda/mma.cuh (llama.cpp), MIT; that project's notice is in NOTICE.”.
  - `prover/hiding/forest_fused_folds.cuh` (multiply_unsigned()), whose comment says: “But for the operand types (u8 for s8), the asm template and the order of its operands are the same as in ggml, ggml/src/ggml-cuda/mma.cuh (llama.cpp) …”.
  - `prover/hiding/forest_fused_second.cuh` (multiply_both_unsigned()), whose comment says: “But for the operand types (u8 for s8), the asm template and the order of its operands are the same as in ggml, ggml/src/ggml-cuda/mma.cuh (llama.cpp) …”.
  - `prover/weights_blocks/decode.cuh` (part for Q8_0 of values()), whose comment says: “where the decoding comes from: see fold_rule.h, value8() and value16().”.
  - `prover/weights_blocks/decode.cuh` (k_decode_planes_wide, k_cells_wide), whose comment says: “The routine follows k_decode_planes above. Its loop over the tasks and its way from a task to the block's address and to the place in the output are that kernel's.”.
  - `prover/weights_blocks/experts_in_one_launch.cuh` (two_scales()), whose comment says: “Four expressions of two_scales() in this file are those of ggml (llama.cpp), which is under the MIT License.”.
  - `prover/weights_blocks/row_pass_fused.cuh` (mma_bytes()), whose comment says: “The asm template and the order of its operands are the same as in ggml, ggml/src/ggml-cuda/mma.cuh (llama.cpp), MIT; that project's notice is in NOTICE.”.
  - `prover/weights_commitment/fold_weights.cuh` (case Q8_0 of value()), whose comment says: “Layout, origin: fr::value8() …”.
  - `prover/weights_commitment/gguf_file.h` (the reader of GGUF files), whose comment says: “Its layout is described at the head of ggml/include/gguf.h. This reader follows that description and the numbers of gguf-py's gguf/constants.py …”.
  - `scales/gguf_weights.py` (_scales_and_minimums()), whose comment says: “Four expressions of this file are those of ggml (llama.cpp), which is under the MIT License.”.
  - `weight_sums/tests/oracle.py` (the writer of GGUF files), whose comment says: “the writer follows ggml's description of the GGUF container.”.
- From: the tensor library of the llama.cpp project, https://github.com/ggml-org/llama.cpp. For the files of scales/ and weight_sums/: commit 794fe23f29fb40104975c91fe19f23798f7c726e, as carried in llama-cpp-python 0.3.7: ggml/src/ggml-common.h (block_q4_K, block_q5_K, block_q6_K); ggml/src/ggml-quants.c (get_scale_min_k4, dequantize_row_q4_K, _q5_K, _q6_K); ggml/include/gguf.h (enum gguf_type); ggml/include/ggml.h (enum ggml_type). For the definition of a Q8_0 block: ggml-quants.h (QK8_0, block_q8_0) at tag b2261, in a third party's public mirror. The files under prover/ name no other release and no commit, nor do integrations/vllm/integer_model/integer_model/block_weights.py and integrations/vllm/integer_model/tests/test_block_decoding.py: ggml/src/ggml-common.h (block_q4_K, block_q5_K, block_q6_K); ggml/src/ggml-quants.c (get_scale_min_k4, dequantize_row_q4_K, _q5_K, _q6_K, quantize_row_q8_0_ref); ggml/src/ggml-cuda/convert.cu (get_scale_min_k4, dequantize_block_q4_K, _q5_K); ggml/src/ggml-cuda/mma.cuh.
- Copyright: Copyright (c) 2023-2024 The ggml authors
- Licence: MIT (licenses/llama.cpp-794fe23f.LICENSE).

ggml's notice, as `licenses/llama.cpp-794fe23f.LICENSE` has it:

```
MIT License

Copyright (c) 2023-2024 The ggml authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

The share of 9 in 24 routed down projections that `prover/weights_blocks/made_up_blocks.h` and `integrations/vllm/integer_model/integer_model/block_weights.py` store as Q5_K follows the counts in the file headers of a public conversion of Llama 4 Maverick into GGUF files: see "What depends on Meta's models".

## gguf-py

- What:
  - adapted: `integrations/vllm/integer_model/integer_model/block_weights.py` (decoded_values: the cases Q4_K, Q5_K, Q6_K; decoded_values: the cases Q8_0, BF16, F16; the numbers that the GGUF container fixes: GGUF_MAGIC, GGUF_VERSION, GGUF_DEFAULT_ALIGNMENT, the value types and their sizes, the block sizes); `prover/tests/weights_blocks/make_model.cpp` (wider(): the encoder to Q8_0; to_binary16(); to_bfloat16()); `scales/gguf_weights.py` (decoder of Q4_K, Q5_K, Q6_K: _binary16, _half_bytes, _bit_fields, _values, _decode_q4_k, _decode_q5_k, _decode_q6_k; _scales_and_minimums; _decode_q8_0; _decode_bf16); `weight_sums/weight_sums.cpp` (decode_q8_0(); case BF16 of decode()).
  - values: `prover/tests/weights_blocks/make_model.cpp` (FILE_TYPE: the types' numbers); `prover/weights_commitment/gguf_file.h` (the storage types' numbers; the sizes in bytes_of()); `scales/gguf_weights.py` (BLOCK_TYPES: the numbers and sizes of F16, Q8_0, BF16); `weight_sums/tests/oracle.py` (BLOCK: the numbers and sizes of F16, Q8_0, BF16); `weight_sums/tests/refusals.py` (the numbers and sizes of Q4_0, Q3_K, F64; the metadata keys; the tensor name rope_freqs); `weight_sums/weight_sums.cpp` (kTypeF16, kTypeQ80, kTypeBF16 and the sizes of Q8_0, F16, BF16).
  - `integrations/vllm/integer_model/integer_model/block_weights.py` (_read_shard: the order of the header's fields), whose comment says: “and the order of the header's fields is that of its gguf/gguf_reader.py.”.
  - `integrations/vllm/integer_model/tests/test_block_decoding.py` (pack_q8_0: the layout), whose comment says: “The layout is that of the package gguf, version 0.19.0 (gguf/quants.py, the class Q8_0).”.
  - `prover/weights_blocks/fold_rule.h` (code(), value6(); the order of the arithmetic steps), whose comment says: “The order of the arithmetic steps, and the form of code() and value6(), follow the same project's Python reader, gguf-py (gguf/quants.py; THIRD_PARTY.md, "gguf-py").”.
  - `prover/weights_commitment/fold_weights.cuh` (parts of the file), whose comment says: “Parts of this file follow gguf-py (gguf/quants.py), which is under the MIT License.”.
  - `scales/tests/test_gguf_weights.py` (the layouts of the blocks of Q8_0, F16, BF16), whose comment says: “The layouts are those of the package gguf, version 0.19.0 (gguf/constants.py, GGML_QUANT_SIZES; gguf/quants.py, the classes Q8_0 and BF16 and the function dequantize).”.
  - `weight_sums/README.md` (the decoder of bin/weight_sums), whose comment says: “For Q8_0 and BF16 the decoder of `bin/weight_sums` is adapted from those of gguf-py 0.19.0 (`gguf/quants.py`), for F16 it follows that file's `dequantize`, and the …”.
  - `weight_sums/tests/oracle.py` (the writer: the padding after every tensor; the three keys of a split model), whose comment says: “Two habits of the written files, the padding after every tensor and the three keys of a split model, are those of the writer in gguf-py 0.19.0 (gguf/gguf_writer.py; MIT licence).”.
  - `weight_sums/weight_sums.cpp` (case F16 of decode()), whose comment says: “the case F16 follows its dequantize(). It is under the MIT License.”.
- From: the llama.cpp project's Python package gguf (the folder gguf-py of github.com/ggml-org/llama.cpp), release 0.19.0: gguf/quants.py (the classes Q4_K, Q5_K, Q6_K; Q4_K.get_scale_min; Q8_0.dequantize_blocks; Q8_0.quantize_blocks; BF16.dequantize_blocks; BF16.quantize_blocks; dequantize; quantize); gguf/constants.py (GGUF_MAGIC, GGUF_VERSION, GGUF_DEFAULT_ALIGNMENT, GGUFValueType, GGML_QUANT_SIZES, GGMLQuantizationType); gguf/gguf_reader.py; gguf/gguf_writer.py (write_header_to_file, _pack).
- Copyright: Copyright (c) 2023 Georgi Gerganov
- Licence: MIT (licenses/gguf-0.19.0.LICENSE).

gguf-py's notice, as `licenses/gguf-0.19.0.LICENSE` has it:

```
MIT License

Copyright (c) 2023 Georgi Gerganov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Hugging Face transformers

- What:
  - adapted: `scales/float_forward.py` (the forward pass: rms_norm, the rotary step, attention, feed-forward, forward); `scales/tests/test_calibrate.py` (reference(): the rotary step).
  - quoted: `scales/float_forward.py` (docstring: the comment "the 128E model does not use qk_norm"); `scales/gguf_weights.py` (comments at MOE_INTERVAL and NO_ROPE_INTERVAL); `scales/policy_tables.py` (docstring of rotary_tables).
  - values: `integrations/vllm/integer_model/examples/try_limits.py` (the nearby model: the names router and experts, what the router gives, the order of the experts' rows); `integrations/vllm/integer_model/integer_model/float_export.py` (tensor names, shapes and configuration fields of a Llama 4 checkpoint); `scales/float_forward.py` (8192 (attention_chunk_size, floor_scale), 0.1 (attn_scale)); `scales/gguf_weights.py` (SHAPE_KEYS: 1e-5 (rms_norm_eps), 500000.0 (default_theta); NO_ROPE_INTERVAL: 4 (no_rope_layer_interval)).
  - `docs/model.md` (Sources: the Llama 4 specifics), whose comment says: “The Llama 4 specifics follow Hugging Face transformers 5.17.0 (its src/transformers/models/llama4/modeling_llama4.py and configuration_llama4.py).”.
  - `integrations/vllm/integer_model/integer_model/model.py` (the steps that are specific to the Llama 4 family; _rotary: neighbouring entries form a pair), whose comment says: “The steps that are specific to the Llama 4 family follow modeling_llama4.py and configuration_llama4.py of Hugging Face transformers (NOTICE).”.
  - `integrations/vllm/integer_model/integer_model/tables.py` (rotary_tables: the angles), whose comment says: “The angles follow modeling_llama4.py and configuration_llama4.py of Hugging Face transformers (NOTICE).”.
  - `scales/float_forward.py` (rms_norm; the rotary frequencies; the query factor), whose comment says: “Single statements are the same: the body of rms_norm, the rotary frequencies and the query factor.”.
  - `scales/make_calibration_set.py` (docstring: step 1, as the guide "Perplexity of fixed-length models" does), whose comment says: “as Hugging Face's guide "Perplexity of fixed-length models" does with the test split of the same corpus (Transformers documentation, huggingface.co/docs/transformers/perplexity).”.
  - further files under `lean/`, which `lean/CREDITS-FILES.tsv` lists.
- From: version 5.17.0 (https://github.com/huggingface/transformers): src/transformers/models/llama4/modeling_llama4.py (apply_rotary_emb; Llama4TextMoe; lines 131, 214, 362, 370 to 372); src/transformers/models/llama4/configuration_llama4.py (class Llama4TextConfig: lines 111, 151, 168, 171-173, 180 to 183, 186 to 196); src/transformers/models/mixtral/modeling_mixtral.py (the loop over experts); src/transformers/integrations/gguf/dequant.py.
- Copyright: Copyright 2018- The Hugging Face team. All rights reserved. (the package's file LICENSE); Copyright 2025 The LLAMA4 and HuggingFace Inc. team. All rights reserved. (modeling_llama4.py and configuration_llama4.py); Copyright 2023 Mistral AI and the HuggingFace Inc. team. All rights reserved. (modeling_mixtral.py); Copyright 2026 The HuggingFace Team. All rights reserved. (integrations/gguf/dequant.py)
- Licence: Apache-2.0 (LICENSE).

NOTICE prints three heads of files of transformers: that of modeling_llama4.py and configuration_llama4.py; that of modeling_mixtral.py, listed for `integrations/vllm/integer_model/integer_model/model.py` (_feed_forward); and that of integrations/gguf/dequant.py, listed for `scales/gguf_weights.py` and `integrations/vllm/integer_model/integer_model/block_weights.py`.

## The program that makes the calibration set

### Optimum of Hugging Face

- What: quoted: `scales/make_calibration_set.py` (docstring at the head: the description of get_dataset; the calls random.seed(seed) and random.randint(0, n - seqlen - 1)).
- From: version 1.18.1 (https://github.com/huggingface/optimum): optimum/gptq/data.py (get_dataset, lines 221 to 245, get_wikitext2, lines 115 to 131).
- Copyright: Copyright 2023 The HuggingFace Team. All rights reserved.
- Licence: Apache-2.0 (LICENSE).

### The GPTQ authors' repository

- What:
  - `scales/make_calibration_set.py` (apply_rule: steps 1 to 3 of the sampling rule), whose comment says: “Steps 1 to 3 are after the WikiText-2 sampler in datautils.py of the GPTQ authors' repository (github.com/IST-DASLab/gptq, get_wikitext2).”.
- From: https://github.com/IST-DASLab/gptq: datautils.py (get_wikitext2).
- Copyright: none is given.
- Licence: Apache-2.0 (licenses/GPTQ.LICENSE).

## A public model's weight files

### a public conversion of Llama 4 Maverick into GGUF files

- What:
  - quoted: `examples/maverick/README.md` (the two fields of the conversion's card, "license: other" and "license_name: llama4"); `examples/maverick/run.sh` (the two fields of the conversion's card, "license: other" and "license_name: llama4").
  - values: `examples/maverick/README.md` (the names of the publisher's repository and of Meta's, meta-llama/Llama-4-Maverick-17B-128E-Instruct); `examples/maverick/run.sh` (the address, the revision and the files' lengths in bytes); `integrations/vllm/integer_model/OPERATING.md` (the public conversion's address and its folder UD-Q4_K_XL); `integrations/vllm/integer_model/integer_model/block_weights.py` (SyntheticBlocks.storage_type: the same share); `prover/weights_blocks/made_up_blocks.h` (type_of(): 9 of the 24 routed down projections are Q5_K).
- From: huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF, revision 41032e5471dd6ea5b349062d978626215d6c5bba, the folders UD-Q4_K_XL, Q8_0 and BF16: the weight files, the counts in their headers and the repository's card.
- Copyright: not known.
- Licence: "license: other", "license_name: llama4".

### Llama 4 Maverick

- What: values: `scales/gguf_weights.py` (the architecture llama4).
- From: Meta's public model (huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct): the public model's shape and configuration.
- Copyright: Copyright © Meta Platforms, Inc. All Rights Reserved.
- Licence: Llama 4 Community License Agreement.

## What depends on Meta's models

Meta's own files, and the weight files that third parties have made from Meta's, are fetched by the user and are not in this repository. This table lists, value by value, what the repository holds that was computed from them, from their file headers, from Meta's tokenizers or from replies that the models served, what may depend on them, and the public figures of the model's shape that it repeats. "A third party's public conversion" is the one of "A public model's weight files", above. The table may lack a place, above all under lean/, for the numbers of the model's shape, and for a value in another coding or unit, a value that is reckoned and not written out, or a short run of small numbers. Under lean/ the searches behind this table reach, of its values, 53,687, 25,856, the parameter line "s 15", the lines S 12, rx 8 and texp0 to texp7 and the scales, and not most of the others; the rows on the model's sizes are not all their places: the vocabulary's size alone, 202,048, stands in 163 Lean files; no search covers the other numbers of the model's shape; S = 12 and rx = 8, written as equations, stand in 32 Lean files beside those listed; the rotary base 500,000 stands in 44 Lean files, and no row names it under lean/; numbers that the planner reckons from the full-size files (plans, ranges, counts, the layout as read values) stand written out in many more Lean files and are not listed; and the searches do not reach every figure that a run on the public weight files gave. What the table says of where each value comes from rests on the records of how the files were made, which are not in this repository, and on what the repository's own files and pages say of it. A value that was fitted, or that a run made under a fitted configuration (a commitment ID, most digests of its files, what the run measured), depends, beside what its row names, on the tokens that its fit read, wherever the value stands. For the fits of examples/maverick to the 4-bit, the 8-bit and the 16-bit files, and what ran under them, these tokens are the calibration set: the rows that list its SHA-256 sum name the works behind it, and for WikiText-2 examples/maverick/README.md, "What is not here, and why", tells what its listing gives as its licences. No other row names the calibration set.

| Value | Where | Where it came from |
|---|---|---|
| a run's finding that every range condition held on each of some hundreds of windows of 1,024 tokens | `lean/GAPS.md`, `lean/LIMITS.md`, `lean/THEOREM.md` | may depend on Llama 4 Maverick |
| the width 5,120 and the head width 128 | `prover/weights_blocks/gguf_source.cuh`, `weight_sums/weight_sums.cpp`, `weight_sums/weight_sums.py` | computed from Llama 4 Maverick: the public model's shape and configuration |
| length and SHA-256 sum of tokenizer.json (Llama 2) | `scales/README.md`, `scales/make_calibration_set.py` | computed from Llama 2: Meta's tokenizer file tokenizer.json (huggingface.co/meta-llama/Llama-2-7b-hf) |
| length and SHA-256 sum of tokenizer.json (Llama 4) | `scales/README.md`, `scales/make_calibration_set.py` | computed from Llama 4 Maverick: Meta's tokenizer file tokenizer.json (huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) |
| 9 of 24 routed down projections stored as Q5_K | `integrations/vllm/integer_model/integer_model/block_weights.py`, `integrations/vllm/integer_model/tests/test_host_memory.py`, `prover/weights_blocks/made_up_blocks.h` | computed from Llama 4 Maverick: the file headers of a third party's public 4-bit GGUF conversion (folder UD-Q4_K_XL) |
| SHA-256 sums of the five weight files | `examples/maverick/weights.sha256` | computed from Llama 4 Maverick: a third party's public 4-bit GGUF conversion (folder UD-Q4_K_XL; the five files of examples/maverick/weights.sha256) |
| the commitment ID; SHA-256 digests of files that a run makes | `examples/maverick/expected.txt` | computed from Llama 4 Maverick: a third party's public 4-bit GGUF conversion (folder UD-Q4_K_XL; the five files of examples/maverick/weights.sha256) |
| the five weight files' lengths in bytes, and the sum of the lengths | `examples/maverick/run.sh` | computed from Llama 4 Maverick: a third party's public 4-bit GGUF conversion (folder UD-Q4_K_XL; the five files of examples/maverick/weights.sha256) |
| SHA-256 sums of the eighteen 16-bit weight files | `examples/maverick/16-bit/weights.sha256` | computed from Llama 4 Maverick: the same third party's 16-bit files (folder BF16, eighteen files) |
| SHA-256 sums of the nine 8-bit weight files | `examples/maverick/8-bit/weights.sha256` | computed from Llama 4 Maverick: the same third party's 8-bit files (folder Q8_0, nine files) |
| the commitment ID; SHA-256 digests of files that its run made | `examples/maverick/16-bit/expected.txt` | computed from Llama 4 Maverick: the same third party's 16-bit files (folder BF16, eighteen files) |
| the commitment ID; SHA-256 digests of files that its run made | `examples/maverick/8-bit/expected.txt` | computed from Llama 4 Maverick: the same third party's 8-bit files (folder Q8_0, nine files) |
| the nine 8-bit and the eighteen 16-bit weight files' lengths in bytes, and the sums of the lengths | `examples/maverick/run.sh` | computed from Llama 4 Maverick: the same third party's 8-bit files (folder Q8_0, nine files) and 16-bit files (folder BF16, eighteen files) |
| `s 15`, the exponent of the scale of the logits | `examples/full-size-clipped-gap/config/policy.txt`, `examples/full-size/config/policy.txt`, `lean/CREDITS-FILES.tsv`, `lean/CREDITS.md`, `lean/Johnson/G/KitFull.lean`, `lean/Johnson/Goal/Meaning/ScopeBytes.lean`, `lean/Johnson/Road/FormsKeysRunK.lean`, `lean/Ladder/Guards.lean`, `lean/Ladder/Stored.lean`, `lean/OnBytes/K/Oracle/SaysLinesFull.lean`, `lean/OnBytes/Model/TestConfig.lean`, `lean/OnBytes/Model/TestConfigBytes.lean`, `lean/OnBytes/Model/TestStages.lean`, `lean/OnBytes/Tie/Pv/Plan/PlanKits.lean`, `lean/STATEMENTS.html`, `paper/paper.pdf`, `prover/examples/target_full_f2/policy.txt`, `prover/examples/target_full_f2_gap/policy.txt`, `prover/examples/target_small_file_f2/policy.txt`, `prover/head/forward.cuh`, `prover/tests/configurations/pinned/policy.txt`, `prover/tests/configurations/pinned_no_proof_of_work/policy.txt`, `prover/tests/configurations/pinned_no_proof_of_work_roomy/policy.txt` | computed from Llama 4 Maverick |
| 53,687 (`eps1_lo`, `eps2_lo`, `epsF_lo`) | `examples/full-size-clipped-gap/config/policy.txt`, `examples/full-size/config/policy.txt`, `prover/examples/target_full_f2/policy.txt`, `prover/examples/target_full_f2_gap/policy.txt` | may depend on Llama 4 Maverick |
| the full-size table of scales | `examples/full-size-clipped-gap/config/KL.bin`, `examples/full-size/config/KL.bin`, `lean/Johnson/G/KitFull.lean`, `lean/Johnson/Road/FormsKeysRunK.lean`, `lean/OnBytes/Tie/Pv/Plan/PlanKits.lean`, `paper/paper.pdf`, `prover/examples/target_full_f2/KL.bin`, `prover/examples/target_full_f2_gap/KL.bin` | may depend on Llama 2 7B and Llama 4 Maverick |
| layout.txt of the full-size configurations, which follows from policy.txt and KL.bin | `examples/full-size-clipped-gap/config/layout.txt`, `examples/full-size/config/layout.txt`, `lean/Johnson/G/KitFull.lean`, `lean/OnBytes/Tie/Pv/Plan/PlanLayoutFull.lean`, `prover/examples/target_full_f2/layout.txt`, `prover/examples/target_full_f2_gap/layout.txt` | may depend on Llama 2 7B and Llama 4 Maverick |
| a piece of layout.txt of the full-size configurations, as decimal bytes | `lean/Johnson/Unfolds/Layout/P01.lean`, `lean/Johnson/Unfolds/Layout/P02.lean`, `lean/Johnson/Unfolds/Layout/P03.lean`, `lean/Johnson/Unfolds/Layout/P04.lean`, `lean/Johnson/Unfolds/Layout/P05.lean`, `lean/Johnson/Unfolds/Layout/P06.lean`, `lean/Johnson/Unfolds/Layout/P07.lean`, `lean/Johnson/Unfolds/Layout/P08.lean`, `lean/Johnson/Unfolds/Layout/P09.lean`, `lean/Johnson/Unfolds/Layout/P10.lean`, `lean/Johnson/Unfolds/Layout/P11.lean`, `lean/Johnson/Unfolds/Layout/P12.lean`, `lean/Johnson/Unfolds/Layout/P13.lean`, `lean/Johnson/Unfolds/Layout/P14.lean` | may depend on Llama 2 7B and Llama 4 Maverick |
| layout.txt of the small configurations, which follows from policy.txt and KL.bin | `examples/accepted-proof-clipped-gap/config/layout.txt`, `examples/accepted-proof/config/layout.txt`, `lean/Johnson/Bind/TwinAnchor.lean`, `lean/OnBytes/Plan/TestReqFrame.lean`, `lean/OnBytes/Tie/Pv/Plan/PlanLayoutSmall.lean`, `prover/examples/target_small_f2/layout.txt`, `prover/examples/target_small_f2_gap/layout.txt`, `prover/examples/target_small_f2_roomy/layout.txt`, `prover/tests/configurations/pinned/layout.txt`, `prover/tests/configurations/pinned_no_proof_of_work/layout.txt`, `prover/tests/configurations/pinned_no_proof_of_work_roomy/layout.txt` | may depend on Llama 2 7B and Llama 4 Maverick |
| the exponent 10 of the residual stream's scale (`--eres 10`; in scales/README.md, e_res 10) | `examples/maverick/run.sh`, `scales/README.md` | may depend on Llama 4 Maverick |
| SHA-256 sum of the calibration set | `examples/maverick/run.sh`, `scales/README.md` | computed from Llama 2: Meta's tokenizer file tokenizer.json (huggingface.co/meta-llama/Llama-2-7b-hf) and WikiText-2 |
| SHA-256 sums of the calibration set and of the rows drawn for it | `scales/make_calibration_set.py` | computed from Llama 2: Meta's tokenizer file tokenizer.json (huggingface.co/meta-llama/Llama-2-7b-hf) and WikiText-2 |
| SHA-256 sum of the calibration set | `examples/maverick/run.sh`, `scales/README.md` | computed from Llama 4 Maverick: Meta's tokenizer file tokenizer.json (huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) and WikiText-2 |
| SHA-256 sum of the calibration set | `scales/make_calibration_set.py` | computed from Llama 4 Maverick: Meta's tokenizer file tokenizer.json (huggingface.co/meta-llama/Llama-4-Maverick-17B-128E-Instruct) and WikiText-2 |
| times, peaks of host memory, and the amounts of host memory and disk space that the example checks for | `docs/figures.tsv`, `docs/limits-all-rows.tsv`, `docs/quickstart.md`, `examples/maverick/figures.txt`, `guide/getting-started.md`, `guide/performance.md`, `guide/serving.md` | computed from Llama 4 Maverick: a third party's public 4-bit GGUF conversion (folder UD-Q4_K_XL; the five files of examples/maverick/weights.sha256) |
| the like times, peaks and amounts for the 8-bit and the 16-bit files | `examples/maverick/16-bit/figures.txt` | computed from Llama 4 Maverick: the same third party's 16-bit files (folder BF16, eighteen files) |
| the like times, peaks and amounts for the 8-bit and the 16-bit files | `examples/maverick/8-bit/figures.txt` | computed from Llama 4 Maverick: the same third party's 8-bit files (folder Q8_0, nine files) |
| the like times, peaks and amounts for the 8-bit and the 16-bit files | `examples/maverick/README.md` | computed from Llama 4 Maverick: the same third party's 8-bit files (folder Q8_0, nine files) and 16-bit files (folder BF16, eighteen files) |
| shares of positions at which the top tokens of two models differed | `docs/figures.tsv`, `docs/limits-all-rows.tsv`, `docs/limits.md`, `guide/known-limits.md`, `paper/paper.pdf` | computed from Llama 4 Maverick: 16-bit weights in two third parties' public copies (the folder BF16 of huggingface.co/unsloth/Llama-4-Maverick-17B-128E-Instruct-GGUF; huggingface.co/RedHatAI/Llama-4-Maverick-17B-128E-Instruct) and the public 4-bit files |
| `sh 8` and `sT 4` | `prover/head/forward.cuh` | may depend on Llama 4 Maverick |
| `sh 8`, and `sT 4` (`sh` - 4) | `examples/accepted-proof-clipped-gap/config/policy.txt`, `examples/full-size-clipped-gap/config/policy.txt`, `prover/examples/target_full_f2_gap/policy.txt`, `prover/examples/target_small_f2_gap/policy.txt` | may depend on Llama 4 Maverick |
| `sh 8`, and three lines that follow from it | `examples/accepted-proof/config/policy.txt`, `examples/full-size/config/policy.txt`, `integrations/vllm/integer_model/tests/configurations/tiny_without_clamps/policy.txt`, `prover/examples/target_full_f2/policy.txt`, `prover/examples/target_small_f2/policy.txt`, `prover/examples/target_small_f2_roomy/policy.txt`, `prover/examples/target_small_file_f2/policy.txt`, `prover/tests/configurations/pinned/policy.txt`, `prover/tests/configurations/pinned_no_proof_of_work/policy.txt`, `prover/tests/configurations/pinned_no_proof_of_work_roomy/policy.txt` | may depend on Llama 4 Maverick |
| 2^sh, as 2^8 and as 256 | `paper/paper.pdf` | may depend on Llama 4 Maverick |
| `MF` and `KF` | `examples/accepted-proof-clipped-gap/config/policy.txt`, `examples/accepted-proof/config/policy.txt`, `examples/full-size-clipped-gap/config/policy.txt`, `examples/full-size/config/policy.txt`, `integrations/vllm/integer_model/tests/configurations/tiny_without_clamps/policy.txt`, `integrations/vllm/integer_model/tests/test_arithmetic_edges.py`, `lean/Inst/Model/Common/LedgerSound.lean`, `lean/Inst/Model/FF/Scales.lean`, `lean/Inst/Model/Head/Trace.lean`, `lean/Inst/Model/SanityRef.lean`, `lean/Inst/Model/ScalesGuard.lean`, `lean/Johnson/AtFull/ProofRows/K3HeadRow.lean`, `lean/Johnson/Bind/TwinAnchor.lean`, `lean/Johnson/Desc/Kinds/Law/CellsWithCostsAt.lean`, `lean/Johnson/Desc/Kinds/Law/K06Condition.lean`, `lean/Johnson/Desc/Kinds/Law/K07MemberAt.lean`, `lean/Johnson/Desc/Kinds/Law/LaidSays.lean`, `lean/Johnson/G/KitFull.lean`, `lean/Johnson/G/TestOneGround.lean`, `lean/Johnson/Goal/Files/MeetsPolicyAtFiles.lean`, `lean/Johnson/Goal/Files/ScopeAtNarrowNorm.lean`, `lean/Johnson/Goal/Meaning/HeadAtFullIs.lean`, `lean/Johnson/Goal/Meaning/ScopeRun.lean`, `lean/Johnson/Meaning/PublicCore.lean`, `lean/Johnson/Road/FormsKeysLaw.lean`, `lean/Johnson/Road/FormsKeysRunK.lean`, `lean/Johnson/Road/HeadL5Pub.lean`, `lean/Johnson/Road/HeadLedgerLin.lean`, `lean/Johnson/Road/HeadLinkMet.lean`, `lean/Johnson/Road/HeadRunK.lean`, `lean/Johnson/Road/HeadSizes.lean`, `lean/Johnson/Road/Lane5TopPub.lean`, `lean/Ladder/Guards.lean`, `lean/Ladder/Stored.lean`, `lean/OnBytes/K/Oracle/SaysLinesFull.lean`, `lean/OnBytes/K/Oracle/SaysLinesSmall.lean`, `lean/OnBytes/Model/TestConfig.lean`, `lean/OnBytes/Model/TestHeadLayout.lean`, `lean/OnBytes/Params/SmallFiles.lean`, `lean/OnBytes/Tie/Pv/Plan/PlanKits.lean`, `lean/Policy/Front/MeetsPolicyIdleAtScales.lean`, `lean/Policy/Front/TeethAllChannels.lean`, `paper/paper.pdf`, `prover/examples/target_full_f2/policy.txt`, `prover/examples/target_full_f2_gap/policy.txt`, `prover/examples/target_small_f2/policy.txt`, `prover/examples/target_small_f2_gap/policy.txt`, `prover/examples/target_small_f2_roomy/policy.txt`, `prover/examples/target_small_file_f2/policy.txt`, `prover/tests/configurations/pinned/policy.txt`, `prover/tests/configurations/pinned_no_proof_of_work/policy.txt`, `prover/tests/configurations/pinned_no_proof_of_work_roomy/policy.txt` | not known |
| four configuration keys with their values (`num_experts_per_tok` 1, `attention_bias` false, `tie_word_embeddings` false, `rope_scaling` null), and five values in words | `lean/IntegerModel/Model.lean` | computed from Llama 4 Maverick: the public model's shape and configuration |
| the rotary base 500,000 | `examples/full-size-clipped-gap/config/policy.txt`, `examples/full-size/config/policy.txt`, `prover/examples/target_full_f2/policy.txt`, `prover/examples/target_full_f2_gap/policy.txt`, `prover/tests/weights_blocks/make_model.cpp` | computed from Llama 4 Maverick: the public model's shape and configuration |
| the two feed-forward widths 16,384 and 8,192 (Construction D.3); that a token uses one routed expert (Section 7.8; Construction D.3) | `paper/paper.pdf` | computed from Llama 4 Maverick: the public model's shape and configuration |
| seven numbers of the model's shape, `(d_model, n_h, n_g, d_h, f, f_e, N_E) = (5120, 40, 8, 128, 16384, 8192, 128)` | `paper/paper.pdf` | computed from Llama 4 Maverick: the public model's shape and configuration |
| seven numbers of the model's shape | `examples/full-size-clipped-gap/config/policy.txt`, `examples/full-size/config/policy.txt`, `prover/examples/target_full_f2/policy.txt`, `prover/examples/target_full_f2_gap/policy.txt` | computed from Llama 4 Maverick: the public model's shape and configuration |
| `C 25856`, `theta 512` and `cap 1`, the policy of the full-size example with a clipped gap | `examples/full-size-clipped-gap/config/policy.txt`, `prover/examples/target_full_f2_gap/policy.txt` | may depend on Llama 4 Maverick: 25,856, 25,600 and 512 are multiples of 2^sh, and sh has a row of its own |
| `C 25856`, `theta 512` and `cap 1`, the policy of the full-size example with a clipped gap, and `C` less 2^sh, 25,600 | `paper/paper.pdf` | may depend on Llama 4 Maverick: 25,856, 25,600 and 512 are multiples of 2^sh, and sh has a row of its own |
| 25,856 | `integrations/vllm/integer_model/OPERATING.md`, `lean/AtFullSide/Bind/LiveAtNamed.lean`, `lean/AtFullSide/Bind/TheTieAtNamed.lean`, `lean/AtFullSide/Honest/K3/SixAtGap.lean`, `lean/CREDITS-FILES.tsv`, `lean/CREDITS.md`, `lean/GAPS.md`, `lean/Johnson/AtFull/Item/ItemAtGap.lean`, `lean/Johnson/AtFull/Item/ItemSaysNo.lean`, `lean/Johnson/AtFull/Top/SoundAtStored.lean`, `lean/Johnson/Bind/Cpp/JoinControls.lean`, `lean/Johnson/Bind/Cpp/OfTie.lean`, `lean/Johnson/Bind/Cpp/SoundAtCommitted.lean`, `lean/Johnson/Bind/Cpp/TheTie.lean`, `lean/Johnson/Bind/Policy/SaltAtGap.lean`, `lean/Johnson/Goal/Description/LaidAtPolicies.lean`, `lean/Johnson/Goal/Files/FilesReadAtRef.lean`, `lean/Johnson/Goal/Files/FilesReadProof.lean`, `lean/Johnson/Goal/Files/MeetsPolicyAtFiles.lean`, `lean/Johnson/Goal/Files/ScopeAtPolicies.lean`, `lean/Johnson/Goal/Meaning/HeadAtFullIs.lean`, `lean/Johnson/Goal/Meaning/ServedMeans.lean`, `lean/Johnson/Goal/Numbers/PlanWrittenScope.lean`, `lean/Johnson/Goal/PolicyAndBinding.lean`, `lean/Johnson/Goal/Skeleton/SoundOfVerifyGaps.lean`, `lean/Johnson/IMR/MeetsPolicyAtPort.lean`, `lean/Johnson/LPFPol/LiveCorollaries.lean`, `lean/Johnson/LPFPol/PublicInScope.lean`, `lean/Johnson/PVR/Program.lean`, `lean/LIMITS.md`, `lean/Ladder.lean`, `lean/Ladder/FilesReadLaws.lean`, `lean/Ladder/Guards.lean`, `lean/Ladder/Stored.lean`, `lean/Policy/Front/MeetsPolicyAtGap.lean`, `lean/Policy/Front/MeetsPolicyBounds.lean`, `lean/Policy/Front/MeetsPolicyIdleAtScales.lean`, `lean/Policy/Rel/AtChosen.lean`, `lean/THEOREM.md`, `prover/examples/TARGET-KITS.md`, `tools/lean-verify/README.md`, `tools/lean-verify/lean_verify_test.sh` | may depend on Llama 4 Maverick: 25,856, 25,600 and 512 are multiples of 2^sh, and sh has a row of its own |
| 25,856 and 25,600 | `examples/full-size-clipped-gap/README.md` | may depend on Llama 4 Maverick: 25,856, 25,600 and 512 are multiples of 2^sh, and sh has a row of its own |
| the length of KL.bin in examples/full-size/config/ | `docs/what-is-not-hidden.md` | may depend on Llama 2 7B and Llama 4 Maverick |
| the lengths and the SHA-256 digests of policy.txt, layout.txt and KL.bin in examples/full-size/config/ | `paper/paper.pdf` | may depend on Llama 2 7B |
| the SHA-256 digests of layout.txt and KL.bin in examples/full-size/config/ | `examples/full-size-clipped-gap/SHA256SUMS` | may depend on Llama 2 7B |
| the SHA-256 digests of policy.txt, layout.txt and KL.bin in examples/full-size/config/ | `examples/full-size/SHA256SUMS` | may depend on Llama 2 7B |
| one or more of the SHA-256 digests of policy.txt, layout.txt and KL.bin in examples/full-size/config/ | `lean/Cb/Pv/KeyLookup/FullPub.lean`, `lean/Johnson/Bridge/BaseAtWritten.lean`, `lean/Johnson/G/KitFull.lean`, `lean/Johnson/G/PricedKeyLit.lean`, `lean/Johnson/Top/H/KeyFamilies.lean`, `lean/Johnson/Top/H/KeyRows.lean`, `lean/Johnson/Top/H/KeyTowers.lean`, `lean/Johnson/Top/H/KeyValues.lean`, `lean/Johnson/Top/Own/OwnStoredFull.lean`, `lean/Ladder/Stored.lean`, `lean/OnBytes/K/Left/TestSaysLines.lean`, `lean/OnBytes/K/Oracle/SaysLinesFull.lean`, `lean/OnBytes/Tie/Pv/KeyStages.lean` | may depend on Llama 4 Maverick |
| the lengths and the SHA-256 digests of policy.txt, layout.txt and KL.bin in examples/full-size/config/ | `paper/paper.pdf` | may depend on Llama 4 Maverick |
| the SHA-256 digests of layout.txt and KL.bin in examples/full-size/config/ | `examples/full-size-clipped-gap/SHA256SUMS` | may depend on Llama 4 Maverick |
| the SHA-256 digests of policy.txt, layout.txt and KL.bin in examples/full-size/config/ | `examples/full-size/SHA256SUMS` | may depend on Llama 4 Maverick |
| the `rp` flags | `examples/accepted-proof-clipped-gap/config/KL.bin`, `examples/accepted-proof/config/KL.bin`, `examples/full-size-clipped-gap/config/KL.bin`, `examples/full-size/config/KL.bin`, `integrations/vllm/integer_model/tests/configurations/tiny_without_clamps/KL.bin`, `prover/examples/target_full_f2/KL.bin`, `prover/examples/target_full_f2_gap/KL.bin`, `prover/examples/target_small_f2/KL.bin`, `prover/examples/target_small_f2_gap/KL.bin`, `prover/examples/target_small_f2_roomy/KL.bin`, `prover/examples/target_small_file_f2/KL.bin`, `prover/tests/configurations/pinned/KL.bin`, `prover/tests/configurations/pinned_no_proof_of_work/KL.bin`, `prover/tests/configurations/pinned_no_proof_of_work_roomy/KL.bin` | computed from Llama 4 Maverick: the public model's shape and configuration |
| a small table of scales, KL.bin, with `M1` = `M2` = 44 in every layer | `prover/examples/target_small_f2/KL.bin`, `prover/examples/target_small_f2_gap/KL.bin` | may depend on Llama 2 7B and Llama 4 Maverick |
| a small table of scales, KL.bin | `prover/examples/target_small_f2_roomy/KL.bin` | may depend on Llama 2 7B and Llama 4 Maverick |
| the bytes of target_small_f2/KL.bin | `lean/Johnson/Road/FormsKeysRunK.lean`, `lean/OnBytes/Model/TestConfig.lean`, `lean/OnBytes/Tie/Pv/Plan/PlanKits.lean`, `prover/tests/configurations/pinned/KL.bin`, `prover/tests/configurations/pinned_no_proof_of_work/KL.bin` | may depend on Llama 2 7B and Llama 4 Maverick |
| the bytes of target_small_f2_roomy/KL.bin | `examples/accepted-proof/config/KL.bin`, `lean/Johnson/Bind/TwinAnchor.lean`, `lean/OnBytes/Params/SmallFiles.lean`, `prover/tests/configurations/pinned_no_proof_of_work_roomy/KL.bin` | may depend on Llama 2 7B and Llama 4 Maverick |
| a small table of scales, KL.bin, with `sh` = 8 in every layer | `integrations/vllm/integer_model/tests/configurations/tiny_without_clamps/KL.bin` | may depend on Llama 2 7B and Llama 4 Maverick |
| a small table of scales, KL.bin, of the small test model in target_small_file_f2 | `prover/examples/target_small_file_f2/KL.bin` | may depend on Llama 2 7B and Llama 4 Maverick |
| 53,687 | `docs/sources-notes.tsv`, `lean/CREDITS-FILES.tsv`, `lean/CREDITS.md`, `lean/Inst/Model/Common/LedgerSound.lean`, `lean/Inst/Model/Head/NoWrapN.lean`, `lean/Inst/Model/Head/Trace.lean`, `lean/Inst/Model/Program.lean`, `lean/Inst/Model/SanityAtt.lean`, `lean/Inst/Model/SanityExp.lean`, `lean/Inst/Model/SanityRef.lean`, `lean/Inst/Model/Scales.lean`, `lean/Johnson/AtFull/ProofRows/K3HeadRow.lean`, `lean/Johnson/Desc/Kinds/Law/CellsFromLogitsAt.lean`, `lean/Johnson/Desc/Kinds/Law/CellsOfSaysAt.lean`, `lean/Johnson/Desc/Kinds/Law/CellsWithCostsAt.lean`, `lean/Johnson/Desc/Kinds/Law/K06Condition.lean`, `lean/Johnson/Desc/Kinds/Law/K07MemberAt.lean`, `lean/Johnson/Desc/Kinds/Law/LaidSays.lean`, `lean/Johnson/Flat/TestScalesOf.lean`, `lean/Johnson/G/KitFull.lean`, `lean/Johnson/Goal/Files/FilesReadChecks.lean`, `lean/Johnson/Goal/Meaning/HeadAtFullIs.lean`, `lean/Johnson/Meaning/PublicCore.lean`, `lean/Johnson/Road/FormsKeysLaw.lean`, `lean/Johnson/Road/FormsKeysLawP.lean`, `lean/Johnson/Road/FormsKeysRunK.lean`, `lean/Johnson/Road/HeadL5Pub.lean`, `lean/Johnson/Road/HeadLedgerLin.lean`, `lean/Johnson/Road/HeadLinkMet.lean`, `lean/Johnson/Road/HeadRunK.lean`, `lean/Johnson/Road/HeadSizes.lean`, `lean/Johnson/Road/Lane5ScalesRef.lean`, `lean/Johnson/Road/Lane5TopPub.lean`, `lean/Ladder/Guards.lean`, `lean/Ladder/Stored.lean`, `lean/OnBytes/K/Oracle/SaysLinesFull.lean`, `lean/OnBytes/Tie/Pv/Plan/PlanKits.lean`, `lean/Policy/Front/TeethAllChannels.lean`, `lean/STATEMENTS.html`, `prover/examples/TARGET-KITS.md`, `scales/worst-case-bounds.md` | may depend on Llama 4 Maverick |
| 53,687, written 53687 | `integrations/vllm/integer_model/tests/test_arithmetic_edges.py` | may depend on Llama 4 Maverick |
| the bytes of target_small_f2_gap/KL.bin | `examples/accepted-proof-clipped-gap/config/KL.bin` | may depend on Llama 2 7B and Llama 4 Maverick |
| one or more of the SHA-256 digests of the small configurations' policy.txt, layout.txt and KL.bin | `examples/accepted-proof-clipped-gap/SHA256SUMS`, `examples/accepted-proof/SHA256SUMS`, `lean/Johnson/Bridge/BaseAtWritten.lean`, `lean/OnBytes/K/Left/TestSaysLines.lean`, `lean/OnBytes/K/Oracle/SaysLinesSmall.lean` | may depend on Llama 2 7B and Llama 4 Maverick |
| the planner's digest of policy.txt, layout.txt and KL.bin in examples/full-size/config/ together | `lean/Johnson/G/KitFull.lean`, `lean/Ladder/Guards.lean`, `lean/OnBytes/Params/EntryDryRecord.lean`, `lean/OnBytes/Plan/TestVariants.lean` | may depend on Llama 2 7B and Llama 4 Maverick |
| how many stored replies of serving engines passed under fitted limits; at how many positions the served token was ahead of the twenty tokens that vLLM lists; counts of replies and requests that got a proof or got none | `integrations/vllm/integer_model/OPERATING.md` | computed from Llama 4 Maverick: replies served by vLLM 0.29.0 from a third party's FP8 checkpoint (by the record of the reply sets: huggingface.co/RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8) and from a 16-bit copy of the model (a record of a fetching names huggingface.co/RedHatAI/Llama-4-Maverick-17B-128E-Instruct) |
| memory and times of runs on real weight files of Llama 4 Maverick | `integrations/vllm/integer_model/OPERATING.md` | computed from Llama 4 Maverick: real weight files of Llama 4 Maverick |
| part of the model's shape, in the docstring | `integrations/vllm/integer_model/tests/test_arithmetic_edges.py` | computed from Llama 4 Maverick: the public model's shape and configuration |
| sizes of the model's shape or its vocabulary size | `integrations/vllm/integer_model/tests/lean_vectors/CompareVectors.lean`, `integrations/vllm/integer_model/tests/lean_vectors/README.md`, `integrations/vllm/integer_model/tests/lean_vectors/make_vectors.py`, `integrations/vllm/integer_model/tests/lean_vectors/run.py` | computed from Llama 4 Maverick: the public model's shape and configuration |
| some of the counts of runs on real weight files of Llama 4 Maverick, and a few more of the same runs | `integrations/vllm/integer_model/OPERATING.md` | computed from Llama 4 Maverick: replies served by vLLM 0.29.0 from a third party's FP8 checkpoint (by the record of the reply sets: huggingface.co/RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8) and from a 16-bit copy of the model (a record of a fetching names huggingface.co/RedHatAI/Llama-4-Maverick-17B-128E-Instruct) |
| the model's shape, in a comment | `integrations/vllm/integer_model/tests/block_tools.py` | computed from Llama 4 Maverick: the public model's shape and configuration |
| the model's shape | `docs/glossary.md`, `guide/terms.md`, `integrations/vllm/integer_model/tests/test_host_memory.py` | computed from Llama 4 Maverick: the public model's shape and configuration |
| the vocabulary size, 202,048 | `integrations/vllm/integer_model/tests/test_vllm_engine.py`, `integrations/vllm/integer_model/tests/test_vllm_model_class.py` | computed from Llama 4 Maverick: the public model's shape and configuration |
| counts on what was served from Llama 4 Scout | `docs/limits-all-rows.tsv`, `docs/results.md`, `integrations/LIMITS.md`, `integrations/vllm/integer_model/OPERATING.md` | computed from Llama 4 Scout: replies and tokens served by vLLM 0.29.0 from a third party's copy of the model's weights (by the record of the fetching: huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct at revision afd8e498c87bda51c7ea8ec68ea2f7c066e6340b) |
| 3,076,096 matrix rows; 1,527,800 pairs of rows | `blocks/frame/files.h` | computed from Llama 4 Maverick: the public model's shape and configuration |
| 3,076,096 rows; 1,527,800 pairs of rows | `blocks/frame/tests/wf_frame_test.cpp` | computed from Llama 4 Maverick: the public model's shape and configuration |
| 3,076,096 rows | `prover/hash_commitment/tests/hf_params_test.cpp` | computed from Llama 4 Maverick: the public model's shape and configuration |
| a table of 3,076,096 rows | `prover/hash_commitment/hf_opening_params.h` | computed from Llama 4 Maverick: the public model's shape and configuration |
| the 48 layers (Definition 3.6); the vocabulary, as the tokens 0 to 202047 (Definition 3.1); the 3,076,096 rows and the 25 segments | `paper/paper.pdf` | computed from Llama 4 Maverick: the public model's shape and configuration |
| the model's shape (48 layers, 128 routed experts) | `examples/full-size/README.md` | computed from Llama 4 Maverick: the public model's shape and configuration |
| the start token, the id 200,000 | `scales/make_calibration_set.py` | computed from Llama 4 Maverick: the public model's shape and configuration |
| which layers are without a rotary step (rp) | `docs/model.md` | computed from Llama 4 Maverick: the public model's shape and configuration |
| 14 cuts of replies that Maverick had sampled met the limits and got a proof | `integrations/vllm/REFERENCE.md` | computed from Llama 4 Maverick: stored replies |
| counts of one run with proofs at full size on 178 requests, 128 of them greedy replies that vLLM served | `integrations/LIMITS.md`, `integrations/vllm/integer_model/OPERATING.md`, `integrations/vllm/integer_model/README.md` | computed from Llama 4 Maverick: replies served by vLLM 0.29.0 from a 16-bit copy of the model (a record of a fetching names huggingface.co/RedHatAI/Llama-4-Maverick-17B-128E-Instruct); stored replies |
| counts of runs in which vLLM served 16 greedy replies from the integer model itself, and each got a proof | `docs/limits-all-rows.tsv`, `docs/results.md`, `integrations/LIMITS.md`, `integrations/README.md`, `integrations/vllm/README.md`, `integrations/vllm/REFERENCE.md`, `integrations/vllm/integer_model/OPERATING.md`, `integrations/vllm/integer_model/README.md` | computed from Llama 4 Maverick: replies served by vLLM from the integer model, loaded from a third party's public 4-bit conversion |
| counts on Maverick's own served replies and tokens | `docs/results.md` | computed from Llama 4 Maverick: replies served by vLLM 0.29.0 from a third party's FP8 checkpoint (by the record of the reply sets: huggingface.co/RedHatAI/Llama-4-Maverick-17B-128E-Instruct-FP8) and from a 16-bit copy of the model (a record of a fetching names huggingface.co/RedHatAI/Llama-4-Maverick-17B-128E-Instruct) |
| figures of the prover's runs on requests made from stored replies | `docs/results.md` | computed from Llama 4 Maverick: stored replies |
| rates, times, sizes, counts and shares from runs on the public weight files and from served replies | `README.md`, `docs/figures.tsv`, `docs/limits.md`, `docs/quickstart.md`, `guide/getting-started.md`, `guide/known-limits.md`, `guide/performance.md`, `paper/paper.pdf` | computed from Llama 4 Maverick: the public weight files; replies served |
| the sum of the five weight files' lengths, 232.13 GB (232 GB in README.md) | `README.md`, `docs/figures.tsv`, `docs/quickstart.md`, `guide/getting-started.md` | computed from Llama 4 Maverick: a third party's public 4-bit GGUF conversion (folder UD-Q4_K_XL; the five files of examples/maverick/weights.sha256) |
| three ranges of cost that are reckoned against plain baselines | `docs/figures.tsv`, `guide/performance.md` | computed from Llama 4 Maverick: the model's FP8 release, served by stock vLLM 0.29.0 |
| counts on what was served from Llama 4 Scout (SO-26.1) | `docs/limits-further.md` | computed from Llama 4 Scout: replies and tokens served by vLLM 0.29.0 from a third party's copy of the model's weights (by the record of the fetching: huggingface.co/unsloth/Llama-4-Scout-17B-16E-Instruct at revision afd8e498c87bda51c7ea8ec68ea2f7c066e6340b) |
| some of the model's sizes (48 layers, d = 5120, 128 experts), in the head comment | `lean/Johnson/Road/K4/PubLinesRec.lean` | computed from Llama 4 Maverick: the public model's shape and configuration |
| the full-size example's public parameter file whole, as a string or as a list of its lines | `lean/Johnson/G/KitFull.lean`, `lean/Johnson/Road/FormsKeysRunK.lean`, `lean/OnBytes/Tie/Pv/Plan/PlanKits.lean` | may depend on Llama 4 Maverick |
| a small configuration's parameter file whole, as a list of its lines | `lean/Johnson/Bind/TwinAnchor.lean`, `lean/Johnson/Road/FormsKeysRunK.lean`, `lean/OnBytes/Model/TestConfig.lean`, `lean/OnBytes/Tie/Pv/Plan/PlanKits.lean` | may depend on Llama 2 7B and Llama 4 Maverick |
| the model's sizes, as lines of the public parameter file | `lean/Ladder/Stored.lean`, `lean/STATEMENTS.html` | computed from Llama 4 Maverick: the public model's shape and configuration |
| the model's sizes, as literals | `lean/Inst/Model/Program.lean`, `lean/IntegerModel/Arrays/ToTensors.lean`, `lean/IntegerModel/Runnable/ArrayModel.lean`, `lean/STATEMENTS.html`, `lean/VerifiableInference/Bounds/SomeWeightsPass.lean`, `lean/VerifiableInference/Bounds/SomeWeightsPassAtValidSettings.lean`, `lean/VerifiableInference/ModelWeights.lean` | computed from Llama 4 Maverick: the public model's shape and configuration |
| one or more of the SHA-256 digests of policy.txt, layout.txt and KL.bin in examples/full-size/config/ | `lean/Cb/Pv/KeyLookup/FullPub.lean`, `lean/Johnson/Bridge/BaseAtWritten.lean`, `lean/Johnson/G/KitFull.lean`, `lean/Johnson/G/PricedKeyLit.lean`, `lean/Johnson/Top/H/KeyFamilies.lean`, `lean/Johnson/Top/H/KeyRows.lean`, `lean/Johnson/Top/H/KeyTowers.lean`, `lean/Johnson/Top/H/KeyValues.lean`, `lean/Johnson/Top/Own/OwnStoredFull.lean`, `lean/Ladder/Stored.lean`, `lean/OnBytes/K/Left/TestSaysLines.lean`, `lean/OnBytes/K/Oracle/SaysLinesFull.lean`, `lean/OnBytes/Tie/Pv/KeyStages.lean` | may depend on Llama 2 7B |
| the library's reference table of scales | `lean/Cm/Rel2Guards.lean`, `lean/Inst/Model/Program.lean`, `lean/Inst/Model/Scales.lean`, `lean/Johnson/Flat/SettingsRead.lean`, `lean/Johnson/Goal/Meaning/ScopeRun.lean`, `lean/Ladder/Guards.lean`, `lean/STATEMENTS.html` | may depend on Llama 2 7B and Llama 4 Maverick |
| 2,307, the number of rows of the softmax table at `S 12` and `rx 8` (`expRows` in the Lean library) | `docs/format/FORMAT.md`, `lean/CREDITS-FILES.tsv`, `lean/CREDITS.md`, `lean/Inst/Model/Att/HonestArith.lean`, `lean/Inst/Model/Att/Index.lean`, `lean/Inst/Model/Att/NoWrap2.lean`, `lean/Inst/Model/Att/Scales.lean`, `lean/Inst/Model/Att/Trace.lean`, `lean/Inst/Model/ProgTests.lean`, `lean/Inst/Model/Program.lean`, `lean/Inst/Model/Scales.lean`, `lean/Inst/Model/ScalesGuard.lean`, `lean/Inst/Model/Tables.lean`, `lean/Inst/Model/TablesData.lean`, `lean/Inst/Model/TablesExp.lean`, `lean/Inst/Model/TablesRef.lean`, `lean/Inst/Model/TablesSanity.lean`, `lean/IntegerModel/Arrays/OfTensors.lean`, `lean/IntegerModel/Readable/Exists.lean`, `lean/IntegerModel/Runnable/ArrayTables.lean`, `lean/IntegerModel/Runnable/AtModelSizes.lean`, `lean/Johnson/AtFull/Desc/Lots/K3Written.lean`, `lean/Johnson/AtFull/Public/AttLaws.lean`, `lean/Johnson/AtFull/Shape/Scope/RelScope.lean`, `lean/Johnson/Desc/ProofFile/Honest/K5/Law/SizesOffPlanes.lean`, `lean/Johnson/Flat/ClaimsAny.lean`, `lean/Johnson/G/XShapeLit.lean`, `lean/Johnson/HonestTrace/Joint/K3PadEqAttention.lean`, `lean/Johnson/HonestTrace/Places.lean`, `lean/Johnson/HonestTrace/Run/SetUpAttPlan.lean`, `lean/Johnson/HonestTrace/Run/SetUpAttSmall.lean`, `lean/Johnson/Road/FormsKeysFits.lean`, `lean/Johnson/Road/Lane5AttAddr.lean`, `lean/Johnson/Road/Lane5AttCalls.lean`, `lean/Johnson/Road/Lane5AttExp.lean`, `lean/Johnson/Road/Lane5AttLink.lean`, `lean/Johnson/Road/Lane5AttMet.lean`, `lean/Johnson/Road/Lane5AttPort.lean`, `lean/Johnson/Road/Lane5AttShape.lean`, `lean/Johnson/Road/Lane5AttTabs.lean`, `lean/Johnson/Road/Lane5AttTabsEq.lean`, `lean/Johnson/Road/Lane5AttWit.lean`, `lean/Johnson/Top/H/LookupC.lean`, `lean/Johnson/Top/Own/OwnStoredTablesRun.lean`, `lean/Johnson/Unfolds/Exp/Closed.lean`, `lean/Johnson/Unfolds/Exp/Digest.lean`, `lean/OnBytes/Model/TestAttentionShape.lean`, `lean/OnBytes/Tie/Pv/Lookup/Test/FoNumsTest.lean`, `lean/OnBytes/Tie/Pv/Lookup/Test/TestTablesOfModel.lean`, `lean/STATEMENTS.html`, `lean/VerifiableInference/ModelWeights.lean`, `lean/scripts/statements-html/front/rows.txt`, `scales/tests/test_policy_tables.py` | may depend on Llama 2 7B and Llama 4 Maverick |
| `S 12`, `rx 8`, and the digest lines texp0 to texp7 that follow from them | `examples/accepted-proof-clipped-gap/config/policy.txt`, `examples/accepted-proof/config/policy.txt`, `examples/full-size-clipped-gap/config/policy.txt`, `examples/full-size/config/policy.txt`, `lean/Johnson/Bind/TwinAnchor.lean`, `lean/Johnson/Flat/SettingsRead.lean`, `lean/Johnson/Flat/TestModelLines.lean`, `lean/Johnson/G/KitFull.lean`, `lean/Johnson/Goal/Meaning/ScopeBytes.lean`, `lean/Johnson/Goal/Meaning/ScopeScan.lean`, `lean/Johnson/HonestTrace/Run/SetUpAttSmall.lean`, `lean/Johnson/Road/FormsKeysRunK.lean`, `lean/Johnson/Road/Lane5AttWit.lean`, `lean/Johnson/Road/Lane5KvCfg.lean`, `lean/Johnson/Unfolds/Exp/Closed.lean`, `lean/Ladder/Stored.lean`, `lean/OnBytes/K/Oracle/SaysLinesFull.lean`, `lean/OnBytes/K/Oracle/SaysLinesSmall.lean`, `lean/OnBytes/Model/TestAttentionConfig.lean`, `lean/OnBytes/Model/TestAttentionPlan.lean`, `lean/OnBytes/Model/TestAttentionShape.lean`, `lean/OnBytes/Model/TestConfig.lean`, `lean/OnBytes/Model/TestConfigBytes.lean`, `lean/OnBytes/Model/TestVerifyPlan.lean`, `lean/OnBytes/Params/SmallFiles.lean`, `lean/OnBytes/Tie/Pv/Lookup/Test/TestTablesOfModel.lean`, `lean/OnBytes/Tie/Pv/Plan/PlanKits.lean`, `lean/STATEMENTS.html`, `prover/examples/target_full_f2/policy.txt`, `prover/examples/target_full_f2_gap/policy.txt`, `prover/examples/target_small_f2/policy.txt`, `prover/examples/target_small_f2_gap/policy.txt`, `prover/examples/target_small_f2_roomy/policy.txt`, `prover/examples/target_small_file_f2/policy.txt`, `prover/tests/configurations/pinned/policy.txt`, `prover/tests/configurations/pinned_no_proof_of_work/policy.txt`, `prover/tests/configurations/pinned_no_proof_of_work_roomy/policy.txt` | may depend on Llama 2 7B and Llama 4 Maverick |
| some of the library's reference scales | `lean/Inst/Model/Common/LedgerSound.lean`, `lean/Inst/Model/FF/Scales.lean`, `lean/Inst/Model/ProgTests.lean`, `lean/Inst/Model/SanityRef.lean`, `lean/Inst/Model/ScalesGuard.lean`, `lean/Inst/Model/ScalesSanity.lean`, `lean/IntegerModel/Arrays/ToTensors.lean`, `lean/Johnson/AtFull/ProofRows/K3HeadRow.lean`, `lean/Johnson/Bind/TwinAnchor.lean`, `lean/Johnson/Desc/Kinds/Law/CellsWithCostsAt.lean`, `lean/Johnson/Desc/Kinds/Law/K06Condition.lean`, `lean/Johnson/Desc/Kinds/Law/K07MemberAt.lean`, `lean/Johnson/Desc/Kinds/Law/LaidSays.lean`, `lean/Johnson/G/TestOneGround.lean`, `lean/Johnson/Goal/Files/MeetsPolicyAtFiles.lean`, `lean/Johnson/Goal/Files/ScopeAtNarrowNorm.lean`, `lean/Johnson/Goal/Meaning/HeadAtFullIs.lean`, `lean/Johnson/Meaning/PublicCore.lean`, `lean/Johnson/Road/FormsKeysLaw.lean`, `lean/Johnson/Road/HeadL5Pub.lean`, `lean/Johnson/Road/HeadLedgerLin.lean`, `lean/Johnson/Road/HeadLinkMet.lean`, `lean/Johnson/Road/HeadRunK.lean`, `lean/Johnson/Road/HeadSizes.lean`, `lean/Johnson/Road/Lane5TopPub.lean`, `lean/Ladder/FilesReadLaws.lean`, `lean/OnBytes/K/Oracle/SaysLinesFull.lean`, `lean/OnBytes/K/Oracle/SaysLinesSmall.lean`, `lean/OnBytes/Model/TestConfig.lean`, `lean/OnBytes/Model/TestHeadLayout.lean`, `lean/OnBytes/Model/TestLayer.lean`, `lean/OnBytes/Model/TestLayerLayout.lean`, `lean/OnBytes/Params/SmallFiles.lean`, `lean/Policy/Front/MeetsPolicyIdleAtScales.lean`, `lean/Policy/Front/TeethAllChannels.lean` | may depend on Llama 2 7B and Llama 4 Maverick |
| a table of scales made by a written rule from Lean's ranges and these files' tests, two of whose shifts, and the norms' bits - shift, have the reference table's number, or numbers reckoned from them | `lean/VerifiableInference/Bounds/LeastBoundOfLength.lean`, `lean/VerifiableInference/Bounds/PassingWeightsTellPromptsApart.lean`, `lean/VerifiableInference/Bounds/SomeWeightsDoNotPass.lean`, `lean/VerifiableInference/Bounds/SomeWeightsPass.lean`, `lean/VerifiableInference/Bounds/SomeWeightsPassLengthCheck.lean` | may depend on Llama 2 7B and Llama 4 Maverick as far as those places do: the value shift is fixed by a test that, by arithmetic with no theorem, holds at the reference table too; the ranges and the tests are those at the model's sizes |
| the sum of the five weight files' lengths, rounded, 232 GB | `prover/hash_commitment/hf_commit_weights.cuh` | computed from Llama 4 Maverick: a third party's public 4-bit GGUF conversion (folder UD-Q4_K_XL; the five files of examples/maverick/weights.sha256) |
| a layer's record of target_small_f2/KL.bin | `lean/Johnson/Road/FormsKeysRunK.lean`, `lean/OnBytes/Model/TestConfig.lean`, `lean/OnBytes/Model/TestLayer.lean`, `lean/OnBytes/Model/TestLayerLayout.lean`, `lean/OnBytes/Tie/Pv/Plan/PlanKits.lean` | may depend on Llama 2 7B and Llama 4 Maverick |
| verdicts, counts and one share from one test of hiding, run on commitments to these weights | `docs/limits-all-rows.tsv`; a count from it also in `guide/weight-hiding.md`, `docs/how-hiding-is-checked.md` | computed from Llama 4 Maverick: a third party's public 4-bit GGUF conversion (folder UD-Q4_K_XL; the five files of examples/maverick/weights.sha256) |

## VerInf

- What:
  - `integrations/vllm/integer_model/README.md` (Third-party work in this package: the entry on VerInf), whose comment says: “**VerInf** (J. Petrie, commit 89903c96). Used in integer_model/block_weights.py (committed_integers: the conversion rule grew out of VerInf's).”.
  - `integrations/vllm/integer_model/integer_model/block_weights.py` (committed_integers: the rule), whose comment says: “The rule grew out of that of VerInf (github.com/JamesPetrie/VerInf, MIT licence).”.
  - `prover/hash_commitment/hf_commit_windows.cuh` (stored_integer(): the rule), whose comment says: “The rule of stored_integer() grew out of VerInf's rule for converting stored weights to integers: J. Petrie, VerInf, source code and README, github.com/JamesPetrie/VerInf, commit 89903c96 …”.
  - `prover/weights_blocks/fold_rule.h` (the rule for converting the stored weights to integers), whose comment says: “The rule for converting the stored weights to integers grew out of VerInf's … as far as we can tell, the rule only, and no text but two local names in value6().”.
  - `prover/weights_commitment/fold_weights.cuh` (the rule of the conversion), whose comment says: “The rule of this conversion, and as far as we can tell only the rule, grew out of VerInf's: see fold_rule.h's head.”.
  - `weight_sums/weight_sums.cpp` (the rule (file head)), whose comment says: “The rule grew out of VerInf's rule for converting stored weights to integers: J. Petrie, VerInf, source code and README, github.com/JamesPetrie/VerInf, commit 89903c96 (28 August 2026), MIT License.”.
  - `weight_sums/weight_sums.py` (the rule (comment at the rule's constants)), whose comment says: “The rule grew out of VerInf's rule for converting stored weights to integers: J. Petrie, VerInf, source code and README, github.com/JamesPetrie/VerInf, commit 89903c96 (28 August 2026), MIT License.”.
- From: J. Petrie, https://github.com/JamesPetrie/VerInf, commit 89903c96 (28 August 2026).
- Copyright: Copyright (c) 2026 James Petrie
- Licence: MIT (licenses/VerInf-89903c96.LICENSE; the MIT text: NOTICE, under ggml).

The rule for converting the stored weights to integers grew out of VerInf's. `docs/model.md`, "11 From the model's files to the integers", defines the rule. The files that apply it: the files of `prover/weights_blocks/`, `prover/weights_commitment/fold_weights.cuh`, `prover/hash_commitment/hf_commit_windows.cuh` (stored_integer), `prover/hash_commitment/hf_range_windows.cuh` (k_block_keys), `prover/tests/weights_blocks/make_model.cpp`, `integrations/vllm/integer_model/integer_model/block_weights.py`, `weight_sums/weight_sums.cpp`, `weight_sums/weight_sums.py`; `weight_sums/tests/oracle.py` has a further implementation of the rule.

## BLAKE2b

### BLAKE2 reference source code package

- What:
  - adapted: `blocks/hash/blake2b_f.h` (blake2b_IV0 to blake2b_IV7, blake2b_sigma, b2_right (host code), BLK_B2_G, BLK_B2_ROUND, F, param_word0, begin64, first64 [from librsync's copy]); `blocks/hash/hash.h` (HasherT: the members h, t, f, buf and buflen of its state; load64, store32 and store64; the counter's increment; the last block's flag; the loop that loads a block's words; push() and finish(); by way of prover/second_writing/blake2b_plain.h); `blocks/hash/lanes.cuh` (blake2b_sigma<r>, BLK_LANES_G, BLK_LANES_ROUND, LaneCall<Blake2bF>::begin, LaneCall<Blake2bF>::call [from librsync's copy]); `docs/format/format_rules.py` (from rotr64 to the end of blake2b_F; the first chaining value [from Qt's copy]); `integrations/vllm/integer_model/integer_model/tables.py` (from _rotr64 to the end of _compress; the first chaining value [from Qt's copy]); `lean/Run/Blake2b/Compress.lean` (IV, rotr64, G, ROUND, State, compress); `lean/Run/Blake2b/Hash.lean` (the definitions of sections 2 to 5); `lean/Run/Blake2b/Salted.lean` (all five definitions); `lean/Run/Blake2b/Spec.lean` (the whole plain definition); `lean/STATEMENTS.html` (the Lean source that the page carries of Run/Blake2b/Compress.lean, Run/Blake2b/Hash.lean and Run/Blake2b/SaltedPlain.lean); `prover/common/blake2b_f.h` (blake2b_IV, blake2b_sigma, right (host code), F2B_G, F2B_ROUND, F, first_state [from librsync's copy]); `prover/second_writing/blake2b_plain.h` (the two tables, the rotation, G, ROUND, blake2b_compress but for its first loop [from Qt's copy]; the three types, load64, store32, store64, the first loop of blake2b_compress, blake2b_set_lastblock, blake2b_increment_counter, blake2b_init0, blake2b_init_param, blake2b_init, blake2b_update, blake2b_final, blake2b [from librsync's copy]); `prover/tests/coins_second_text/format_hash.h` (the two tables, the rotation, G, ROUND, compress, load64, store64, the fields of the state and of the parameter block, that block's first word, the value of the set flag, the loop that writes h out [from Qt's copy]); `scales/policy_tables.py` (from _rotr64 to the end of _compress; the first chaining value [from Qt's copy]); `tests/s8_model.py` (from rotr64 to the end of F; the first chaining value [from Qt's copy]); `verifier/src/vf_earlier_hash_f.h` (G, ROUND, F [from Qt's copy]); `verifier/tests/known_answers/blake2b_f.py` (the two tables, the rotation, G, ROUND, f, the first chaining value, two constants, blake2b (the whole hash with an optional key) [from Qt's copy]).
  - quoted: `blocks/hash/blake2b_f.h` (the comment that prints the head of blake2b-ref.c); `blocks/hash/lanes.cuh` (the comment that prints the head of blake2b-ref.c); `docs/format/format_rules.py` (the comment that prints the head of blake2b-ref.c); `integrations/vllm/integer_model/integer_model/tables.py` (the comment that prints the head of blake2b-ref.c); `lean/Run/Blake2b/Compress.lean` (the comment that prints the head of blake2b-ref.c); `prover/common/blake2b_f.h` (the comment that prints the head of blake2b-ref.c); `prover/second_writing/blake2b_plain.h` (the comment that prints the head of blake2b-ref.c); `prover/tests/coins_second_text/format_hash.h` (the comment that prints the head of blake2b-ref.c); `scales/policy_tables.py` (the comment that prints the head of blake2b-ref.c); `tests/s8_model.py` (the comment that prints the head of blake2b-ref.c); `verifier/src/vf_earlier_hash_f.h` (the comment that prints the head of blake2b-ref.c); `verifier/tests/known_answers/blake2b_f.py` (the comment that prints the head of blake2b-ref.c).
  - values: `lean/Fmt/Blake2bKnown.lean` (one row of eight words); `lean/Run/Blake2b/Spec.lean` (the digest of 1,000 bytes); `lean/Run/Blake2b/Test/Vectors.lean` (counterAnswers (six digests)).
  - `verifier/src/vf_earlier_hash_f.h` (IV, blake2b_sigma, rotr64 [from one of the two copies]), whose comment says: “Taken from blake2b-ref.c: the tables blake2b_IV (called IV here) and blake2b_sigma, the macros G and ROUND, and blake2b_compress without its first loop (called F here).”.
  - further files under `lean/`, which `lean/CREDITS-FILES.tsv` lists.
- From: https://github.com/BLAKE2/BLAKE2: blake2b-ref.c, blake2.h and blake2-impl.h. The text goes back to two copies of these three files. The copies that librsync 2.3.4 bundles (out of Ubuntu's source package of librsync 2.3.4): for `prover/common/blake2b_f.h`, `blocks/hash/blake2b_f.h` and `blocks/hash/lanes.cuh`, and in `prover/second_writing/blake2b_plain.h` for the types, load64, store32, store64, the first loop of blake2b_compress and the eight functions. The copies that Qt 6.4.2 bundles under src/3rdparty/blake2/ (out of Ubuntu's source package qt6-base 6.4.2+dfsg-21.1build5, archive qt6-base_6.4.2+dfsg.orig.tar.xz, SHA-256 4f14b89a9463a540fd84a6718b8519a2297119b454ef547721015a9e9019ae9e; of these copies Qt's file qt_attribution.json says: commit 54f4faa4c16ea34bcd59d16e8da46a64b259fc07): for F, G and ROUND in `verifier/src/vf_earlier_hash_f.h`, for `prover/tests/coins_second_text/format_hash.h`, for the Python files, and in `prover/second_writing/blake2b_plain.h` for the two tables, the rotation, G, ROUND and the rest of blake2b_compress. From which of the two the two tables and the rotation in `verifier/src/vf_earlier_hash_f.h` were taken is not known. SHA-256 sums: blake2b-ref.c e2bf9872a8f0a51711b765936d420a7f8c34797db4d938948c24f2ddbc1dc588 in both; in Qt's bundle blake2.h 389bc87a83cdd9e25569a294d01a3347970d117237a66eee9df8edd6058736a4 and blake2-impl.h bc0ead7f3259a415325fa40ddebb1876f903d5062d888fc5994e8b2d9e616ec4; in librsync's blake2.h adc34cb6ac8045dcf1a59e354771b70e5718c85fe20e0a6572a149d77fbe6026 and blake2-impl.h 8d5813e96210ba4b6e30967ae44c6c8deda8c8687ae0edd86cf3e92d4a0f44cc. The licence text that ships, `licenses/BLAKE2-54f4faa4.COPYING`, is the file COPYING of Qt's bundle.
- Copyright: Copyright 2012, Samuel Neves <sneves@dei.uc.pt>.
- Licence: "the CC0, the OpenSSL Licence, or the Apache Public License 2.0, at your option"; this repository takes CC0 1.0 Universal (SPDX: CC0-1.0; licenses/BLAKE2-54f4faa4.COPYING).

### libb2

- What:
  - values: `lean/Run/Blake2b/KnownThm.lean` (digests in theorems: the first and the last entry of each array); `lean/Run/Blake2b/Spec.lean` (expected digests (#guard): entries 0, 1, 127, 128, 129 and 255 of each array).
  - further files under `lean/`, which `lean/CREDITS-FILES.tsv` lists.
- From: libb2 0.98.1: blake2-kat.h (the arrays blake2b_kat and blake2b_keyed_kat).
- Copyright: Written in 2012 by Samuel Neves <sneves@dei.uc.pt>
- Licence: CC0-1.0 (licenses/BLAKE2-54f4faa4.COPYING).

### blake2b_simd

- What: `blocks/hash/lanes.cuh` (the head, "Public sources of the scheme": leaving b in place), whose comment says: “blake2b_simd's src/avx2.rs has the same scheme (diagonalize, undiagonalize; MIT licence; https://docs.rs/crate/blake2b_simd/latest/source/src/avx2.rs), and so has BLAKE3's SSE and AVX code …”.
- From: https://github.com/oconnor663/blake2_simd, folder blake2b: src/avx2.rs (diagonalize, undiagonalize).
- Copyright: Copyright (c) 2018 Jack O'Connor
- Licence: MIT (licenses/blake2_simd.LICENSE).

### sneves/blake2-avx2

- What: `blocks/hash/lanes.cuh` (the head, "Public sources of the scheme": leaving b in place), whose comment says: “leaving b in place: https://github.com/sneves/blake2-avx2/pull/4; blake2b_simd's src/avx2.rs has the same scheme …”.
- From: https://github.com/sneves/blake2-avx2: pull request 4.
- Copyright: none is given.
- Licence: CC0 1.0 Universal (licenses/blake2-avx2.LICENSE).

## ChaCha20

### chacha-merged.c by D. J. Bernstein

- What:
  - adapted: `blocks/coins/chacha.h` (chacha_quarter_round; BLK_CHACHA_DOUBLE_ROUND (the order of the eight quarter rounds)); `prover/common/coin_chacha.h` (quarter_round; COIN_DOUBLE_ROUND (the order of the eight quarter rounds)); `prover/hash_commitment/hc_chacha.cuh` (hc_quarter_round; HC_DOUBLE_ROUND (the order of the eight quarter rounds)); `prover/hash_commitment/hf_mask_ref.h` (block (the quarter round, the loop of ten double rounds, the final addition of the initial state, how the key's bytes and the constant become words)); `prover/second_writing/chacha_block.h` (quarter_round; chacha_block; chacha_blocks (the round loop)); `prover/tests/coins_second_text/chacha_block.h` (quarter_round; chacha_block; chacha_blocks (the round loop)); `prover/tests/longrow/made_key.h` (second::block (the quarter round, the loop of ten double rounds, the final addition of the initial state, the state's first four words)); `prover/tests/public_weights.py` (quarter_round in derived_seed; the order of the eight quarter rounds); `tools/audit/coins.py` (_block (quarter_round, the loop, the words x0 to x15); f).
  - values: `tools/audit/coins.py` (CONST).
  - `tools/audit/coins.py` (_rotate(), _plus() and their forms for arrays), whose comment says: “The four functions below are the macros ROTATE and PLUS of D. J. Bernstein's chacha-merged.c, each written twice. They follow the copy that OpenSSH 9.6p1 …”.
- From: version 20080118; the copy that OpenSSH 9.6p1 carries as chacha.c: the macros QUARTERROUND, ROTATE and PLUS; sigma; chacha_encrypt_bytes (the copy of the initial state into the working state, the round loop, the additions after the loop); the key setup and the encryption function; the places of input.
- Copyright: none is given.
- Licence: public domain.

### ecrypt-portable.h, a header of the ECRYPT Stream Cipher Project

- What: listed for: `prover/hash_commitment/hf_mask_ref.h`, `prover/tests/longrow/made_key.h`, `tools/audit/coins.py`.
- From: no code file names this header; the macros ROTL32, U8TO32_LITTLE and U32V, as the copy of chacha-merged.c that OpenSSH 9.6p1 carries as chacha.c has them (`tools/audit/coins.py` names them); the header itself: cr.yp.to/snuffle/ecrypt-portable.h.
- Copyright: none is given.

### RFC 8439, "ChaCha20 and Poly1305 for IETF Protocols"

- What:
  - quoted: `prover/hash_commitment/hf_mask_own.cuh` (the comment at RFC8439_LAST); `prover/hiding/coins_own.cuh` (the comment at CoinsRfc8439Vector); `prover/tests/coins_rfc8439_test.cpp` (the comment at Rfc8439Vector).
  - values: `blocks/coins/coins.h` (coins_own_test: the sixteen words of the block); `blocks/coins/tests/coin_values.py` (main: the sixteen words of the block and its inputs); `blocks/coins/tests/coins_test.cpp` (rfc_vector: the sixteen words of the block and its inputs); `prover/hash_commitment/hf_mask_own.cuh` (RFC8439_LAST, RFC8439_STATE); `prover/hash_commitment/tests/hc_chacha_test.cpp` (the quarter round's values, RFC_232, ZERO_BLOCK); `prover/hiding/coins_own.cuh` (COINS_RFC8439); `prover/tests/coins_rfc8439_test.cpp` (the table of published vectors; the quarter round's values); `tools/audit/coins.py` (own_test: the sixteen words of the block and its inputs); `toy/mask_rules.py` (rfc_vector_holds: the sixteen words of the block and its inputs; the key in test_seed; the last four input words in known_inputs); `verifier/tests/known_answers/hiding_coins.py` (main: the block as 64 octets with four input words; the state, the four places and the four results of the quarter round).
  - `blocks/coins/tests/coin_values.py` (quarter, f), whose comment says: “`f` implements ChaCha20's block function from RFC 8439, sections 2.1 to 2.3, and is checked against the RFC's test vector of 2.3.2 before anything is printed.”.
  - `prover/tests/coin_keys_values.py` (quarter, f), whose comment says: “A third implementation of that header's table of keys, besides prover/second_writing/schedule_keys.h, on hashlib's BLAKE2b and ChaCha20's block function as specified in RFC 8439, 2.1 to 2.3.”.
  - `toy/mask_rules.py` (CONSTANTS, ROUNDS, block), whose comment says: “block() is typed from the RFC (RFC 8439, sections 2.1 to 2.3), and the rest is typed from the lines above.”.
  - `verifier/tests/known_answers/hiding_coins.py` (CONSTANTS, ROUNDS, quarter, block, blocks), whose comment says: “It follows RFC 8439, sections 2.1 to 2.3 (the quarter round, the block function).”.
- From: https://www.rfc-editor.org/rfc/rfc8439, June 2018: sections 2.1 to 2.3 (the quarter round, the block function); the test vectors of sections 2.1.1, 2.2.1 and 2.3.2 and of Appendix A.1 (test vectors 1 to 5).
- Copyright: Copyright (c) 2018 IETF Trust and the persons identified as the document authors. All rights reserved.
- Licence: the RFC's copyright notice (licenses/RFC-8439.copyright-notice).

### A file of the Bitcoin Core developers ("Python implementation of ChaCha20 cipher")

- What: listed for: `blocks/coins/tests/coin_values.py`, `prover/hash_commitment/hf_mask_ref.h`, `prover/tests/coin_keys_values.py`, `prover/tests/longrow/made_key.h`, `prover/tests/public_weights.py`, `tools/audit/coins.py`, `toy/mask_rules.py`, `verifier/tests/known_answers/hiding_coins.py`.
- From: no address is given.
- Copyright: Copyright (c) 2022 The Bitcoin Core developers
- Licence: MIT (the MIT text: NOTICE, under ggml).

## SHA-256

### sha256-x86.c by Jeffrey Walton

- What: adapted: `prover/weights_commitment/sha256.h` (blocks (the path for the x86 SHA instructions)).
- From: the copy that btrfs-progs 6.6.3 carries as crypto/sha256-x86.c: sha256_process_x86.
- Copyright: Written and place in public domain by Jeffrey Walton; Based on code from Intel, and by Sean Gulley for the miTLS project.
- Licence: public domain.

### RustCrypto's sha2

- What: `prover/weights_commitment/sha256.h` (blocks: the eight shuffles before and after the loop), whose comment says: “The eight shuffles here are the same operations, with the same constants and in the same order, as in RustCrypto's sha2 (src/sha256/x86.rs; MIT OR Apache-2.0), which is written in Rust.”.
- From: https://github.com/RustCrypto/hashes, folder sha2, release 0.10.9: src/sha256/x86.rs.
- Copyright: Copyright (c) 2006-2009 Graydon Hoare; Copyright (c) 2009-2013 Mozilla Foundation; Copyright (c) 2016 Artyom Pavlov
- Licence: MIT OR Apache-2.0 (the MIT text: NOTICE, under ggml; Apache-2.0: LICENSE).

### DI Management, "Test vectors for SHA-1, SHA-2 and SHA-3"

- What: values: `weight_sums/tests/unit_test.cpp` (test_sha256_published: the published digests).
- From: di-mgt.com.au/sha_testvectors.html.
- Copyright: not known.

### RFC 6234, "US Secure Hash Algorithms (SHA and SHA-based HMAC and HKDF)"

- What:
  - quoted: `weight_sums/weight_sums.cpp` (the comment at the round constants in Sha256::compress; the comment at state_ in Sha256).
  - values: `weight_sums/tests/unit_test.cpp` (test_sha256_published: the digests of "abc", of the message of 56 bytes and of one million times "a").
  - `weight_sums/weight_sums.cpp` (the head and the comments of the class Sha256), whose comment says: “RFC 6234, "US Secure Hash Algorithms", gives the same algorithm with test vectors …”.
- From: https://www.rfc-editor.org/rfc/rfc6234: sections 4.1, 5.1, 6.1 and 6.2; section 8.5, the test driver (TEST1, TEST2_1, TEST3).
- Copyright: Copyright (c) 2011 IETF Trust and the persons identified as the document authors.  All rights reserved.; Copyright (c) 2011 IETF Trust and the persons identified as authors of the code.  All rights reserved.
- Licence: the RFC's copyright notice (licenses/RFC-6234.copyright-notice); for its sample code: three-clause BSD terms (licenses/RFC-6234.sample-code.LICENSE.txt).

RFC 6234's second copyright line and the terms of its sample code (the comment at the head of sha.h in the RFC's section 8.1.1) are listed for `weight_sums/weight_sums.cpp` (Sha256::compress).

## SplitMix64

### splitmix64.c by Sebastiano Vigna

- What:
  - `blocks/chain/tests/rounds_test.cpp` (TestCoins::word), whose comment says: “splitmix64 (Steele, Lea, Flood, 2014); for tests only …”.
  - `blocks/code/tests/code_test.cpp` (word), whose comment says: “SplitMix64 (S. Vigna, splitmix64.c, public domain): test values only.”.
  - `blocks/code/tests/cosets_test.cpp` (word), whose comment says: “SplitMix64 (S. Vigna, splitmix64.c, public domain): test values only.”.
  - `blocks/field/tests/field_test.cpp` (word), whose comment says: “SplitMix64 (S. Vigna, splitmix64.c, public domain): test values only.”.
  - `blocks/frame/tests/wf_frame_test.cpp` (mix), whose comment says: “SplitMix64 (Steele, Lea, Flood, OOPSLA 2014) …”.
  - `blocks/hash/tests/mode_test.cpp` (rnd), whose comment says: “splitmix64 (S. Vigna, https://prng.di.unimi.it/splitmix64.c): test data only …”.
  - `blocks/iop/tensor.h` (merge_entries: the multiplier), whose comment says: “The hash multiplies the index by 2^64 / the golden ratio, rounded down: the same constant as the increment of SplitMix64 (splitmix64.c by Sebastiano Vigna, public domain).”.
  - `blocks/iop/tests/fold_base_test.cpp` (TestCoins::word), whose comment says: “SplitMix64 (Steele, Lea, Flood, OOPSLA 2014). Not for anything secret.”.
  - `blocks/iop/tests/iopp_support.h` (TestCoins::word), whose comment says: “SplitMix64 (Steele, Lea, Flood, OOPSLA 2014).”.
  - `blocks/iop/tests/laid_test.cpp` (word), whose comment says: “SplitMix64 (Steele, Lea, Flood, OOPSLA 2014) …”.
  - `blocks/iop/tests/tensor_test.cpp` (word), whose comment says: “Test data (SplitMix64, Steele, Lea, Flood, OOPSLA 2014).”.
  - `blocks/iop/tests/toy_support.h` (TestRng::word), whose comment says: “A small generator for test data (SplitMix64, Steele, Lea, Flood: "Fast splittable pseudorandom number generators", OOPSLA 2014). Not for anything secret.”.
  - `blocks/tree/tests/tree_test.cpp` (TestCoins::word), whose comment says: “A small generator for tests (splitmix64: Steele, Lea, Flood, "Fast splittable pseudorandom number generators", 2014).”.
  - `integrations/vllm/integer_model/integer_model/synthetic_weights.py` (mix64), whose comment says: “The step is the function next() of S. Vigna's splitmix64.c (https://prng.di.unimi.it/splitmix64.c, public domain).”.
  - `prover/common/common.h` (mix64), whose comment says: “One SplitMix64 step (increment, then output mix): a pseudo-random word from z. Not cryptographic.”.
  - `prover/common/host_common.h` (mix64), whose comment says: “one step of SplitMix64 …”.
  - `prover/compiler_sample.cu` (drawn), whose comment says: “One step of SplitMix64, from a fixed start (THIRD_PARTY.md, "SplitMix64"): the drawn inputs.”.
  - `prover/engine/tests/wf_heavy_test.cu` (Random::word), whose comment says: “splitmix64 (Steele, Lea, Flood, 2014); for tests only …”.
  - `prover/hash_commitment/hc_format.h` (hc_mix64), whose comment says: “public source: splitmix64.c by Sebastiano Vigna, one step …”.
  - `prover/hash_commitment/tests/hf_code_test.cu` (word), whose comment says: “SplitMix64 …”.
  - `prover/prover/fork_memory.h` (k_fork_memory_digest: the output mix, and the first constant as a multiplier), whose comment says: “The output mix of SplitMix64 (THIRD_PARTY.md).”.
  - `prover/prover/forward.cuh` (synth_mix), whose comment says: “one step of SplitMix64 …”.
  - `prover/prover/memory_as_set_up.cuh` (k_memory_digest: the output mix, and the first constant as a multiplier), whose comment says: “The output mix of SplitMix64 (THIRD_PARTY.md).”.
  - `prover/tests/base_kernels_test.cu` (mix), whose comment says: “one step of SplitMix64 (THIRD_PARTY.md, "SplitMix64") …”.
  - `prover/tests/coins_second_text/process_keys.h` (mix64), whose comment says: “One step of SplitMix64 from the state z (splitmix64.c by Sebastiano Vigna).”.
  - `prover/tests/door_test_host.cuh` (word), whose comment says: “A small generator for test data (SplitMix64: Steele, Lea, Flood, "Fast splittable pseudorandom number generators", OOPSLA 2014). Not for anything secret.”.
  - `prover/weights_blocks/made_up_blocks.h` (mix), whose comment says: “Scrambles a 64-bit word with SplitMix64's increment and output mix. Different inputs give different outputs.”.
  - `toy/mask_rules.py` (mix64), whose comment says: “Public source: splitmix64.c by Sebastiano Vigna, prng.di.unimi.it/splitmix64.c (public domain), which has the step with the same three constants.”.
  - `verifier/src/vf_common.h` (mix64), whose comment says: “The SplitMix64 mixer; not cryptographic.”.
  - `weight_sums/tests/oracle.py` (Draws.word), whose comment says: “word() is one step of SplitMix64 (../../THIRD_PARTY.md, "SplitMix64", has the citation).”.
  - further files under `lean/`, which `lean/CREDITS-FILES.tsv` lists.
- From: https://prng.di.unimi.it/splitmix64.c: next().
- Copyright: Written in 2015 by Sebastiano Vigna (vigna@acm.org)
- Licence: public domain, with a permission notice (printed in NOTICE).

## The fields' arithmetic

### plonky2

- What:
  - `blocks/field/gl64.h` (reduce128()), whose comment says: “These are the steps of the function reduce128 in the plonky2 library (Polygon Zero, field/src/goldilocks_field.rs, https://github.com/0xPolygonZero/plonky2, under the MIT or the Apache-2.0 licence) …”.
  - further files under `lean/`, which `lean/CREDITS-FILES.tsv` lists.
- From: https://github.com/0xPolygonZero/plonky2, branch main, of 28 July 2024: field/src/goldilocks_field.rs, function reduce128.
- Copyright: Copyright (c) 2022-2025 The Plonky2 Authors
- Licence: Apache-2.0 or MIT; this repository takes MIT (licenses/plonky2.LICENSE-MIT).

### Plonky3

- What: `blocks/field/gl64.h` (reduce128()), whose comment says: “Plonky3's crate p3-goldilocks (src/goldilocks.rs; MIT or Apache-2.0) has the same steps.”.
- From: https://github.com/Plonky3/Plonky3, crate p3-goldilocks 0.8.0: src/goldilocks.rs.
- Copyright: not known.
- Licence: "MIT OR Apache-2.0"; this repository takes Apache-2.0 (LICENSE).

### Plonky3: the crates p3-whir, p3-sumcheck and p3-security

- What: listed for: `blocks/iop/fold.h`, `blocks/iop/fold_rounds.h`, `blocks/iop/psc.h`.
- From: https://github.com/Plonky3/Plonky3, crates p3-whir 0.7.0 (src/pcs/zk/), p3-sumcheck 0.7.0 (src/zk/) and p3-security 0.8.0 (src/whir/hiding.rs).
- Copyright: not known.
- Licence: "MIT OR Apache-2.0"; this repository takes Apache-2.0 (LICENSE).

### ff

- What: listed for: `prover/hiding/psc_device.cuh`.
- From: https://github.com/zkcrypto/ff, release 0.14.0: src/batch.rs.
- Copyright: Copyright (c) 2017 Sean Bowe
- Licence: "MIT/Apache-2.0"; this repository takes MIT (licenses/ff-0.14.0.LICENSE-MIT).

## sppark

- What:
  - `prover/field/transform_groups_gl64.cuh` (out), whose comment says: “The instructions of this statement are, step for step and in order, those of the reduction in class gl64_t of sppark (Supranational LLC; ff/gl64_t.cuh; Apache-2.0).”.
  - `prover/field/transform_groups_gl64.cuh` (times), whose comment says: “Part of this statement takes the same steps as the product in class gl64_t of sppark (Supranational LLC; ff/gl64_t.cuh; Apache-2.0).”.
- From: https://github.com/supranational/sppark: ff/gl64_t.cuh, class gl64_t (the reduction; the product); ff/mont_t.cuh, class mont_t (the wide product).
- Copyright: Copyright Supranational LLC
- Licence: Apache-2.0 (LICENSE).

## MegEngine

- What:
  - `prover/common/zfold_turned.cuh` (k_zfold_turn), whose comment says: “the 4 x 4 byte transpose by eight __byte_perm calls … is in MegEngine … k_zfold_turn makes the same eight calls, with the same operands and selectors (see NOTICE).”.
  - `prover/hash_commitment/hf_passes.cuh` (turn4), whose comment says: “The same eight __byte_perm calls, with the same operands and selectors, are in MegEngine, dnn/src/cuda/memory_utils.cuh, transpose_int8_4x4_impl … there the second and the third call stand in the other order.”.
  - `prover/hiding/gram_products.cuh` (transpose_bytes), whose comment says: “The same eight __byte_perm calls, with the same operands and selectors, are in MegEngine, dnn/src/cuda/memory_utils.cuh, transpose_int8_4x4_impl … there the second and the third call stand in the other order.”.
- From: https://github.com/MegEngine/MegEngine: dnn/src/cuda/memory_utils.cuh (transpose_int8_4x4_impl), in one copy of that file (SHA-256 cd5b393ce855c150b446333f5a5e7fab97e0a28f1b23fec8abe764e2513b6c2f; its head says "Copyright (c) 2014-2021"). No commit is known for the copy.
- Copyright: Copyright (c) 2014-2023 Megvii Inc. All rights reserved.; Copyright (c) 2014-2021 Megvii Inc. All rights reserved.
- Licence: Apache-2.0 (licenses/MegEngine.LICENSE); licenses/MegEngine.memory_utils.cuh.head.

## NVIDIA CUTLASS

- What:
  - quoted: `prover/weights_blocks/experts_in_one_launch.cuh` (the comment above four_codes: the notice); `prover/weights_blocks/row_pass_fused.cuh` (the comment above lowest_bytes: the notice).
  - `prover/engine/products.cuh` (k_product_row_bytes), whose comment says: “The loop below takes the same steps, in other statements, as the byte transposition in NVIDIA CUTLASS … its five selectors … are those of that transposition.”.
  - `prover/weights_blocks/experts_in_one_launch.cuh` (four_codes), whose comment says: “four_codes() makes the same __byte_perm calls, with the same selectors, as NVIDIA CUTLASS … of that routine's four groups of three calls, the first … and the second …”.
  - `prover/weights_blocks/row_pass_fused.cuh` (lowest_bytes), whose comment says: “The three __byte_perm calls and their selectors are the same as those for the first row in NVIDIA CUTLASS, include/cutlass/transform/thread/transpose.h (int8_t 4x4 transpose), BSD-3-Clause …”.
- From: https://github.com/NVIDIA/cutlass: include/cutlass/transform/thread/transpose.h (int8_t 4x4 transpose; the file's notice), in a copy that came without a repository, a path or a tag (SHA-256 of the copy: 03c2164b7a9942249c2c12a2a60990a4aa5f75cf7d0c2d974e7640b0c2f7cc09; the years at its head are 2017 - 2022).
- Copyright: Copyright (c) 2017 - 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
- Licence: BSD-3-Clause (licenses/CUTLASS.LICENSE.txt); licenses/CUTLASS.transpose.h.notice.

## Code that comments name for an idea, and NOTICE

### Marlin

- What: `prover/weights_blocks/row_pass_fused.cuh` (mma_bytes), whose comment says: “A later comparison found a like form in Marlin's kernel, flashinfer/mma.cuh, SageAttention.”.
- From: https://github.com/IST-DASLab/marlin, as carried in vLLM 0.6.6.post1.
- Copyright: Modified by Neural Magic; Copyright (C) Marlin.2024 Elias Frantar
- Licence: Apache-2.0 (licenses/Marlin.LICENSE; licenses/Marlin.in-vllm-0.6.6.post1.LICENSE).

### flashinfer

- What: `prover/weights_blocks/row_pass_fused.cuh` (mma_bytes), whose comment says: “A later comparison found a like form in Marlin's kernel, flashinfer/mma.cuh, SageAttention.”.
- From: https://github.com/flashinfer-ai/flashinfer, release 0.6.18: include/flashinfer/mma.cuh.
- Copyright: Copyright (c) 2023 by FlashInfer team.
- Licence: Apache-2.0 (licenses/flashinfer-0.6.18.LICENSE).

### SageAttention

- What: `prover/weights_blocks/row_pass_fused.cuh` (mma_bytes), whose comment says: “A later comparison found a like form in Marlin's kernel, flashinfer/mma.cuh, SageAttention.”.
- From: https://github.com/thu-ml/SageAttention: mma.cuh.
- Copyright: Adapted from Flashinfer, https://github.com/flashinfer-ai/flashinfer/blob/v0.1.5/include/flashinfer/mma.cuh; Copyright (c) 2023 by FlashInfer team.; Modifications copyright (c) 2024 by SageAttention team.; Copyright 2024 Jintao Zhang, Haofeng Huang
- Licence: Apache-2.0 (licenses/SageAttention-2.2.0.LICENSE; licenses/SageAttention.mma.cuh.head).

### Apache MXNet

- What: named in: `prover/prover/forward_by_four.cuh`.
- From: https://github.com/apache/mxnet: cuda_vectorization.cuh (CheckAlignment, VectorizedKernelLauncher).
- Copyright: Copyright 2017-2023 The Apache Software Foundation; Copyright 2019-2020 The Apache Software Foundation
- Licence: Apache-2.0 (licenses/MXNet.LICENSE); licenses/MXNet.NOTICE.

### The TFHE library

- What: named in: `prover/field/lazy_gl64.cuh`, `prover/hash_commitment/hf_passes.cuh`, `prover/hiding/gram_products.cuh`.
- From: https://github.com/tfhe/tfhe: src/libtfhe/tgsw-functions.cpp, tgsw.cpp.
- Copyright: Copyright 2016 - Nicolas Gama <nicolas.gama@gmail.com> et al.
- Licence: Apache-2.0 (LICENSE; licenses/TFHE.LICENSE).

## CPython's standard library

### CPython

- What:
  - values: `blocks/coins/tests/run_coins_test.cpp` (main(): a known answer); `blocks/hash/tests/functions_test.cpp` (expected digests); `blocks/hash/tests/mode_vectors.txt` (known answers, made with hashlib); `lean/Fmt/Blake2bKnown.lean` (the expected digests); `lean/OnBytes/Hash/TestBlake2b.lean` (the known answers); `lean/Run/Blake2b/Test/` (the expected digests in KnownAnswers.lean, Salted.lean, Vectors.lean, SpecKnown.lean, OtherOrigin.lean); `prover/tests/coin_keys_values.txt` (known answers, made with hashlib among others); `prover/tests/coins_own_gpu_test.cu` (main(): want[8]); `prover/tests/hash_test.cpp` (expected digests); `prover/tests/hmac_sha256_test.cpp` (the other HMAC values).
  - `integrations/check_proof.py` (class NoRedirects: the chained assignment), whose comment says: “The chained assignment is as in CPython's HTTPRedirectHandler (PSF-2.0).”.
  - `scales/make_calibration_set.py` (a comment on random() and randint), whose comment says: “Python's documentation promises the same sequence in every version for random(), not for randint (The Python Standard Library, "random", "Notes on Reproducibility").”.
- From: https://github.com/python/cpython: hashlib.blake2b, hashlib.sha256 and hmac (their output; files under lean/ name CPython 3.13 and 3.13.16, among others; the other files name no version); Lib/urllib/request.py (HTTPRedirectHandler; 3.12.3 and 3.13.7); Python's documentation of random.
- Copyright: Copyright (c) 2001-2024 Python Software Foundation; All Rights Reserved
- Licence: PYTHON SOFTWARE FOUNDATION LICENSE VERSION 2 (licenses/CPython-3.13.16.LICENSE.txt).

### psutil

- What: listed for: `integrations/vllm/end_to_end/end_to_end.py`, `prover/tests/gate_children.py`.
- From: https://github.com/giampaolo/psutil, release 7.1.3.
- Copyright: Copyright (c) 2009, Giampaolo Rodola'. All rights reserved.; Copyright (c) 2009, Jay Loden, Dave Daeschler, Giampaolo Rodola All rights reserved.
- Licence: BSD-3-Clause (licenses/psutil-7.1.3.LICENSE).

## Linux, the C library and POSIX

### the Linux kernel's user-space headers

- What: values: `tools/audit/kept.py` (PR_SET_DUMPABLE, MCL_CURRENT, MCL_FUTURE as numbers); `tools/confine.py` (CLONE_NEWNS, the MS_* flags, MOUNT_SETATTR, AT_FDCWD, AT_RECURSIVE, MOUNT_ATTR_RDONLY; the layout of struct mount_attr; Landlock's and seccomp's numbers and layouts (system-call numbers, access rights, PR_*, SECCOMP_*, AUDIT_ARCH_*, BPF_* codes, struct sock_filter, sock_fprog, landlock_ruleset_attr, landlock_path_beneath_attr)); `tools/confine_test.py` (some of the same numbers; bit 21 of a process's capabilities (CAP_SYS_ADMIN)); `tools/lean-verify/lean_verify_test.sh` (36, PR_SET_CHILD_SUBREAPER).
- From: sched.h, mount.h and the manual page mount_setattr(2) for the mount values; for the other values no header is named.
- Copyright: not known.

### runc

- What: named in: `tools/confine.py`.
- From: no file or version of runc is named.
- Copyright: not known.

### The Linux kernel's sample samples/landlock/sandboxer.c

- What: `tools/confine.py` (main(): the Landlock calls), whose comment says: “The calls below take the same steps as the Linux kernel's sample samples/landlock/sandboxer.c (BSD-3-Clause).”.
- From: the Linux kernel, samples/landlock/sandboxer.c, in a copy that came without a repository or a version.
- Copyright: Copyright © 2017-2020 Mickaël Salaün <mic@digikod.net>; Copyright © 2020 ANSSI.
- Licence: BSD-3-Clause (licenses/linux-sample-sandboxer.LICENSE).

## vLLM

- What:
  - adapted: `integrations/vllm/integer_model/tests/test_vllm_model_class.py` (test_the_chosen_token_survives_sampling_settings: the statements for the repetition penalty).
  - quoted: `prover/prover/forward.cuh` (the comment above k_epi_gate_up); `prover/prover/forward_moe.cuh` (the comment above k_moe_prep).
  - values: `integrations/proof_middleware.py` (send_error(): the four fields of the error body; the type words BadRequestError, NotFoundError, InternalServerError); `integrations/tests/recorded_vllm/chat.json` (the whole file (a recorded response body)); `integrations/tests/recorded_vllm/chat_stream.txt` (the whole file (a recorded response body)); `integrations/tests/recorded_vllm/completion.json` (the whole file (a recorded response body)); `integrations/tests/recorded_vllm/completion_stream.txt` (the whole file (a recorded response body)).
  - `integrations/proof_middleware.py` (class ProofMiddleware: the pure ASGI middleware pattern), whose comment says: “Follows the pure ASGI middleware pattern of Starlette's documentation (BSD-3-Clause) and of vLLM's middleware (Apache-2.0).”.
  - `integrations/proof_middleware.py` (ProofMiddleware.__call__: the path without a proxy's root path), whose comment says: “The path without a proxy's root path, as in vLLM's authenticate.py (Apache-2.0).”.
  - `integrations/proof_middleware.py` (check_api_key() of ProofMiddleware: the split of the Authorization header's value and the comparison of its scheme), whose comment says: “The value is split, and its scheme compared, as in vLLM's authenticate.py (Apache-2.0).”.
  - `integrations/vllm/end_to_end/end_to_end.py` (Programs.start(), stop_all(), reap(), wait_for_groups(): ending a process group), whose comment says: “The steps are those of vLLM's own tests (Apache-2.0): tests/utils.py, which also looks at /proc for what is left of the group, and tests/v1/kv_connector/rocm_pd_accuracy_utils.py.”.
  - `integrations/vllm/end_to_end/end_to_end.py` (start_vllm(): the loop that waits for /health), whose comment says: “The loop has the structure of the one in vLLM's own tests (Apache-2.0): tests/utils.py …”.
  - `integrations/vllm/integer_model/integer_model/noise.py` (table(), _g(), noise()), whose comment says: “Their sources are vllm/v1/worker/gpu/sample/gumbel.py (Apache-2.0) and triton/language/random.py of Triton 3.7.1 (MIT).”.
  - `integrations/vllm/integer_model/integer_model_vllm/__init__.py` (register()), whose comment says: “Source: the example in "Plugin System" in vLLM's documentation (vLLM: Apache License 2.0).”.
  - `integrations/vllm/integer_model/integer_model_vllm/loader.py` (IntegerModelLoader: the class's form; the methods' arguments), whose comment says: “Source: the class has the form of the example in the docstring of register_model_loader (vllm/model_executor/model_loader/__init__.py in vLLM 0.29.0, Apache License 2.0), and its methods have …”.
  - `integrations/vllm/integer_model/integer_model_vllm/model.py` (IntegerModelForCausalLM: the names and arguments of __init__, embed_input_ids, forward, compute_logits, load_weights), whose comment says: “The names and arguments of __init__, embed_input_ids, forward, compute_logits and load_weights are vLLM's.”.
  - `integrations/vllm/integer_model/tests/test_vllm_model_class.py` (test_the_chosen_token_survives_sampling_settings: the order of the steps), whose comment says: “The order of the steps (the bias, the penalties, the temperature) is that of sampler.py in the same directory.”.
  - `integrations/vllm/integer_model/tests/vllm_tools.py` (shut_down(): the calls), whose comment says: “The two calls are internals of vLLM 0.29.0, and vLLM's tests shut an engine down with them (Apache License 2.0).”.
  - `prover/prover/weight_stream.cuh` (the comment at WeightStream's rooms that are lent), whose comment says: “Public source: the same between workloads is described by vLLM's documentation of its Sleep Mode.”.
  - `prover/weights_blocks/experts_in_one_launch.cuh` (the file's head), whose comment says: “vLLM's fused_moe kernel likewise reads its padded token count from GPU memory.”.
- From: https://github.com/vllm-project/vllm, release 0.29.0: vllm/v1/worker/gpu/sample/penalties.py; the example on the page "Plugin System" of vLLM's documentation (docs.vllm.ai/en/latest/design/plugin_system/, as of August 2026); the example in the docstring of register_model_loader (vllm/model_executor/model_loader/__init__.py); BaseModelLoader (vllm/model_executor/model_loader/base_loader.py); vllm/model_executor/models/interfaces_base.py; the page "Basic Model" (docs.vllm.ai/en/latest/contributing/model/basic/, as of August 2026); vllm/entrypoints/serve/middleware/authenticate.py (verify_token); vllm/entrypoints/serve/middleware/x_request_id.py; vllm/entrypoints/serve/engine/protocol.py; vllm/entrypoints/serve/exception_handling/error_response.py; vllm/v1/worker/gpu/sample/gumbel.py; tests/utils.py (_wait_for_server); vllm/v1/worker/cpu_worker.py (init_cpu_memory_env); vllm/v1/worker/gpu/sample/sampler.py; vllm/v1/sample/sampler.py; vllm/model_executor/layers/utils.py; vllm/v1/engine/input_processor.py; vllm/tokenizers/hf.py; output of vLLM 0.29.0's routes around a fake engine. tests/v1/kv_connector/rocm_pd_accuracy_utils.py. vLLM's API documentation, version 0.11.2: SiluAndMul; version 0.10.1.1: moe_align_block_size(). vLLM's documentation of its Sleep Mode; vLLM's fused_moe kernel.
- Copyright: SPDX-FileCopyrightText: Copyright contributors to the vLLM project
- Licence: Apache-2.0 (LICENSE; licenses/vllm-0.29.0.LICENSE).

### torchao

- What: `integrations/vllm/integer_model/integer_model/model.py` (_rotary), whose comment says: “Sources: NOTICE, "torchao" (apply_rotary_emb has the view-rotate-stack form) and "Hugging Face transformers" (neighbouring entries form a pair).”; named in: `integrations/vllm/integer_model/README.md`.
- From: https://github.com/pytorch/ao, release 0.10.0: torchao/_models/llama/model.py (apply_rotary_emb).
- Copyright: Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.; Copyright 2023 Meta
- Licence: BSD-3-Clause (licenses/torchao-0.10.0.LICENSE).

### vllm-gguf-plugin

- What: `integrations/vllm/integer_model/integer_model_vllm/__init__.py` (register(): a loader registered from the entry point), whose comment says: “The idea of registering a loader from this entry point is that of vllm-gguf-plugin 0.0.5 (https://github.com/vllm-project/vllm-gguf-plugin, Apache License 2.0).”.
- From: https://github.com/vllm-project/vllm-gguf-plugin, release 0.0.5.
- Copyright: SPDX-FileCopyrightText: Copyright contributors to the vLLM project
- Licence: Apache-2.0 (LICENSE).

### PyTorch

- What:
  - quoted: `scales/float_forward.py` (forward: a comment on torch.max).
- From: https://github.com/pytorch/pytorch; the quoted words: the documentation of torch.max, torch 2.9.
- Copyright: Copyright (c) 2016-     Facebook, Inc            (Adam Paszke); Copyright (c) 2014-     Facebook, Inc            (Soumith Chintala); Copyright (c) 2011-2014 Idiap Research Institute (Ronan Collobert); Copyright (c) 2012-2014 Deepmind Technologies    (Koray Kavukcuoglu); Copyright (c) 2011-2012 NEC Laboratories America (Koray Kavukcuoglu); Copyright (c) 2011-2013 NYU                      (Clement Farabet); Copyright (c) 2006-2010 NEC Laboratories America (Ronan Collobert, Leon Bottou, Iain Melvin, Jason Weston); Copyright (c) 2006      Idiap Research Institute (Samy Bengio); Copyright (c) 2001-2004 Idiap Research Institute (Ronan Collobert, Samy Bengio, Johnny Mariethoz)
- Licence: BSD-style (licenses/torch-2.13.0.LICENSE).

### gpt-fast

- What: `integrations/vllm/integer_model/integer_model/model.py` (_rotary), of which `integrations/vllm/integer_model/README.md` says: “_rotary in integer_model/model.py is like gpt-fast's model.py in form: both view the last axis as neighbouring pairs, rotate each by the rotary angle and stack them into one axis.”.
- From: https://github.com/pytorch-labs/gpt-fast: model.py (the rotary function).
- Copyright: Copyright 2023 Meta
- Licence: BSD-3-Clause (licenses/gpt-fast.LICENSE).

### Starlette

- What:
  - `integrations/proof_middleware.py` (class ProofMiddleware: the pure ASGI middleware pattern), whose comment says: “Follows the pure ASGI middleware pattern of Starlette's documentation (BSD-3-Clause) and of vLLM's middleware (Apache-2.0).”.
- From: https://github.com/Kludex/starlette, release 1.6.0: the examples under "Pure ASGI Middleware" on the page "Middleware" of Starlette's documentation (https://starlette.dev/middleware/, as of August 2026).
- Copyright: Copyright © 2018, [Encode OSS Ltd](https://www.encode.io/). All rights reserved.
- Licence: BSD-3-Clause (licenses/starlette-1.6.0.LICENSE.md).

### Triton

- What:
  - values: `integrations/vllm/integer_model/integer_model/noise.py` (the comment above _FLOAT32_SCALE: the float32 4.6566127342e-10).
  - `integrations/vllm/integer_model/integer_model/noise.py` (table(), slots(), _g(), noise()), whose comment says: “Their sources are vllm/v1/worker/gpu/sample/gumbel.py (Apache-2.0) and triton/language/random.py of Triton 3.7.1 (MIT).”.
- From: https://github.com/triton-lang/triton, release 3.7.1: triton/language/random.py.
- Copyright: Copyright 2018-2020 Philippe Tillet; Copyright 2020-2022 OpenAI
- Licence: MIT (licenses/triton-3.7.1.LICENSE; the MIT text: NOTICE, under ggml).

### Random123

- What:
  - values: `integrations/vllm/integer_model/tests/test_seeded_score.py` (EXPECTED_VALUES: three known answers of Philox4x32-10).
- From: https://github.com/DEShawResearch/random123, release 1.14.0: tests/kat_vectors.
- Copyright: Copyright 2010-2012, D. E. Shaw Research. All rights reserved.; Copyright 2010-2011, D. E. Shaw Research.
- Licence: BSD-3-Clause (licenses/Random123-1.14.0.LICENSE).

### libcu++ (libcudacxx) in CCCL

- What: listed for: `integrations/vllm/integer_model/integer_model/noise.py`, `integrations/vllm/integer_model/tests/test_seeded_score.py`; named in: `integrations/vllm/integer_model/README.md`.
- From: https://github.com/NVIDIA/cccl, release 3.3.2.
- Copyright: SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
- Licence: Apache-2.0 WITH LLVM-exception (LICENSE; licenses/libcudacxx-CCCL-3.3.2.LICENSE.TXT).

## uvicorn

- What:
  - `integrations/proof_middleware.py` (read_body()), whose comment says: “Follows read_body in uvicorn's documentation (BSD-3-Clause).”.
  - `integrations/tests/test_proof_middleware.py` (a comment on what a web server does once a client has left; the logger's name), whose comment says: “The web server reports the disconnect, and takes what is sent all the same, as uvicorn does.”.
- From: https://github.com/Kludex/uvicorn, release 0.53.0; the example read_body under "Reading the request body" in uvicorn's documentation, of which no version is named.
- Copyright: Copyright © 2017-present, [Encode OSS Ltd](https://www.encode.io/). All rights reserved.
- Licence: BSD-3-Clause (licenses/uvicorn-0.53.0.LICENSE.md).

## Lean packages and the Lean toolchain

### Lean

- What: `lean/Johnson/Unfolds/MergeSort.lean`, `lean/Johnson/Unfolds/SliceFoldl.lean` and `lean/Johnson/Unfolds/SplitOn.lean`, adapted from Lean's core library; further Lean files: NOTICE names them, and `lean/CREDITS.md`, "5. Lean's own source", lists them.
- From: https://github.com/leanprover/lean4, the toolchain leanprover/lean4:v4.29.1: src/lean/Init/Data/List/Sort/Basic.lean; src/lean/Init/Data/String/Slice.lean; src/lean/Init/Data/String/Legacy.lean.
- Copyright: Copyright (c) 2020 Microsoft Corporation. All rights reserved.; Copyright (c) 2019 Microsoft Corporation. All rights reserved.; Copyright (c) 2022 Microsoft Corporation. All rights reserved.; Copyright (c) 2016 Microsoft Corporation. All rights reserved.; Copyright (c) 2024 Lean FRO. All rights reserved.; Copyright (c) 2025 Lean FRO, LLC. All rights reserved.
- Licence: Apache-2.0 (LICENSE).

### Mathlib

- What: the 13 Lean files under `lean/ThirdParty/MathlibBackport/`, adapted; further Lean files: `lean/CREDITS.md`, "2. Lean, Mathlib and the other packages", lists them.
- From: https://github.com/leanprover-community/mathlib4, v4.34.0, commit 5ed2965256430c3649e86755f9576b54eca72435 (the 13 files); revision 5e932f97dd25535344f80f9dd8da3aab83df0fe6 (the pinned package).
- Copyright: Copyright (c) 2024 Daniel Weber. All rights reserved.; Copyright (c) 2014 Parikshit Khanna. All rights reserved.; Copyright (c) 2018 Kenny Lau; Copyright (c) 2017 Johannes Hölzl. All rights reserved.; Copyright (c) 2018 Chris Hughes. All rights reserved.; Copyright (c) 2020 Anne Baanen. All rights reserved.; Copyright (c) 2021 Johan Commelin. All rights reserved.; Copyright (c) 2021 Kim Morrison. All rights reserved.; Copyright (c) 2024 Antoine Chambert-Loir, María Inés de Frutos-Fernández. All rights reserved.; Copyright (c) 2024 Fangming Li. All rights reserved.; Copyright (c) 2025 Andrew Yang. All rights reserved.; Copyright (c) 2026 Yongle Hu. All rights reserved.
- Licence: Apache-2.0 (LICENSE); for the 13 files: Apache-2.0 (licenses/Mathlib-v4.34.0.LICENSE; the same text is lean/ThirdParty/MathlibBackport/LICENSE).

### batteries

- What: named in: `lean/Johnson/Desc/First/TailClaims/LayAtTest.lean`; see `lean/CREDITS.md`, "2. Lean, Mathlib and the other packages".
- From: https://github.com/leanprover-community/batteries, revision 756e3321fd3b02a85ffda19fef789916223e578c.
- Copyright: Copyright (c) 2016 Microsoft Corporation
- Licence: Apache-2.0 (LICENSE).

## ArkLib

- What: the 126 Lean files under `lean/ThirdParty/ArkLib/`, adapted, with `lean/ThirdParty/ArkLib/AUTHORS` and `LICENSE`; `lean/ThirdParty/LICENSE`; entries of ArkLib's list of references and quoted words in `lean/ThirdParty/README.md`; quoted words in `lean/ThirdParty/scripts/compare_with_sources.py`; `lean/Cb/Bridge/CodeListRSFromLines.lean` and `lean/Cb/Bridge/CodeListRSProved.lean`, whose heads say that they “do not count as written independently of” a file of ArkLib that is not in `lean/ThirdParty/ArkLib/` (ArkLib/Data/CodingTheory/ReedSolomon/MutualCorrelatedAgreement/Johnson/Probability.lean); further Lean files: `lean/CREDITS.md`, "3. ArkLib", and `lean/CREDITS-FILES.tsv` list them file by file.
- From: https://github.com/Verified-zkEVM/ArkLib, from a copy marked as commit 2817d8c7d2f28d706b17f8de0f885880d0b2eb13; file by file: `lean/ThirdParty/TAKEN-OVER.tsv`.
- Copyright: Copyright (c) 2026 ArkLib Contributors. All rights reserved.; Copyright (c) 2024-2026 ArkLib Contributors. All rights reserved.; Copyright (c) 2024-2025 ArkLib Contributors; Copyright ArkLib Contributors
- Licence: Apache-2.0 (licenses/ArkLib-2817d8c7.LICENSE; the same text is lean/ThirdParty/ArkLib/LICENSE).

### kz99/rs-ld-mca

- What: adapted: `lean/ThirdParty/ArkLib/Data/CodingTheory/HiddenDerivative/Interpolation/Local/ConstraintMap.lean` (parts of the file); `lean/ThirdParty/ArkLib/Data/MvPolynomial/WeightedOrder.lean` (parts of the file); `lean/ThirdParty/ArkLib/Data/Polynomial/Differential/Types.lean` (the definitions).
- From: named in ArkLib's own comments in these three files: "adapted, with permission, from" "Kai Zhe Zheng's kz99/rs-ld-mca" (the third: "kz99/rs-ld-mca").
- Copyright: none is given.

### CompPoly

- What: `lean/ThirdParty/ArkLibShim/BivariateDegreeX.lean`, whose head says: "Stands in for two names of the package CompPoly".
- From: https://github.com/Verified-zkEVM/CompPoly, module CompPoly.ToMathlib.Polynomial.BivariateDegree.
- Copyright: Copyright 2025 Demerzel Solutions Ltd (t/a Nethermind)
- Licence: Apache-2.0 (LICENSE).

## VCV-io

- What: Lean files whose comments name it: `lean/CREDITS.md`, "4. VCV-io", lists them.
- From: https://github.com/Verified-zkEVM/VCV-io: LatticeCrypto/MLDSA/Concrete/Encoding.lean (lines 448 to 450 and 306 to 309); VCVio/CryptoFoundations/MerkleTree/Addressed/NatIndexed.lean; ProgramLogic/Relational/SimulateQ/UntilBad.lean; QueryTracking/RandomOracle/EagerTable.lean; DependentTable.lean; ToMathlib/MeasureTheory/Measure/UniformTable.lean.
- Copyright: The authors of each source file of VCV-io are named in its header; Copyright (c) 2026 Quang Dao (LatticeCrypto/MLDSA/Concrete/Encoding.lean)
- Licence: Apache-2.0 (LICENSE).

## proximity-prize

- What: `lean/ThirdParty/ArkLib/Data/Polynomial/FractionFieldResultant.lean`, in ArkLib's own comment (`lean/CREDITS.md`, "3. ArkLib", names it).
- From: https://github.com/proximity-prize/proximity-prize, commit cc822c5c; ProximityPrize/SubmissionLower/BCHKSConcreteGoodSpecialization.lean, lines 272–316, at a commit that begins 19bc7d3e (credited in ArkLib's own comment in `lean/ThirdParty/ArkLib/Data/Polynomial/FractionFieldResultant.lean`).
- Copyright: Copyright (c) 2026 Proximity Prize Contributors. All rights reserved.
- Licence: Apache-2.0 (LICENSE).

## succinctlabs/veil-formal-verification

- What: adapted: `lean/STATEMENTS.html`; `lean/scripts/statements-html/front/setting.txt`; `lean/scripts/statements-html/shell/page.css`; `lean/scripts/statements-html/shell/page.js`; `lean/scripts/statements-html/shell/page_template.html`.
- From: succinctlabs/veil-formal-verification: frontend/ and README.
- Copyright: Copyright 2026 Succinct Labs
- Licence: Apache-2.0 (LICENSE).

## Fonts in the PDF under paper/

### Latin Modern fonts

- What: embedded: `paper/paper.pdf` (subsets of 35 fonts).
- From: https://www.gust.org.pl/projects/e-foundry/latin-modern: LMMathExtension10-Regular, LMMathItalic5-Regular, LMMathItalic6-Regular, LMMathItalic7-Regular, LMMathItalic8-Regular, LMMathItalic9-Regular, LMMathItalic10-Regular, LMMathSymbols5-Regular, LMMathSymbols6-Regular, LMMathSymbols7-Regular, LMMathSymbols8-Regular, LMMathSymbols9-Regular, LMMathSymbols10-Regular, LMMono8-Regular, LMMono9-Regular, LMMono10-Regular, LMMono10-Italic, LMRoman5-Regular, LMRoman6-Regular, LMRoman6-Bold, LMRoman7-Regular, LMRoman7-Bold, LMRoman8-Regular, LMRoman8-Bold, LMRoman8-Italic, LMRoman9-Regular, LMRoman9-Bold, LMRoman9-Italic, LMRoman10-Regular, LMRoman10-Bold, LMRoman10-Italic, LMRoman12-Bold, LMSans8-Regular, LMSans9-Regular, LMSans10-Regular.
- Copyright: Copyright 2003--2009 by B. Jackowski and J.M. Nowacki (on behalf of TeX USERS GROUPS).
- Licence: GUST Font License (licenses/GUST-FONT-LICENSE.txt). It gives the conditions of the LaTeX Project Public License, version 1.3c or later (licenses/LPPL-1.3c.txt).

### MSBM10, a font of the American Mathematical Society

- What: embedded: `paper/paper.pdf` (a subset of this font).
- From: http://www.ams.org, out of TeX Live's package amsfonts: MSBM10, version 003.002 by its own head.
- Copyright: Copyright (c) 1997, 2009 American Mathematical Society (<http://www.ams.org>), with Reserved Font Name MSBM10.
- Licence: SIL Open Font License, Version 1.1 (licenses/OFL-1.1.txt).
