# Sequential Data Generation

## Overview
Two generative models trained on a 16-sentence AI/ML dataset to produce new text sequences.

---

## Files
```
component1_lstm.py         # Component I  — Character-level LSTM
component2_transformer.py  # Component II — Word-level Transformer
README.md
```

## Requirements
```bash
pip install torch numpy
```

---

## Component I — LSTM

Character-level model. Predicts the next character from the previous 40.

**Architecture:** `Embedding(64) → LSTM×2(256) → Linear → Softmax`

| Param | Value |
|---|---|
| Sequence length | 40 chars |
| Hidden dim | 256 |
| Epochs | 100 |
| Optimizer | Adam + StepLR |

**Run:**
```bash
python component1_lstm.py
```

**Sample output:**
```
Seed: "deep learning"
→ deep learning models improve sequence learning and generative
  models create new samples from learned patterns...
```

---

## Component II — Transformer

Word-level model with positional encoding and causal attention mask.

**Architecture:** `Embedding(128) → PosEncoding → TransformerEncoder×3 → Linear`

| Param | Value |
|---|---|
| Sequence length | 10 words |
| d_model / heads | 128 / 4 |
| Epochs | 150 |
| Optimizer | Adam + CosineAnnealing |

**Run:**
```bash
python component2_transformer.py
```

**Sample output:**
```
Seed: ['neural', 'networks']
→ neural networks simulate human brain structures optimization
  algorithms improve learning efficiency...
```

---

## Comparison

| | LSTM | Transformer |
|---|---|---|
| Tokenization | Character | Word |
| Context | Hidden state | Attention window |
| Vocab size | ~40 | ~90 |

## Tips
- Lower `temperature` (0.6) → more coherent output
- Higher `temperature` (1.0) → more creative output
- Scripts auto-detect GPU; CPU works fine on this dataset
