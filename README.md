# 📚 Sequence Labeling System for English Phrase Chunking

This project implements a sequence labeling pipeline using four different methods:

1. **Emission-based tagging**  
2. **Viterbi decoding with HMM parameters**  
3. **Top-k Viterbi decoding (k = 4)**  
4. **Alternative Design** – Structured Perceptron


## 🧪 Requirements

- Python 3.7+
- No external libraries required (pure Python implementation)


## 🚀 How to Run

1. **Install dependencies** (if any):

```bash
pip install -r requirements.txt
```

Ensure data is placed correctly:
- Training data in: EN/train
- Input test file in: EN/dev.in or EN/test.in

Then run:

```bash
# For development data
python src/main.py dev.in

# For test data
python src/main.py test.in
```

Generated Output Files:
- EN/dev.p1.out — Emission-only model predictions
- EN/dev.p2.out — Viterbi predictions using HMM
- EN/dev.p3.out — 4th best Viterbi output
- EN/dev.p4.out or EN/test.p4.out — Structured Perceptron predictions


## ✅ Evaluation

To evaluate your output using the official script:

1. Ensure you're in the project root directory.
2. Run:

```bash
python EvalScript/evalResult.py EN/{reference_file}.out EN/{your_output_file}.out

#Example to run the test output
python EvalScript/evalResult.py EN/test.out EN/test.p4.out
```

## 📂 Directory Structure

```
project/
├── src/
│   ├── main.py
│   ├── emission.py
│   ├── transition.py
│   ├── viterbi.py
│   ├── viterbi_top_k.py
│   ├── structured_perceptron.py
│   └── utils.py
├── EN/
│   ├── train/         # Tagged training data
│   ├── dev.in         # Input sentences for dev
│   ├── dev.out        # Reference output (for eval)
│   ├── test.in        # Input sentences for test
│   └── *.out          # Output files
├── EvalScript/
│   └── evalResult.py  # Evaluation script
└── README.md
```