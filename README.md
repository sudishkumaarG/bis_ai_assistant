# BIS AI Assistant

## Problem
Small businesses struggle to identify relevant BIS standards from large documents.

## Solution
- Hybrid retrieval using FAISS + BM25
- Intelligent ranking with domain rules
- Fast and accurate recommendations

## Results
- Hit@3: 100%
- MRR@5: 0.93
- Latency: 0.06 sec

## How to Run
```bash
python app.py --input data/public_test_set.json --output output.json
python eval_script.py --results output.json