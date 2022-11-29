#!/usr/bin/env bash

set -e

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=5

python src/evaluate_seq2seq_qa.py \
    --model-name="google/flan-t5-large" \
    --dataset-name="Rexhaif/mintaka-qa-en" \
    --dataset-split="test" \
    --question-column="question" \
    --answer-column="answer" \
    --batch-size=16 \
    --use-8bit \
    --max-new-tokens=128 \
    --prefix="Q: " \
    --output-file="./results/flan-t5-large.csv"