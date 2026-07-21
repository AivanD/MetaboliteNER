#!/usr/bin/env python3
"""
Inference script for TABoLiSTM metabolite NER model.

Loads a trained TABoLiSTM model and finds metabolite entities in text.

Usage:
    # Single sentence:
    python infer.py -t "Glucose and lactate levels were measured."

    # Process a file (one sentence per line):
    python infer.py -f path/to/sentences.txt

    # Use a specific model:
    python infer.py -t "Glucose metabolism" -j TrainedModels/tabolistem_chemtok50.json -m TrainedModels/best_chemtok50_model.pth

    # Batch process with JSON output:
    python infer.py -t "Glucose and lactate" --json
"""

import argparse
import json
import sys
import os

import tabolistem_model as tabm

def load_model(json_path, model_path):
    """Load a trained TABoLiSTM model."""
    tm = tabm.TaboListem()
    tm.load(json_path, model_path)
    return tm


def find_metabolites(tm, text, use_argmax=True):
    """
    Find metabolite entities in a single sentence.

    Args:
        tm: Loaded TaboListem model
        text: Input sentence string
        use_argmax: If True, use argmax BIOES decoding (recommended).
                    If False, use threshold-based decoding.

    Returns:
        List of (start_idx, end_idx, metabolite_name) tuples
    """
    if use_argmax:
        return tm.process_argmax(text)
    else:
        return tm.process(text, threshold=0.5)


def format_results(text, entities):
    """Format entities as a human-readable string."""
    if not entities:
        return "  (no metabolites found)"
    lines = []
    for start, end, name in entities:
        context_before = text[max(0, start-20):start]
        context_after = text[end:min(len(text), end+20)]
        lines.append(f"  '{name}' (chars {start}-{end})")
    return "\n".join(lines)


def main():
    # Load model
    tm = load_model("TrainedModels/tabolistem_biobert50.json", "TrainedModels/best_biobert50_model.pth")
    print("Model loaded.", file=sys.stderr)

    # Collect input sentences
    sentences = [
        "The LIPO window carries information particularly on lipoprotein lipids and albumin, whereas the LMWM window contains signals from smaller metabolites such as creatinine and glucose (; ).",
        "DKD was associated with elevated triglycerides, lower HDL cholesterol and decreased albumin in the LIPO window.",
        "Other studies have also reported the connection between albuminuria and triglycerides (; ; ), but the exact role of HDL metabolism remains unclear.",
        "Serum creatinine and urea are two waste products that are normally excreted by the kidneys and, accordingly, the LMWM window revealed elevated values for the macroalbuminuric group, although none of the patients had end-stage renal disease.",
        "Our results from the metabonomic analysis were similar: the SOM regions with patients that have a detectable loss in kidney function (i.e., elevated creatinine and urea, decreased serum albumin) overlapped with insulin resistance and related problems in glucose metabolism (dyslipidemia, high insulin dose, high HbA_1c , elevated lactate and acetate and high fasting glucose) (; ; ; ).",
    ]

    # Process all sentences
    results = []
    for i, text in enumerate(sentences):
        entities = find_metabolites(tm, text, use_argmax=True)
        results.append({
            "index": i,
            "text": text,
            "entities": [
                {"start": s, "end": e, "metabolite": m}
                for s, e, m in entities
            ]
        })

    # Output
    for r in results:
        print(f"\nSentence {r['index'] + 1}: {r['text']}")
        print(f"Metabolites ({len(r['entities'])} found, argmax BIOES):")
        if not r['entities']:
            print("  (no metabolites found)")
        else:
            for ent in r['entities']:
                print(f"  '{ent['metabolite']}' (chars {ent['start']}-{ent['end']})")


if __name__ == '__main__':
    main()
