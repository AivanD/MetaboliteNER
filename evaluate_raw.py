import tabolistem_model as tabm
import pandas as pd
import torch
import numpy as np

def load_data(text_path, annot_path):
    corpus_df = pd.read_csv(text_path, sep='\t', names=['corpus','section','text'], encoding='utf-8-sig')
    annot_df = pd.read_csv(annot_path, encoding='utf-8-sig', sep='\t', names=['corpus','section','start','end','metabolite'])
    sentences = {}
    for _, row in corpus_df.iterrows():
        sentences[(row['corpus'], row['section'])] = row['text']
    gold = {}
    for key, group in annot_df.groupby(['corpus','section']):
        gold[key] = {(int(r['start']), int(r['end'])) for _, r in group.iterrows()}
    return sentences, gold

def compute_f1(gold_ents, pred_ents):
    tp = len(gold_ents & pred_ents)
    fp = len(pred_ents - gold_ents)
    fn = len(gold_ents - pred_ents)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, prec, rec, f1

def evaluate_raw(json_path, model_path, text_path, annot_path, use_argmax=True, threshold=1.0):
    print("Loading model...")
    tm = tabm.TaboListem()
    tm.load(json_path, model_path)

    print("Loading data...")
    sentences, gold = load_data(text_path, annot_path)
    all_keys = sorted(sentences.keys())
    texts = [sentences[k] for k in all_keys]

    if use_argmax:
        print(f"Evaluating {len(texts)} sentences (raw, argmax BIOES decoding)...")
    else:
        print(f"Evaluating {len(texts)} sentences (raw, threshold={threshold})...")

    tm.model.eval()
    total_tp = total_fp = total_fn = 0

    for idx, key in enumerate(all_keys):
        text = texts[idx]
        if not text:
            total_fn += len(gold.get(key, set()))
            continue

        seq = tm._str_to_seq(text)
        if len(seq["tokens"]) == 0:
            total_fn += len(gold.get(key, set()))
            continue

        tm._prepare_seqs([seq], verbose=False, save_path=None)
        tx_id, att_mask, tx_ni = tm._seq_to_tensors(seq)

        with torch.no_grad():
            outputs = tm.model(tx_id, att_mask, tx_ni)[0]
        outputs = outputs.cpu().numpy()
        seq["tagfeat"] = outputs[1:-1]

        if use_argmax:
            raw_ents = tm.decode_bioes(seq)
        else:
            raw_ents = tm.score_to_ent(seq, threshold=threshold)
        pred_ents = set(raw_ents.keys())
        gold_ents = gold.get(key, set())

        tp, fp, fn, _, _, _ = compute_f1(gold_ents, pred_ents)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        if (idx + 1) % 1000 == 0:
            print(f"  Processed {idx+1}/{len(all_keys)}...")

    prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    decoding = "argmax" if use_argmax else f"threshold={threshold}"
    print(f"\n{'='*50}")
    print(f"Raw model evaluation (no post-processing)")
    print(f"Decoding: {decoding}")
    print(f"{'='*50}")
    print(f"Gold entities:     {total_tp + total_fn}")
    print(f"Predicted:         {total_tp + total_fp}")
    print(f"TP:                {total_tp}")
    print(f"FP:                {total_fp}")
    print(f"FN:                {total_fn}")
    print(f"{'='*50}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1-Score:   {f1:.4f}")
    print(f"{'='*50}")
    return prec, rec, f1

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-j', default='TrainedModels/tabolistem_biobert50.json')
    parser.add_argument('-m', default='TrainedModels/best_biobert50_model.pth')
    parser.add_argument('-t', default='Corpus/GoldStandard.txt')
    parser.add_argument('-a', default='Corpus/GoldStandardAnnot.tsv')
    parser.add_argument('--no-argmax', action='store_true', help='Use threshold decoding instead of argmax')
    parser.add_argument('--threshold', type=float, default=1.0)
    args = parser.parse_args()
    evaluate_raw(args.j, args.m, args.t, args.a,
                 use_argmax=not args.no_argmax, threshold=args.threshold)
