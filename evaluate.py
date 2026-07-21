import argparse
import tabolistem_model as tabm
import pandas as pd

def load_gold_standard(text_path, annot_path):
    corpus_df = pd.read_csv(text_path, sep='\t', names=[
        'corpus', 'section', 'text'], encoding="utf-8-sig")
    annot_df = pd.read_csv(annot_path, encoding="utf-8-sig", sep='\t',
                           names=['corpus', 'section', 'start', 'end', 'metabolite'])

    sentences = {}
    for _, row in corpus_df.iterrows():
        key = (row['corpus'], row['section'])
        sentences[key] = row['text']

    gold = {}
    grouped = annot_df.groupby(['corpus', 'section'])
    for key, group in grouped:
        ents = set()
        for _, row in group.iterrows():
            ents.add((int(row['start']), int(row['end'])))
        gold[key] = ents

    return sentences, gold

def compute_f1(gold_ents, pred_ents):
    tp = len(gold_ents & pred_ents)
    fp = len(pred_ents - gold_ents)
    fn = len(gold_ents - pred_ents)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, prec, rec, f1

def evaluate(json_path, model_path, text_path, annot_path, use_argmax=True):
    print("Loading model...")
    tm = tabm.TaboListem()
    tm.load(json_path, model_path)

    print("Loading gold standard...")
    sentences, gold = load_gold_standard(text_path, annot_path)

    all_keys = sorted(sentences.keys())
    texts = [sentences[k] for k in all_keys]

    if use_argmax:
        print(f"Evaluating {len(texts)} sentences (argmax BIOES decoding)...")
        results = tm.batchprocess_argmax(texts)
    else:
        print(f"Evaluating {len(texts)} sentences (threshold decoding)...")
        results = tm.batchprocess(texts, threshold=0.5)

    total_tp = 0
    total_fp = 0
    total_fn = 0
    sentence_count = 0
    sentence_f1_sum = 0.0

    for i, key in enumerate(all_keys):
        gold_ents = gold.get(key, set())
        pred_ents = set()
        for (start, end, _) in results[i]:
            pred_ents.add((start, end))

        tp, fp, fn, prec, rec, f1 = compute_f1(gold_ents, pred_ents)
        total_tp += tp
        total_fp += fp
        total_fn += fn
        sentence_f1_sum += f1
        sentence_count += 1

    total_prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    total_rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    total_f1 = 2 * total_prec * total_rec / (total_prec + total_rec) if (total_prec + total_rec) > 0 else 0.0
    avg_f1 = sentence_f1_sum / sentence_count if sentence_count > 0 else 0.0

    decoding = "argmax" if use_argmax else "threshold"
    print(f"\n{'='*50}")
    print(f"Gold standard entities:  {total_tp + total_fn}")
    print(f"Predicted entities:      {total_tp + total_fp}")
    print(f"Correct (TP):            {total_tp}")
    print(f"False positives:         {total_fp}")
    print(f"False negatives:         {total_fn}")
    print(f"{'='*50}")
    print(f"Precision:  {total_prec:.4f}")
    print(f"Recall:     {total_rec:.4f}")
    print(f"F1-Score:   {total_f1:.4f}")
    print(f"Avg F1/Sent: {avg_f1:.4f}")
    print(f"{'='*50}")

    return total_prec, total_rec, total_f1

def main():
    parser = argparse.ArgumentParser(description='Evaluate TABoLiSTM against gold standard.')
    parser.add_argument('-j', '--json_path', type=str,
                        default='TrainedModels/tabolistem_biobert50.json',
                        help='Path to model JSON file')
    parser.add_argument('-m', '--model_path', type=str,
                        default='TrainedModels/best_biobert50_model.pth',
                        help='Path to model weights file')
    parser.add_argument('-t', '--text_path', type=str,
                        default='Corpus/GoldStandard.txt',
                        help='Path to gold standard text file')
    parser.add_argument('-a', '--annot_path', type=str,
                        default='Corpus/GoldStandardAnnot.tsv',
                        help='Path to gold standard annotation file')
    parser.add_argument('--no-argmax', action='store_true', help='Use threshold decoding instead of argmax')
    args = parser.parse_args()

    evaluate(args.json_path, args.model_path, args.text_path, args.annot_path,
             use_argmax=not args.no_argmax)

if __name__ == '__main__':
    main()
