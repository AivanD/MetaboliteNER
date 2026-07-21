import tabolistem_model as tabm
import argparse
import os

def main():
    parser = argparse.ArgumentParser(
        description='Train TABoLiSTM (PyTorch) metabolite NER model.')

    parser.add_argument('-t', '--text_path', type=str,
                        help='Path to training text file (.txt)')
    parser.add_argument('-a', '--annot_path', type=str,
                        help='Path to training annotation file (.tsv)')
    parser.add_argument('-o', '--output_name', type=str,
                        help='Output name for saved model files.')
    parser.add_argument('-e', '--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('-b', '--batch_size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('-v', '--val_split', type=float, default=0.1,
                        help='Validation split ratio (default: 0.1, paper uses 75:10:15)')
    parser.add_argument('--clear-cache', action='store_true',
                        help='Delete cached Seqs.npy files to force regeneration')

    args = parser.parse_args()

    if args.clear_cache:
        for f in ['Seqs.npy', 'Seqs_val.npy']:
            if os.path.exists(f):
                os.remove(f)
                print(f"Removed cached {f}")

    tm = tabm.TaboListem()
    tm.train(args.text_path, args.annot_path,
             args.output_name, epochs=args.epochs, batch_size=args.batch_size,
             val_split=args.val_split)

if __name__ == '__main__':
    main()
