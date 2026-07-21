import time
import sys
import os
import random
import json
import numpy as np
import re

from datetime import datetime

import torch
import torch.nn as nn
import transformers
from transformers import BertModel

from featurizer import Featurizer
from utils import *
from corpusreader import CorpusReader

class SpatialDropout1d(nn.Module):
    """Drops entire timesteps (all features at a position) uniformly at random."""

    def __init__(self, p=0.1):
        super().__init__()
        self.p = p

    def forward(self, x):
        # x: (batch, seq, feat)
        if self.training:
            mask = torch.rand(x.shape[0], x.shape[1], 1, device=x.device) > self.p
            return x * mask.to(x.dtype) / (1 - self.p)
        return x


class TaboListemModel(nn.Module):
    """PyTorch model: Conv1D + BERT → BiLSTM → BIOES classification."""

    def __init__(self, nilen, num_labels, bert_pretrain_path='bert-base-cased'):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_pretrain_path)
        for p in self.bert.parameters():
            p.requires_grad = False

        self.bert_drop = SpatialDropout1d(0.25)

        self.conv = nn.Conv1d(nilen, 256, 3, padding='same')
        self.conv_act = nn.ReLU()
        self.conv_drop = nn.Dropout1d(0.5)

        self.lstm = nn.LSTM(
            input_size=768 + 256,
            hidden_size=64,
            num_layers=1,
            bidirectional=True,
            dropout=0.0,
        )
        self.output = nn.Linear(128, num_labels)

    def forward(self, input_ids, attention_mask, ni):
        bert_out = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        bert_out = self.bert_drop(bert_out)

        ni_c = ni.transpose(1, 2)
        conv_out = self.conv_act(self.conv(ni_c))
        conv_out = self.conv_drop(conv_out)
        conv_out = conv_out.transpose(1, 2)

        combined = torch.cat([bert_out, conv_out], dim=2)
        lstm_out, _ = self.lstm(combined)
        return self.output(lstm_out)


class TaboListem(object):
    """A model for metabolite named entity recognition."""

    def __init__(self):
        bert_pretrain_path = 'dmis-lab/biobert-base-cased-v1.2'
        self.bert_pretrain_path = bert_pretrain_path
        self.tokenizer = transformers.BertTokenizerFast.from_pretrained(
            bert_pretrain_path, do_lower_case=False)
        self.max_seq_len = 256
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def _str_to_seq(self, s):
        seq = {"tokens": [], "bio": [],
               "tokstart": [], "tokend": [],
               "chemtok_tokens": [], "chemtok_rep": [], "str": s}

        # BioBERT tokenization directly (matches paper methodology for TABoLiSTM)
        tokenized = self.tokenizer.tokenize(s)

        # Each BioBERT token corresponds to one "chemtok" token (identity mapping)
        chemtok_tokens = list(tokenized)
        chemtok_rep = [1] * len(tokenized)

        # Truncate
        if len(tokenized) > self.max_seq_len - 2:
            chemtok_tokens = chemtok_tokens[:self.max_seq_len - 2]
            tokenized = tokenized[:self.max_seq_len - 2]
            chemtok_rep = chemtok_rep[:self.max_seq_len - 2]

        seq["chemtok_tokens"] = chemtok_tokens
        seq["tokens"] = tokenized.copy()
        seq["chemtok_rep"] = chemtok_rep

        while tokenized:
            tok = tokenized.pop()
            if len(tok) > 2:
                if tok[0:2] == '##':
                    tok = tok[2:]

            tokstart = s.rfind(tok)
            tokend = tokstart + len(tok)
            s = s[:tokstart]
            seq["tokstart"].append(tokstart)
            seq["tokend"].append(tokend)
        seq["tokstart"].reverse()
        seq["tokend"].reverse()

        return seq

    def _prepare_seqs(self, seqs, verbose=True, save_path='Seqs.npy'):
        # Cache vocab lookups for massive speedup
        vocab = self.tokenizer.vocab
        cls_id = vocab['[CLS]']
        sep_id = vocab['[SEP]']

        print("Preparing wordn for {} sequences...".format(len(seqs)))
        t_start = time.time()
        for seq in seqs:
            seq["wordn"] = [cls_id] + [vocab[t] for t in seq["tokens"]] + [sep_id]
        print(f"Wordn preparation completed in {time.time() - t_start:.2f} seconds.")

        # Features computed on chemtok tokens, replicated to BERT sub-words
        nilen = len(self.fzr.num_feats_for_tok(seqs[0]['chemtok_tokens'][0]))
        ni_cls = [0] * nilen
        ni_sep = [0] * nilen

        print("Starting single-threaded featurization loop...")
        t_start = time.time()
        for seq in seqs:
            # Replicate chemtok token features to each BERT sub-word
            tok_rep = sum([[t] * j for t, j in zip(seq['chemtok_tokens'], seq['chemtok_rep'])], [])
            seq["ni"] = np.array([ni_cls] + [self.fzr.num_feats_for_tok(i)
                                              for i in tok_rep] + [ni_sep])
        print(f"Featurization loop completed in {time.time() - t_start:.2f} seconds.")

        if save_path:
            np.save(save_path, seqs)
            print("Seqs saved as " + save_path)

        self.nilen = nilen

    def load_seqs(self, load_file='Seqs2.npy'):
        seqs = np.load(load_file, allow_pickle=True)
        self.nilen = len(seqs[0]['ni'][0])
        return seqs

    def build_model(self):
        self.model = TaboListemModel(
            self.nilen, len(self.lablist), self.bert_pretrain_path
        ).to(self.device)

    def train(self, textfile, annotfile, runname, epochs=50, batch_size=4,
              val_split=0.1):
        if os.path.exists("Seqs.npy"):
            print(f"Loading cached sequences from Seqs.npy at {datetime.now()}...", file=sys.stderr)
            train = self.load_seqs("Seqs.npy").tolist()
            seqs = train
        else:
            cr = CorpusReader(textfile, annotfile)
            train = cr.trainseqs
            seqs = train

        # Split into train and validation by article (PMCID)
        # Group sequences by their source article
        # We use the ents field which contains (PMCID, sectionID, ...) tuples
        article_seqs = {}
        for seq in train:
            if seq.get('ents'):
                pmcid = seq['ents'][0][0]
            else:
                pmcid = 'unknown'
            article_seqs.setdefault(pmcid, []).append(seq)

        article_list = list(article_seqs.keys())
        random.shuffle(article_list)
        val_count = int(len(article_list) * val_split)
        val_articles = set(article_list[:val_count])
        train_seqs = []
        val_seqs = []
        for pmcid in article_list:
            if pmcid in val_articles:
                val_seqs.extend(article_seqs[pmcid])
            else:
                train_seqs.extend(article_seqs[pmcid])
        print(f"Train: {len(train_seqs)} sequences, Validation: {len(val_seqs)} sequences", file=sys.stderr)

        seqs = train_seqs if val_seqs else train
        val_data = val_seqs
        # CorpusReader uses 'ss' for sentence string, but TaboListem expects 'str'
        for seq in val_data:
            seq['str'] = seq.get('str', seq.get('ss', ''))

        toklist = []
        tokcounts = {}
        labels = set()
        self.toklist = toklist
        self.tokdict = self.tokenizer.vocab
        self.tokcounts = tokcounts
        self.fzr = None
        self.lablist = None
        self.labdict = None
        self.model = None

        for seq in seqs:
            for i in seq["bio"]:
                labels.add(i)
        lablist = sorted(labels)
        lablist.reverse()
        labdict = {lablist[i]: i for i in range(len(lablist))}
        self.lablist = lablist
        self.labdict = labdict

        for seq in seqs + val_data:
            seq["bion"] = [labdict["O"]] + [labdict[i] for i in seq["bio"]] + [labdict["O"]]

        print("Make featurizer at", datetime.now(), file=sys.stderr)
        fzr = Featurizer(train_seqs if train_seqs else train)
        self.fzr = fzr

        if not os.path.exists("Seqs.npy"):
            self._prepare_seqs(seqs, save_path='Seqs.npy')
        if val_data:
            self._prepare_seqs(val_data, save_path='Seqs_val.npy')

        print("Make train dict at", datetime.now(), file=sys.stderr)
        train_l_d = {}
        for seq in seqs:
            l = len(seq["tokens"])
            if l not in train_l_d:
                train_l_d[l] = []
            train_l_d[l].append(seq)

        # Build validation dict for batching
        val_l_d = {}
        for seq in val_data:
            l = len(seq["tokens"])
            if l not in val_l_d:
                val_l_d[l] = []
            val_l_d[l].append(seq)

        self.build_model()
        model = self.model

        outjo = {
            "fzr": self.fzr.to_json_obj(),
            "lablist": self.lablist,
            "nilen": self.nilen,
            "bert_pretrain_path": self.bert_pretrain_path,
        }

        print("Serialize at", datetime.now(), file=sys.stderr)
        os.makedirs("./TrainedModels", exist_ok=True)
        jf = open("./TrainedModels/tabolistem_%s.json" % runname, "w", encoding="utf-8")
        json.dump(outjo, jf)
        jf.close()

        sizes = list(train_l_d)

        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        best_val_f1 = -1.0
        best_epoch = 0

        def evaluate_model():
            """Run validation with argmax BIOES decoding and return F1 score."""
            model.eval()
            total_tp = total_fp = total_fn = 0
            with torch.no_grad():
                for vsize in val_l_d:
                    for seq in val_l_d[vsize]:
                        tx_id, att_mask, tx_ni = self._seq_to_tensors(seq)
                        outputs = model(tx_id, att_mask, tx_ni)[0]
                        outputs = outputs.cpu().numpy()
                        seq["tagfeat"] = outputs[1:-1]
                        raw_ents = self.decode_bioes(seq)
                        pred_ents = set(raw_ents.keys())
                        # Reconstruct gold from BIOES tags
                        gold_ents = set()
                        bio = seq["bio"]
                        for i in range(len(bio)):
                            if bio[i] == 'S':
                                gold_ents.add((seq["tokstart"][i], seq["tokend"][i]))
                            elif bio[i] == 'B':
                                start = seq["tokstart"][i]
                                j = i + 1
                                while j < len(bio) and bio[j] == 'I':
                                    j += 1
                                if j < len(bio) and bio[j] == 'E':
                                    gold_ents.add((start, seq["tokend"][j]))
                        tp = len(gold_ents & pred_ents)
                        fp = len(pred_ents - gold_ents)
                        fn = len(gold_ents - pred_ents)
                        total_tp += tp
                        total_fp += fp
                        total_fn += fn
            prec = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
            rec = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
            return f1, prec, rec

        for epoch in range(epochs):
            print("Epoch", epoch, "start at", datetime.now(), file=sys.stderr)
            random.shuffle(sizes)

            tnt = 0
            for size in sizes:
                tnt += size * len(train_l_d[size])
            totloss = 0
            totacc = 0
            div = 0

            time_initial = time.time()

            total_batches = sum(len(train_l_d[s]) // batch_size + 1 for s in sizes)
            total_batch_counter = 0

            model.train()
            for size in sizes:
                if size == 1:
                    continue

                batch_counter = 0
                while batch_counter < len(train_l_d[size]):
                    if div == 0:
                        _loss = 0
                        _acc = 0
                    else:
                        _loss = totloss / div
                        _acc = totacc / div

                    print('\rTraining on size {}, batch {}/{}... Loss: {}; Accuracy: {}; Estimated elapsed time: N/A'.format(
                        str(size), str(total_batch_counter), str(total_batches),
                        str(round(_loss, 4)), str(round(_acc, 4))), end='', flush=True)

                    batch = train_l_d[size][batch_counter:batch_counter + batch_size]
                    batch_counter += batch_size
                    total_batch_counter += 1

                    wordn_batch = [seq["wordn"] for seq in batch]
                    ni_batch = [seq["ni"] for seq in batch]
                    bion_batch = [seq["bion"] for seq in batch]

                    tx_id = torch.tensor(wordn_batch, dtype=torch.long, device=self.device)
                    att_mask = torch.ones_like(tx_id, dtype=torch.long, device=self.device)
                    tx_ni = torch.tensor(np.array(ni_batch), dtype=torch.float32, device=self.device)
                    labels_t = torch.tensor(bion_batch, dtype=torch.long, device=self.device)

                    optimizer.zero_grad()
                    outputs = model(tx_id, att_mask, tx_ni)
                    loss = criterion(outputs.view(-1, outputs.size(-1)), labels_t.view(-1))
                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        preds = torch.argmax(outputs, dim=2)
                        correct = (preds == labels_t).sum().item()
                    acc = correct / labels_t.numel()

                    size_factor = size * len(batch)
                    div += size_factor
                    totloss += loss.item() * size_factor
                    totacc += acc * size_factor

            print(f"\nTrained at {datetime.now()}, Loss {round(totloss / div, 4)}, "
                  f"Accuracy {round(totacc / div, 4)}", file=sys.stderr)

            # Validation (skip if no validation split)
            val_f1 = 0.0
            if val_l_d:
                print("Running validation...", file=sys.stderr)
                val_f1, val_prec, val_rec = evaluate_model()
                print(f"Validation F1: {val_f1:.4f} (P: {val_prec:.4f}, R: {val_rec:.4f})", file=sys.stderr)

            torch.save(model.state_dict(),
                       "./TrainedModels/epoch_%s_%s_weights.pt" % (epoch, runname))
            torch.save(model,
                       "./TrainedModels/epoch_%s_%s_model.pth" % (epoch, runname))

            if val_f1 > best_val_f1 or not val_l_d:
                best_val_f1 = val_f1
                best_epoch = epoch
                # Copy best model (always update when no validation split)
                torch.save(model.state_dict(),
                           f"./TrainedModels/best_{runname}_weights.pt")
                torch.save(model,
                           f"./TrainedModels/best_{runname}_model.pth")
                if val_l_d:
                    print(f"*** New best epoch (F1={best_val_f1:.4f}) ***", file=sys.stderr)

            if val_l_d:
                print(f"Best validation F1 so far: {best_val_f1:.4f} at epoch {best_epoch}", file=sys.stderr)

        if val_l_d:
            print(f"\nTraining complete. Best epoch: {best_epoch} (F1={best_val_f1:.4f})", file=sys.stderr)
        else:
            print(f"\nTraining complete. Trained {epochs} epochs on full corpus.", file=sys.stderr)

    def load(self, json_file, model_path=None):
        with open(json_file, "r", encoding="utf-8") as jf:
            jo = json.load(jf)

        print("Loading model...")
        self.lablist = jo["lablist"]
        self.fzr = Featurizer(None, jo["fzr"])
        self.nilen = jo["nilen"]
        self.labdict = {self.lablist[i]: i for i in range(len(self.lablist))}
        print("Auxiliary information read at", datetime.now(), file=sys.stderr)

        if model_path is None:
            # Try to load full model from .pth file next to json
            dirpath = os.path.dirname(json_file)
            base = os.path.splitext(os.path.basename(json_file))[0]
            pth = os.path.join(dirpath, base + ".pth")
            if os.path.exists(pth):
                model_path = pth
            else:
                pth = os.path.join(dirpath, base + "_model.pth")
                if os.path.exists(pth):
                    model_path = pth

        if model_path is None:
            raise FileNotFoundError(
                "No model file found. Provide model_path or place a .pth file next to the JSON.")

        # Auto-append .pth or .pt if missing
        if not model_path.endswith(('.pth', '.pt')):
            if os.path.exists(model_path + '.pth'):
                model_path = model_path + '.pth'
            elif os.path.exists(model_path + '.pt'):
                model_path = model_path + '.pt'

        if model_path.endswith('.pth'):
            # Full model save (includes BERT weights)
            self.model = torch.load(model_path, map_location=self.device, weights_only=False)
        else:
            # Weights-only save — rebuild architecture first
            self.build_model()
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device, weights_only=True)
            )

        self.model.eval()
        print("Model weight loaded at ", datetime.now(), file=sys.stderr)
        print("Model established.")

    def _seq_to_tensors(self, seq):
        tx_id = torch.tensor([seq["wordn"]], dtype=torch.long, device=self.device)
        att_mask = torch.ones_like(tx_id, dtype=torch.long, device=self.device)
        tx_ni = torch.tensor([seq["ni"]], dtype=torch.float32, device=self.device)
        return tx_id, att_mask, tx_ni

    def score_to_ent(self, seq, threshold=0.5):
        l = seq['tagfeat'].shape[0]
        ents = {}
        for i in range(l):
            for j in range(i, l):
                if i == j:
                    if seq['tagfeat'][i, self.labdict['S']] > threshold:
                        ents[(seq["tokstart"][i], seq["tokend"][j])] = \
                            seq['str'][seq["tokstart"][i]:seq["tokend"][j]]
                else:
                    try:
                        pseq = [seq['tagfeat'][i, self.labdict['B']]] + \
                               [k[self.labdict['I']] for k in seq['tagfeat'][i+1:j, :]] + \
                               [seq['tagfeat'][j, self.labdict['E']]]
                    except:
                        print(seq['tagfeat'].shape[0], i, j)
                        raise Exception

                    if min(pseq) > threshold:
                        ents[(seq["tokstart"][i], seq["tokend"][j])] = \
                            seq['str'][seq["tokstart"][i]:seq["tokend"][j]]

                    if seq["tagfeat"][j, self.labdict['I']] <= threshold:
                        break
        return ents

    def decode_bioes(self, seq):
        """Argmax-based BIOES decoding — matches paper evaluation methodology."""
        tagfeat = seq['tagfeat']
        tags = [self.lablist[np.argmax(tagfeat[i])] for i in range(tagfeat.shape[0])]
        entities = {}
        i = 0
        while i < len(tags):
            if tags[i] == 'S':
                entities[(seq["tokstart"][i], seq["tokend"][i])] = seq['str'][seq["tokstart"][i]:seq["tokend"][i]]
                i += 1
            elif tags[i] == 'B':
                start = seq["tokstart"][i]
                j = i + 1
                while j < len(tags) and tags[j] == 'I':
                    j += 1
                if j < len(tags) and tags[j] == 'E':
                    entities[(start, seq["tokend"][j])] = seq['str'][start:seq["tokend"][j]]
                    i = j + 1
                else:
                    i += 1
            else:
                i += 1
        return entities

    def process_argmax(self, text):
        """Run inference with argmax BIOES decoding (no threshold)."""
        results = []
        if len(text) == 0:
            return results
        seq = self._str_to_seq(text)
        if len(seq["tokens"]) == 0:
            return results
        self._prepare_seqs([seq], False, save_path=None)

        tx_id, att_mask, tx_ni = self._seq_to_tensors(seq)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tx_id, att_mask, tx_ni)[0]
        outputs = outputs.cpu().numpy()
        seq["tagfeat"] = outputs[1:-1]

        loc_word_dict = self.decode_bioes(seq)
        results = post_process(loc_word_dict, text)
        return results

    def batchprocess_argmax(self, instrs):
        """Batch inference with argmax BIOES decoding (no threshold)."""
        batch_results = [None] * len(instrs)
        train_l_d = {}
        seq_index = -1

        print("Start processing {} input strings...".format(str(len(instrs))))
        time_initial_all = time.time()
        time_initial = time.time()
        est_hr, est_min, est_sec = 0, 0, 0
        print()
        for text in instrs:
            formatted_time = '%02d:%02d:%02d' % (est_hr, est_min, est_sec)

            seq_index += 1
            print('\rPreparing sequences {}/{} (ETA {})...'.format(
                str(seq_index + 1), str(len(instrs)), formatted_time), end='', flush=True)
            seq = self._str_to_seq(text)
            seq['idx'] = seq_index
            self._prepare_seqs([seq], False, save_path=None)
            token_len = len(seq["ni"])
            if len(text) == 0 or token_len == 0:
                batch_results[seq_index] = []
                continue

            try:
                train_l_d[token_len].append(seq)
            except KeyError:
                train_l_d[token_len] = [seq]

            estimate_time = int((time.time() - time_initial) *
                                (len(instrs) / (seq_index + 1) - 1))
            est_hr = int(estimate_time / 3600)
            est_min = int((estimate_time % 3600) / 60)
            est_sec = int(estimate_time % 60)

        batch_counter = 0
        time_initial = time.time()
        est_hr, est_min, est_sec = 0, 0, 0
        print()
        self.model.eval()
        with torch.no_grad():
            for batch_len in train_l_d:
                formatted_time = '%02d:%02d:%02d' % (est_hr, est_min, est_sec)
                print('\rGenerating predictions {}/{} (ETA {})...'.format(
                    str(batch_counter), str(len(instrs)), formatted_time), end='', flush=True)
                seqs = train_l_d[batch_len]

                tx_id = torch.tensor([seq["wordn"] for seq in seqs],
                                     dtype=torch.long, device=self.device)
                att_mask = torch.ones_like(tx_id, dtype=torch.long, device=self.device)
                tx_ni = torch.tensor([seq['ni'] for seq in seqs],
                                     dtype=torch.float32, device=self.device)

                outputs = self.model(tx_id, att_mask, tx_ni)
                outputs = outputs.cpu().numpy()

                for i, seq in enumerate(seqs):
                    seq["tagfeat"] = outputs[i][1:-1]
                    loc_word_dict = self.decode_bioes(seq)
                    batch_results[seq['idx']] = post_process(loc_word_dict, seq['str'])

                batch_counter += len(seq)
                estimate_time = int((time.time() - time_initial) *
                                    (len(instrs) / (batch_counter) - 1))
                est_hr = int(estimate_time / 3600)
                est_min = int((estimate_time % 3600) / 60)
                est_sec = int(estimate_time % 60)

        estimate_time = int((time.time() - time_initial_all))
        est_hr = int(estimate_time / 3600)
        est_min = int((estimate_time % 3600) / 60)
        est_sec = int(estimate_time % 60)
        formatted_time = '%02d:%02d:%02d' % (est_hr, est_min, est_sec)
        print()
        print('Done. Finished in {}.'.format(formatted_time))
        return batch_results

    def process(self, text, threshold=0.5):
        results = []
        if len(text) == 0:
            return results
        seq = self._str_to_seq(text)
        if len(seq["tokens"]) == 0:
            return results
        self._prepare_seqs([seq], False, save_path=None)

        tx_id, att_mask, tx_ni = self._seq_to_tensors(seq)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(tx_id, att_mask, tx_ni)[0]
        outputs = outputs.cpu().numpy()
        seq["tagfeat"] = outputs[1:-1]

        loc_word_dict = self.score_to_ent(seq, threshold=threshold)
        results = post_process(loc_word_dict, text)
        return results

    def batchprocess(self, instrs, threshold=0.5):
        batch_results = [None] * len(instrs)
        train_l_d = {}
        seq_index = -1

        print("Start processing {} input strings...".format(str(len(instrs))))
        time_initial_all = time.time()
        time_initial = time.time()
        est_hr, est_min, est_sec = 0, 0, 0
        print()
        for text in instrs:
            formatted_time = '%02d:%02d:%02d' % (est_hr, est_min, est_sec)

            seq_index += 1
            print('\rPreparing sequences {}/{} (ETA {})...'.format(
                str(seq_index + 1), str(len(instrs)), formatted_time), end='', flush=True)
            seq = self._str_to_seq(text)
            seq['idx'] = seq_index
            self._prepare_seqs([seq], False, save_path=None)
            token_len = len(seq["ni"])
            if len(text) == 0 or token_len == 0:
                batch_results[seq_index] = []
                continue

            try:
                train_l_d[token_len].append(seq)
            except KeyError:
                train_l_d[token_len] = [seq]

            estimate_time = int((time.time() - time_initial) *
                                (len(instrs) / (seq_index + 1) - 1))
            est_hr = int(estimate_time / 3600)
            est_min = int((estimate_time % 3600) / 60)
            est_sec = int(estimate_time % 60)

        batch_counter = 0
        time_initial = time.time()
        est_hr, est_min, est_sec = 0, 0, 0
        print()
        self.model.eval()
        with torch.no_grad():
            for batch_len in train_l_d:
                formatted_time = '%02d:%02d:%02d' % (est_hr, est_min, est_sec)
                print('\rGenerating predictions {}/{} (ETA {})...'.format(
                    str(batch_counter), str(len(instrs)), formatted_time), end='', flush=True)
                seqs = train_l_d[batch_len]

                tx_id = torch.tensor([seq["wordn"] for seq in seqs],
                                     dtype=torch.long, device=self.device)
                att_mask = torch.ones_like(tx_id, dtype=torch.long, device=self.device)
                tx_ni = torch.tensor([seq['ni'] for seq in seqs],
                                     dtype=torch.float32, device=self.device)

                outputs = self.model(tx_id, att_mask, tx_ni)
                outputs = outputs.cpu().numpy()

                for i, seq in enumerate(seqs):
                    seq["tagfeat"] = outputs[i][1:-1]
                    loc_word_dict = self.score_to_ent(seq, threshold=threshold)
                    batch_results[seq['idx']] = post_process(loc_word_dict, seq['str'])

                batch_counter += len(seq)
                estimate_time = int((time.time() - time_initial) *
                                    (len(instrs) / (batch_counter) - 1))
                est_hr = int(estimate_time / 3600)
                est_min = int((estimate_time % 3600) / 60)
                est_sec = int(estimate_time % 60)

        estimate_time = int((time.time() - time_initial_all))
        est_hr = int(estimate_time / 3600)
        est_min = int((estimate_time % 3600) / 60)
        est_sec = int(estimate_time % 60)
        formatted_time = '%02d:%02d:%02d' % (est_hr, est_min, est_sec)
        print()
        print('Done. Finished in {}.'.format(formatted_time))
        return batch_results

    def process_abbrev(self, text, pmc_id, abbrev_folder, threshold=0.5):
        def get_abbrev_set(pmc_id, abbrev_folder, threshold):
            abbrev_set = set()
            abbrev_path = os.path.join(abbrev_folder, pmc_id + '_abbreviations.json')
            try:
                with open(abbrev_path, 'r', encoding='utf-8-sig') as abbrev_file:
                    abbrev_dict = json.load(abbrev_file)
            except FileNotFoundError:
                print("{} doesn't exist, returning empty set...".format(abbrev_path))
                return abbrev_set

            for abbrev_type in abbrev_dict:
                for abbrev in abbrev_dict[abbrev_type]:
                    entity = abbrev_dict[abbrev_type][abbrev]
                    output = self.process(entity, threshold=threshold)
                    if not output:
                        continue
                    if output[0][2] == entity:
                        abbrev_set.add(abbrev)
            return abbrev_set

        abbrev_set = get_abbrev_set(pmc_id, abbrev_folder, threshold)
        abbrev_res = []
        for abbrev in abbrev_set:
            abbrev_iter = re.finditer(r'\b{}\b'.format(abbrev), text)
            for abb in abbrev_iter:
                pos = abb.span()
                abbrev_res += [(pos[0], pos[1], abbrev)]

        abbrev_res.sort(key=lambda x: x[0])
        return abbrev_res
