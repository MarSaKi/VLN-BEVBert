import os
import json
import jsonlines
import numpy as np

def load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=True):
    data = []
    for split in splits:
        if "/" not in split:    # the official splits
            if dataset == 'r2r':
                with open(os.path.join(anno_dir, 'R2R_%s_enc.json' % split)) as f:
                    new_data = json.load(f)
            elif dataset == 'rxr':
                new_data = []
                with jsonlines.open(os.path.join(anno_dir, 'rxr_%s_guide_enc_xlmr.jsonl'%split)) as f:
                    for item in f:
                        new_data.append(item)
            else:
                raise NotImplementedError('unspported dataset %s' % dataset)

            if not is_test:
                if dataset == 'r4r' and split == 'val_unseen':
                    ridxs = np.random.permutation(len(new_data))[:200]
                    new_data = [new_data[ridx] for ridx in ridxs]
        else:   # augmented data
            print('\nLoading augmented data %s for pretraining...' % os.path.basename(split))
            if dataset == 'r2r':
                with open(split) as f:
                    new_data = json.load(f)
            elif dataset == 'rxr':
                new_data = []
                with jsonlines.open(split) as f:
                    for item in f:
                        new_data.append(item)
            else:
                raise NotImplementedError('unspported dataset %s' % dataset)  
        # Join
        data += new_data
    return data

def construct_instrs(anno_dir, dataset, splits, tokenizer, max_instr_len=512, is_test=True):
    data = []
    for i, item in enumerate(load_instr_datasets(anno_dir, dataset, splits, tokenizer, is_test=is_test)):
        if dataset == 'rxr':
            # rxr annotations are already split
            new_item = dict(item)
            if 'path_id' in item:
                new_item['instr_id'] = '%d_%d'%(item['path_id'], item['instruction_id'])
            else: # test
                new_item['path_id'] = new_item['instr_id'] = str(item['instruction_id'])
            new_item['instr_encoding'] = item['instr_encoding'][:max_instr_len]
            data.append(new_item)
        else:
            # Split multiple instructions into separate entries
            for j, instr in enumerate(item['instructions']):
                new_item = dict(item)
                new_item['instr_id'] = '%s_%d' % (item['path_id'], j)
                new_item['instruction'] = instr
                new_item['instr_encoding'] = item['instr_encodings'][j][:max_instr_len]
                del new_item['instructions']
                del new_item['instr_encodings']
                data.append(new_item)
    return data