import torch
import numpy as np

class WhisperDataCollatorWhithPadding:
    
    def __init__(self, keepDataIndex=False, dataset=None):
        self.keepDataIndex=keepDataIndex
        self.dataset=dataset

    def __call__(self, features):
        input_ids, labels, dec_input_ids, data_idx = [], [], [], []
        for f in features:
            input_ids.append(f["input_ids"])
            labels.append(f["labels"])
            dec_input_ids.append(f["dec_input_ids"])
            data_idx.append(f["dataIndex"])

        input_ids = torch.concat([input_id[None, :] for input_id in input_ids])

        label_lengths = [len(lab) for lab in labels]
        dec_input_ids_length = [len(e) for e in dec_input_ids]
        max_label_len = max(label_lengths+dec_input_ids_length)

        labels = [np.pad(lab, (0, max_label_len - lab_len), 'constant', constant_values=-100) for lab, lab_len in zip(labels, label_lengths)]
        dec_input_ids = [np.pad(e, (0, max_label_len - e_len), 'constant', constant_values=50257) for e, e_len in zip(dec_input_ids, dec_input_ids_length)] # 50257 is eot token id

        batch = {
            "labels": labels,
            "dec_input_ids": dec_input_ids
        }

        batch = {k: torch.tensor(np.array(v), requires_grad=False) for k, v in batch.items()}
        batch["input_ids"] = input_ids

        if self.keepDataIndex: 
            batch["dataIndex"]=data_idx
            batch["dataset"]=self.dataset

        return batch
