"""Padding collator for VideoQADataset batches.

Pads input_ids/labels to the batch max length and returns a `pad_mask` marking
real tokens (1) vs. collator padding (0). The model uses pad_mask to trim each
sample back to its true length before splicing in visual tokens, then re-pads —
so collator padding never reaches the LLM's attention. input_ids are padded with
0 and labels with -100; IMAGE_TOKEN_INDEX (negative) is preserved untouched.
"""

import torch


def collate_fn(batch):
    max_len = max(item["input_ids"].shape[0] for item in batch)
    input_ids, labels, pad_mask, pixel_values = [], [], [], []
    for item in batch:
        ids, lbl = item["input_ids"], item["labels"]
        n = ids.shape[0]
        pad = max_len - n
        if pad > 0:
            ids = torch.cat([ids, torch.zeros(pad, dtype=ids.dtype)])
            lbl = torch.cat([lbl, torch.full((pad,), -100, dtype=lbl.dtype)])
        input_ids.append(ids)
        labels.append(lbl)
        pad_mask.append(torch.cat([torch.ones(n), torch.zeros(pad)]))
        pixel_values.append(item["pixel_values"])
    return {
        "input_ids": torch.stack(input_ids),
        "labels": torch.stack(labels),
        "pad_mask": torch.stack(pad_mask).long(),
        "pixel_values": torch.stack(pixel_values),
    }
