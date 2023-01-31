# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import torch
from torch.cuda import amp


def evaluate(model, test_loader, dtype, use_amp, device):
    # model to eval mode
    model.eval()

    # compute accuracy and loss
    total_num, total_loss, total_correct = 0, 0, 0
    with torch.no_grad():
        for (input_ids, attention_mask, labels) in test_loader:
            # forward with mixed precision
            with amp.autocast(dtype=dtype, enabled=use_amp):
                losses, logits, _ = model(
                    input_ids=input_ids.to(device),
                    attention_mask=attention_mask.to(device),
                    labels=labels.to(device),
                )
                loss = losses.mean()

            total_num += len(input_ids)
            total_loss += loss.item() * len(input_ids)
            total_correct += logits.cpu().argmax(1).eq(labels).sum().item()

    # model to train mode
    model.train()

    return total_correct / total_num, total_loss / total_num
