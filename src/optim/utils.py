import numpy as np
import torch
import torch.nn.functional as F
from contextlib import nullcontext, contextmanager, ExitStack


def get_batch(dataloader, device="cpu"):
    batch = next(dataloader)
    if len(batch) == 2:
        x, y = batch
        tgt_x, tgt_y = None, None
    elif len(batch) == 4:
        x, y, tgt_x, tgt_y = batch
    else:
        raise ValueError(f"Unexpected batch format: {len(batch)} elements in batch")

    if "cuda" in torch.device(device).type:
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
        tgt_x = tgt_x.pin_memory().to(device, non_blocking=True) if tgt_x is not None else None
        tgt_y = tgt_y.pin_memory().to(device, non_blocking=True) if tgt_y is not None else None
    else:
        x = x.to(device)
        y = y.to(device)
        tgt_x = tgt_x.to(device, non_blocking=True) if tgt_x is not None else None
        tgt_y = tgt_y.to(device, non_blocking=True) if tgt_y is not None else None

    if tgt_x is not None:
        return x, y, tgt_x, tgt_y
    return x, y


@torch.no_grad()
def eval(model, data_val_iter, exp_assignment, device='cpu', max_num_batches=24, ctx=nullcontext(), token_loss_tracker=None):
    assert model.training == False

    loss_list_val, acc_list = [], []
    val_exp_assignment_index = 0

    for _ in range(max_num_batches): 
        x, y = get_batch(data_val_iter, device=device)
        b, t = x.size()

        with ctx:
            outputs = model(x, targets=y, get_logits=True, 
                            exp_assignment=exp_assignment,
                            exp_assignment_index=val_exp_assignment_index,
                            token_loss_tracker=token_loss_tracker)
        
            val_exp_assignment_index += b * t

            val_loss = outputs['loss'].mean()

        loss_list_val.append(val_loss)
        acc_list.append((outputs['logits'].argmax(-1) == y).float().mean())

    val_acc = torch.stack(acc_list).mean().item()
    val_loss = torch.stack(loss_list_val).mean().item()
    val_perplexity = 2.71828 ** val_loss

    return val_acc, val_loss, val_perplexity


def save_checkpoint(distributed_backend, model, opt, scheduler, itr, ckpt_path, **extra_args):

    checkpoint = dict({
        'model': distributed_backend.get_raw_model(model).state_dict(),
        'optimizer': opt.state_dict(),
        'scheduler': scheduler.state_dict(),
        'itr': itr,
    }, **extra_args)

    torch.save(checkpoint, ckpt_path)
