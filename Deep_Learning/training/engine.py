"""One-epoch training loop and evaluation."""
import torch
from tqdm import tqdm

from ..evaluation import compute_metrics


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, use_amp, max_grad_norm=5.0):
    model.train()
    total, n = 0.0, 0
    pbar = tqdm(loader, desc="train", leave=False)
    for x, y in pbar:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device.type, enabled=use_amp):
            loss = criterion(model(x), y)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()
        total += loss.item() * len(y)
        n += len(y)
        pbar.set_postfix(loss=f"{total / n:.4f}")
    return total / n


@torch.no_grad()
def predict(model, loader, device, use_amp, tta=False, criterion=None):
    """Return (labels, probs, mean loss or None)."""
    model.eval()
    total, probs, labels = 0.0, [], []
    for x, y in tqdm(loader, desc="eval", leave=False):
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, enabled=use_amp):
            logits = model(x)
            if tta:  # average with horizontally flipped view
                logits = (logits + model(torch.flip(x, dims=[3]))) / 2
        logits = logits.float()
        if criterion is not None:
            total += criterion(logits, y).item() * len(y)
        probs.append(torch.softmax(logits, 1).cpu())
        labels.append(y.cpu())
    labels = torch.cat(labels).numpy()
    return labels, torch.cat(probs).numpy(), (total / len(labels) if criterion is not None else None)


def evaluate(model, loader, criterion, device, use_amp, tta=False):
    """Return (metrics dict, labels, preds, probs)."""
    labels, probs, loss = predict(model, loader, device, use_amp, tta, criterion)
    preds = probs.argmax(1)
    return {"loss": loss, **compute_metrics(labels, preds)}, labels, preds, probs
