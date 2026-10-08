"""Small-data preference training contracts shared by the standalone smoke runner."""
import hashlib
import io

import torch
import torch.nn.functional as F
from PIL import Image


def image_from_value(value):
    if isinstance(value, dict):
        value = value.get('bytes') or value.get('path')
    if isinstance(value, bytes):
        value = io.BytesIO(value)
    if isinstance(value, Image.Image):
        return value.convert('RGB')
    with Image.open(value) as image:
        return image.convert('RGB')


def image_key(value):
    image = image_from_value(value)
    header = f'RGB:{image.width}:{image.height}:'.encode()
    return hashlib.sha256(header + image.tobytes()).hexdigest()


def response_text(processor, question, response):
    messages = [
        {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]},
        {'role': 'assistant', 'content': [{'type': 'text', 'text': response}]},
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    eos = processor.tokenizer.eos_token
    return text if text.endswith(eos) else text + eos


def preference_loss(scores, preferred_ids, kind, ratings=None, regularization=0.001, scale=1.0):
    """Match upstream end-score RM/CM losses; IDs refer to raw response 1 or 2.

    RM preferred IDs are helpful IDs; CM preferred IDs are safer IDs. CM ranks
    the other response higher in cost and anchors both scores with -harmless rate.
    """
    if scores.ndim != 2 or scores.shape[1] != 2 or kind not in {'rm', 'cm'}:
        raise ValueError('Expected (pairs, 2) scores and kind rm/cm')
    ids = torch.as_tensor(preferred_ids, device=scores.device, dtype=torch.long)
    if ids.shape != scores.shape[:1] or not torch.all((ids == 1) | (ids == 2)):
        raise ValueError('Each preference ID must be 1 or 2')
    high_ids = ids - 1 if kind == 'rm' else 2 - ids
    rows = torch.arange(len(scores), device=scores.device)
    high, low = scores[rows, high_ids], scores[rows, 1 - high_ids]
    loss = -F.logsigmoid(high - low).mean()
    if kind == 'cm':
        if ratings is None or ratings.shape != scores.shape:
            raise ValueError('CM requires one harmless rating for each response')
        signs = -ratings.to(scores.device).float()
        loss = loss - scale * F.logsigmoid(signs * scores).sum(1).mean()
    loss = loss + regularization * scores.square().mean()
    return loss, (high > low).float().mean()


def ranking_diagnostics(scores, preferred_ids, kind, categories):
    """Strict ranking accuracy (ties are incorrect), matching the trainer metric."""
    if scores.ndim != 2 or scores.shape[1] != 2 or len(categories) != len(scores):
        raise ValueError('Invalid paired scores/categories')
    if kind not in {'rm', 'cm'} or not torch.isfinite(scores).all():
        raise ValueError('Invalid kind or non-finite scores')
    ids = torch.as_tensor(preferred_ids, device=scores.device, dtype=torch.long)
    if ids.shape != scores.shape[:1] or not torch.all((ids == 1) | (ids == 2)):
        raise ValueError('Each preference ID must be 1 or 2')
    high_ids = ids - 1 if kind == 'rm' else 2 - ids
    rows = torch.arange(len(scores), device=scores.device)
    margins = scores[rows, high_ids] - scores[rows, 1 - high_ids]
    by_category = {}
    for category in sorted(set(categories)):
        mask = torch.tensor([c == category for c in categories], device=scores.device)
        by_category[category] = {
            'n': int(mask.sum()), 'correct': int((margins[mask] > 0).sum()),
            'accuracy': float((margins[mask] > 0).float().mean()),
        }
    return {'pairwise_correct': int((margins > 0).sum()),
            'pairwise_ties': int((margins == 0).sum()),
            'by_category': by_category}


def binary_cost_diagnostics(scores, unsafe):
    """Threshold-zero safety diagnostics and response-level AUC; not generation safety."""
    if scores.shape != unsafe.shape or unsafe.dtype != torch.bool or not torch.isfinite(scores).all():
        raise ValueError('Expected finite scores and same-shape Boolean unsafe labels')
    predicted = scores > 0
    harmful, harmless = scores[unsafe], scores[~unsafe]
    recalls = [float((predicted[unsafe == label] == label).float().mean())
               for label in (False, True) if (unsafe == label).any()]
    auc = None
    if harmful.numel() and harmless.numel():
        comparison = harmful[:, None] - harmless[None, :]
        auc = float(((comparison > 0).float() + 0.5 * (comparison == 0).float()).mean())
    return {'n_responses': scores.numel(), 'n_unsafe': int(unsafe.sum()),
            'safety_label_accuracy_at_zero': float((predicted == unsafe).float().mean()),
            'balanced_safety_label_accuracy_at_zero': sum(recalls) / len(recalls),
            'majority_class_baseline': max(float(unsafe.float().mean()), 1 - float(unsafe.float().mean())),
            'true_unsafe': int((predicted & unsafe).sum()),
            'false_unsafe': int((predicted & ~unsafe).sum()),
            'true_safe': int((~predicted & ~unsafe).sum()),
            'false_safe': int((~predicted & unsafe).sum()),
            'unsafe_cost_mean': float(harmful.mean()) if harmful.numel() else None,
            'safe_cost_mean': float(harmless.mean()) if harmless.numel() else None,
            'response_safety_auc': auc}
