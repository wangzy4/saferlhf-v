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


def preference_loss(scores, preferred_ids, kind, ratings=None, regularization=0.001, scale=1.0,
                    native_precision=False):
    """Match upstream end-score RM/CM losses; IDs refer to raw response 1 or 2.

    RM preferred IDs are helpful IDs; CM preferred IDs are safer IDs. CM ranks
    the other response higher in cost and anchors both scores with -harmless rate.
    native_precision=True preserves upstream integer signs, separate means and
    stack reduction; default False retains the historical FP32 CM absolute term.
    """
    if scores.ndim != 2 or scores.shape[1] != 2 or kind not in {'rm', 'cm'}:
        raise ValueError('Expected (pairs, 2) scores and kind rm/cm')
    ids = torch.as_tensor(preferred_ids, device=scores.device, dtype=torch.long)
    if ids.shape != scores.shape[:1] or not torch.all((ids == 1) | (ids == 2)):
        raise ValueError('Each preference ID must be 1 or 2')
    high_ids = ids - 1 if kind == 'rm' else 2 - ids
    rows = torch.arange(len(scores), device=scores.device)
    high, low = scores[rows, high_ids], scores[rows, 1 - high_ids]
    if native_precision:
        # Match upstream's shared ordered output/chunk autograd node as well as
        # arithmetic: separate gather branches can change gradient accumulation.
        high, low = torch.cat([high, low]).chunk(2)
    loss = None
    if kind == 'rm' or not native_precision:
        loss = -F.logsigmoid(high - low).mean()
    if kind == 'cm':
        if ratings is None or ratings.shape != scores.shape:
            raise ValueError('CM requires one harmless rating for each response')
        if native_precision:
            if (not torch.isfinite(ratings).all() or
                    (ratings.is_floating_point() and not torch.all(ratings == ratings.round()))):
                raise ValueError('Native harmless ratings must be finite integers')
            signs = -ratings.to(device=scores.device, dtype=torch.int64)
            absolute = -F.logsigmoid(signs[rows, high_ids] * high).mean() - F.logsigmoid(
                signs[rows, 1 - high_ids] * low).mean()
            origin_loss = -F.logsigmoid(high - low).mean()
            loss = scale * absolute + origin_loss
        else:
            signs = -ratings.to(scores.device).float()
            loss = loss - scale * F.logsigmoid(signs * scores).sum(1).mean()
    regularized = torch.stack([low, high]) if native_precision else scores
    loss = loss + regularization * regularized.square().mean()
    return loss, (high > low).float().mean()


def preference_score_diagnostics(scores, preferred_ids, kind, ratings=None,
                                 regularization=0.001, scale=1.0, native_precision=False):
    """Detached loss parts and score-space gradients; never backprop into a model.

    These gradient norms refer to the two scalar scores, not parameter gradients.
    A mean over ranks is a mean of local norms, not a global gradient norm.
    Keep the input dtype to expose the same BF16 loss arithmetic as training.
    """
    # Validate through the existing contract without changing training loss.
    preference_loss(scores.detach(), preferred_ids, kind, ratings, regularization, scale,
                    native_precision)
    with torch.enable_grad():
        values = scores.detach().clone().requires_grad_(True)
        ids = torch.as_tensor(preferred_ids, device=values.device, dtype=torch.long)
        high_ids = ids - 1 if kind == 'rm' else 2 - ids
        rows = torch.arange(len(values), device=values.device)
        high, low = values[rows, high_ids], values[rows, 1 - high_ids]
        if native_precision:
            high, low = torch.cat([high, low]).chunk(2)
        margins = high - low
        regularized = torch.stack([low, high]) if native_precision else values
        parts = {'pairwise': -F.logsigmoid(margins).mean(),
                 'regularization': regularization * regularized.square().mean()}
        if kind == 'cm':
            if native_precision:
                signs = -ratings.to(device=values.device, dtype=torch.int64)
                parts['absolute'] = scale * (-F.logsigmoid(signs[rows, high_ids] * high).mean()
                                             - F.logsigmoid(signs[rows, 1 - high_ids] * low).mean())
            else:
                signs = -ratings.to(values.device).float()
                parts['absolute'] = -scale * F.logsigmoid(signs * values).sum(1).mean()
        if native_precision:
            # Preserve the optimized loss graph's creation/reduction order.
            total = preference_loss(values, ids, kind, ratings, regularization, scale, True)[0]
        else:
            total = parts['pairwise']
            if kind == 'cm':
                total = total + parts['absolute']
            total = total + parts['regularization']
        result = {'loss_total': total.detach(), 'score_mean': values.detach().float().mean(),
                  'score_std_population': values.detach().float().std(unbiased=False),
                  'score_positive_fraction': (values.detach() > 0).float().mean(),
                  'preferred_margin_mean': margins.detach().float().mean()}
        for name, loss in {**parts, 'total': total}.items():
            gradient = torch.autograd.grad(loss, values, retain_graph=True)[0]
            result[f'loss_{name}'] = loss.detach()
            result[f'score_gradient_l2_{name}'] = gradient.detach().float().norm()
        if kind == 'cm':
            result['rating_zero_fraction'] = (ratings == 0).float().mean().detach()
        if not all(torch.isfinite(value) for value in result.values()):
            raise ValueError('Non-finite score diagnostic')
        return result


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
        # Mann–Whitney AUC with tie weight 0.5, without an O(N_unsafe*N_safe)
        # comparison matrix. Preserve float64 distinctions; BF16/FP16 values
        # are represented exactly in float32 for portable searchsorted kernels.
        dtype = torch.float64 if scores.dtype == torch.float64 else torch.float32
        negative = harmless.detach().flatten().to(dtype).sort().values.contiguous()
        positive = harmful.detach().flatten().to(dtype).contiguous()
        lower = torch.searchsorted(negative, positive, right=False)
        upper = torch.searchsorted(negative, positive, right=True)
        auc = (lower + upper).sum().item() / (2 * positive.numel() * negative.numel())
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
