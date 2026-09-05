"""
Sliding-window / batch evaluation for cross-entropy loss and Perplexity (PPL).
"""

import math
from typing import List, Dict, Any, Optional
import torch
from transformers import PreTrainedModel, PreTrainedTokenizer
from rlfinetunelab.utils.logging import get_logger

logger = get_logger(__name__)


def compute_perplexity(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    texts: List[str],
    max_length: int = 2048,
    batch_size: int = 2,
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    Computes average loss and perplexity (exp(loss)) over a set of evaluation texts.
    Correctly ignores padding tokens (-100).
    """
    if not texts:
        return {"loss": 0.0, "perplexity": 1.0, "num_samples": 0}

    model.eval()
    if device is None:
        device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0
    loss_fct = torch.nn.CrossEntropyLoss(reduction="sum", ignore_index=-100)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encodings = tokenizer(
            batch_texts,
            truncation=True,
            max_length=max_length,
            padding=True,
            return_tensors="pt"
        ).to(device)

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits

            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()

            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            num_valid = (shift_labels != -100).sum().item()

            total_loss += loss.item()
            total_tokens += num_valid

    avg_loss = total_loss / max(total_tokens, 1)
    ppl = math.exp(min(avg_loss, 100.0)) # Guard against float overflow

    logger.info("Evaluation Perplexity: %.4f (Avg Loss: %.4f, Tokens: %d)", ppl, avg_loss, total_tokens)
    return {
        "loss": round(avg_loss, 4),
        "perplexity": round(ppl, 4),
        "total_tokens": total_tokens,
        "num_samples": len(texts),
    }
