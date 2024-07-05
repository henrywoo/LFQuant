import torch
import torch.nn.functional as tnf


def book_entropy(x: torch.Tensor, book_t, _eps: float = 1e-10):
    logits = x @ book_t
    probs = tnf.softmax(logits, -1)
    log_probs = tnf.log_softmax(logits + _eps, -1)
    entropy = -torch.sum(probs * log_probs, -1)

    entro_mean = torch.mean(entropy)
    mean_probs = probs.mean(dim=tuple(range(probs.dim() - 1)))
    mean_entro = -torch.sum(mean_probs * torch.log(mean_probs + _eps))
    return entro_mean, mean_entro


def generate_sub_book_t(
    d: int, start: int, end: int, device: torch.device
) -> torch.Tensor:
    sub_indices = torch.arange(start, end, device=device)
    sub_book = (
        sub_indices.unsqueeze(0)
        .bitwise_right_shift(torch.arange(d - 1, -1, -1, device=device).unsqueeze(1))
        .remainder(2)
    )
    sub_book[sub_book == 0] = -1
    return sub_book.float()