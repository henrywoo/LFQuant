import torch
import torch.nn.functional as tnf
from torch.nn import Module
from enum import Enum
from typing import Tuple, Optional

from .codebook import book_entropy, generate_sub_book_t

FZERO = torch.tensor(0.0)


class LFQMode(Enum):
    VANILLA = 1
    X_BATCH = 2
    BLOCK = 3


class LFQ:
    def __init__(
        self,
        d: int = 26,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 0.005,
        eps: float = 1e-10,
        x_split: int = 16,
        debug: bool = False,
        mode_delimiter=(19, 26),
    ):
        self.d = d
        self.k = 2**d
        self.mode_delimiter = mode_delimiter
        self.temperature = temperature
        self.eps = eps
        self.indices: Optional[torch.Tensor] = None
        self.debug = debug
        self.alpha = alpha
        self.beta = beta
        self.commit_loss: Optional[torch.Tensor] = None
        self.book_t = None
        self.half_book = None
        self.mode = None
        self.subbook_size = None
        self.num_subbooks = None
        self.subbook_ranges = None
        self.x_split = x_split

    def init_books(self, device: torch.device):
        if self.d < self.mode_delimiter[0]:
            self.mode = LFQMode.VANILLA
            indices = torch.arange(self.k, device=device)
            book = (
                indices.unsqueeze(-1)
                .bitwise_right_shift(torch.arange(self.d - 1, -1, -1, device=device))
                .remainder(2)
            )
            book[book == 0] = -1
            self.book_t = book.float().t()
        elif self.d < self.mode_delimiter[1]:
            self.mode = LFQMode.X_BATCH
            indices = torch.arange(self.k, device=device)
            book = (
                indices.unsqueeze(-1)
                .bitwise_right_shift(torch.arange(self.d - 1, -1, -1, device=device))
                .remainder(2)
            )
            book[book == 0] = -1
            self.book_t = book.float().t()
        else:
            self.mode = LFQMode.BLOCK
            self.subbook_size = 2 ** (min(21, self.d) - 1)
            self.num_subbooks = self.k // self.subbook_size
            self.subbook_ranges = [
                (i * self.subbook_size, (i + 1) * self.subbook_size)
                for i in range(self.num_subbooks)
            ]

    def calc_entro_by_batch(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        assert (x.size(-2) % self.x_split) == 0
        self.mean_probs = torch.zeros(x.size(0), self.k, device=device)
        entro_mean = torch.zeros(x.size(0), device=device)

        num_splits = x.size(1) // self.x_split
        for i in range(num_splits):
            x_chunk = x[:, i * self.x_split : (i + 1) * self.x_split]
            logits = x_chunk.float() @ self.book_t
            l = logits / self.temperature
            probs = tnf.softmax(l, -1)
            log_probs = tnf.log_softmax(l + self.eps, -1)
            entropy = -torch.sum(probs * log_probs, -1)
            mean_probs = probs.mean(dim=tuple(range(probs.dim() - 1)))
            entro_mean += torch.sum(entropy, dim=-1) / x.size(-2)
            self.mean_probs += mean_probs
        self.mean_probs /= num_splits
        if self.debug:
            assert abs(self.mean_probs[0].sum().item()) < 1.1
        mean_entro = -torch.sum(
            self.mean_probs * torch.log(self.mean_probs + self.eps), -1
        )
        return entro_mean.mean(), mean_entro.mean()

    def calc_entro_vanilla(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return book_entropy(x, self.book_t, self.eps)

    def _block(
        self,
        x: torch.Tensor,
        i: int,
        sub_book_t: torch.Tensor,
        entro_mean: torch.Tensor,
        mean_probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x_chunk = x[:, i * self.x_split : (i + 1) * self.x_split]
        logits = x_chunk.float() @ sub_book_t
        probs = tnf.softmax(logits / self.temperature, -1)
        log_probs = tnf.log_softmax(logits / self.temperature + self.eps, -1)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        denorm = x.size(-2) * len(self.subbook_ranges)
        entro_mean += torch.sum(entropy, dim=-1) / denorm
        mean_probs += torch.sum(probs, dim=-2) / denorm
        del sub_book_t, logits, probs, log_probs, entropy, x_chunk
        torch.cuda.empty_cache()
        return entro_mean, mean_probs

    def calc_entro_block(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = x.device
        assert x.size(-2) % self.x_split == 0
        entro_mean = torch.zeros(x.size(0), device=device)
        mean_probs = torch.zeros(x.size(0), self.subbook_size, device=device)
        for start, end in self.subbook_ranges:
            sub_book_t = generate_sub_book_t(self.d, start, end, device)
            num_splits = x.size(-2) // self.x_split
            for i in range(num_splits):
                entro_mean, mean_probs = self._block(
                    x, i, sub_book_t, entro_mean, mean_probs
                )
        mean_entro = -torch.sum(mean_probs * torch.log(mean_probs + self.eps), -1)
        return entro_mean.mean(), mean_entro.mean()

    def run(
        self,
        x: torch.Tensor,
        return_indices: bool = False,
        training: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        device = x.device
        if self.mode is None:
            self.init_books(device)
        q = torch.where(x > 0, 1, -1).to(device)
        if not training:
            return (
                q,
                FZERO.to(device),
                FZERO.to(device),
                FZERO.to(device),
                FZERO.to(device),
                FZERO.to(device),
            )
        if self.mode == LFQMode.VANILLA:
            entro_mean, mean_entro = self.calc_entro_vanilla(x)
        elif self.mode == LFQMode.X_BATCH:
            entro_mean, mean_entro = self.calc_entro_by_batch(x)
        elif self.mode == LFQMode.BLOCK:
            entro_mean, mean_entro = self.calc_entro_block(x)
        else:
            raise ValueError("Wrong LFQMode")
        entro_loss = self.alpha * entro_mean - self.alpha * mean_entro

        q = x + (q - x).detach()
        self.commit_loss = tnf.mse_loss(x, q.detach(), reduction="none")

        if return_indices:
            binary_result = (q + 1) // 2
            powers_of_2 = (
                2
                ** torch.arange(
                    binary_result.size(-1) - 1, -1, -1, device=device
                ).float()
            )
            self.indices = (binary_result.float() @ powers_of_2).long()
        if self.debug:
            print(
                f"D: {self.d} EMean: {entro_mean:.2f}, MEntro: {mean_entro:.2f}, ELoss: {entro_loss:.2f}"
            )
        return (q,
                entro_mean,
                mean_entro,
                entro_loss,
                self.indices,
                self.commit_loss.mean())


class TorchLFQ(Module):
    def __init__(
        self,
        d: int = 26,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 0.005,
        eps: float = 1e-10,
        x_split: int = 16,
        debug: bool = False,
        mode_delimiter=(19, 26),
    ):
        super().__init__()
        self.lfq = LFQ(d, alpha, beta, temperature, eps, x_split, debug, mode_delimiter)

    def forward(
        self,
        x: torch.Tensor,
        return_indices: bool = False,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
    ]:
        return self.lfq.run(x, return_indices, self.training)
