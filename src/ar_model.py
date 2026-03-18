"""MeshLex v2 — Autoregressive Transformer for patch sequence generation.

GPT-2 style decoder-only Transformer that models the joint distribution of
patch token sequences produced by `src.patch_sequence`.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchGPT(nn.Module):
    """GPT-2 style decoder-only Transformer for patch token sequences.

    Args:
        vocab_size: total token vocabulary (from compute_vocab_size).
        d_model: hidden dimension.
        n_heads: number of attention heads.
        n_layers: number of Transformer blocks.
        max_seq_len: maximum sequence length.
        dropout: dropout rate.
    """

    def __init__(
        self,
        vocab_size: int = 2000,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        max_seq_len: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.drop = nn.Dropout(dropout)

        # Transformer blocks
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Final layer norm + output head
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

        # Weight tying: head shares weights with token_emb
        self.head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)

    @staticmethod
    def _causal_mask(T: int, device: torch.device) -> torch.Tensor:
        """Generate causal (upper-triangular) attention mask.

        Returns a (T, T) bool tensor where True means *masked* (blocked).
        """
        return torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            tokens: (B, T) int64 token indices.

        Returns:
            (B, T, vocab_size) logits.
        """
        B, T = tokens.shape
        assert T <= self.max_seq_len, f"Sequence length {T} exceeds max {self.max_seq_len}"

        positions = torch.arange(T, device=tokens.device).unsqueeze(0)  # (1, T)
        x = self.token_emb(tokens) + self.pos_emb(positions)
        x = self.drop(x)

        mask = self._causal_mask(T, tokens.device)
        x = self.blocks(x, mask=mask)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab_size)
        return logits

    def compute_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Next-token prediction loss.

        Predicts tokens[1:] from tokens[:-1].

        Args:
            tokens: (B, T) int64 token indices.

        Returns:
            Scalar cross-entropy loss.
        """
        logits = self.forward(tokens)  # (B, T, V)
        # Shift: predict next token
        logits = logits[:, :-1].contiguous()  # (B, T-1, V)
        targets = tokens[:, 1:].contiguous()   # (B, T-1)
        loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1))
        return loss

    @torch.no_grad()
    def generate(
        self,
        max_len: int = 910,
        temperature: float = 1.0,
        top_k: int = 50,
        prompt: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Autoregressive sampling.

        Args:
            max_len: maximum number of tokens to generate.
            temperature: sampling temperature.
            top_k: top-k filtering (0 = no filtering).
            prompt: optional (1, T_prompt) tensor to condition on.

        Returns:
            (1, L) int64 generated token sequence, L <= max_len.
        """
        self.eval()
        device = next(self.parameters()).device

        if prompt is not None:
            tokens = prompt.to(device)
        else:
            # Start with random token from vocab
            tokens = torch.randint(0, self.vocab_size, (1, 1), device=device)

        for _ in range(max_len - tokens.shape[1]):
            # Truncate to max_seq_len if needed
            ctx = tokens[:, -self.max_seq_len:]
            logits = self.forward(ctx)  # (1, T, V)
            logits = logits[:, -1, :] / temperature  # (1, V)

            # Top-k filtering
            if top_k > 0:
                topk_vals, _ = logits.topk(top_k, dim=-1)
                threshold = topk_vals[:, -1].unsqueeze(-1)
                logits[logits < threshold] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            tokens = torch.cat([tokens, next_token], dim=1)

        return tokens
