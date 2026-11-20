"""
CTC-based word/line recognizer.
Paper 1 uses ResNet + CTC (no STN/BiLSTM for efficiency).
We implement CNN+CTC baseline and optional BiLSTM variant.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class _ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(ch),
        )

    def forward(self, x):
        return F.relu(self.block(x) + x)


class RecognizerCNN(nn.Module):
    """
    CNN feature extractor for text recognition.
    Input: (B, 1, H=32, W) grayscale word crop.
    Output: (B, C, 1, W') feature sequence where W' = W/4.
    """
    def __init__(self, out_channels: int = 256):
        super().__init__()
        self.features = nn.Sequential(
            # Stage 1: 32 -> 16
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            _ResBlock(32),
            # Stage 2: 16 -> 8
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            _ResBlock(64),
            # Stage 3: 8 -> 4
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1), (2, 1)),  # Only reduce height
            _ResBlock(128),
            # Stage 4: 4 -> 1
            nn.Conv2d(128, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, None)),  # Collapse height to 1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)  # (B, C, 1, W')


class CTCRecognizer(nn.Module):
    """
    Full CTC-based recognizer: CNN backbone + optional BiLSTM + linear classifier.

    Args:
        num_classes: vocabulary size + 1 (for CTC blank at index 0).
        use_bilstm: if True, add a BiLSTM sequence module.
        bilstm_hidden: BiLSTM hidden size per direction.
        cnn_channels: CNN output channel count.
    """
    def __init__(self, num_classes: int, use_bilstm: bool = False,
                 bilstm_hidden: int = 256, cnn_channels: int = 256):
        super().__init__()
        self.num_classes = num_classes
        self.use_bilstm = use_bilstm

        self.cnn = RecognizerCNN(out_channels=cnn_channels)

        if use_bilstm:
            self.bilstm = nn.LSTM(
                cnn_channels, bilstm_hidden, num_layers=2,
                bidirectional=True, batch_first=True
            )
            self.classifier = nn.Linear(bilstm_hidden * 2, num_classes)
        else:
            self.bilstm = None
            self.classifier = nn.Linear(cnn_channels, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, 1, H, W) grayscale image. H should be 32.
        Returns:
            logits: (T, B, num_classes) for CTC loss, where T = W/4 approx.
        """
        feats = self.cnn(x)  # (B, C, 1, W')
        feats = feats.squeeze(2)  # (B, C, W')
        feats = feats.permute(0, 2, 1)  # (B, W', C) = (B, T, C)

        if self.bilstm is not None:
            feats, _ = self.bilstm(feats)  # (B, T, 2*hidden)

        logits = self.classifier(feats)  # (B, T, num_classes)
        logits = logits.permute(1, 0, 2)  # (T, B, num_classes) for CTC
        return logits

    def decode_greedy(self, logits: torch.Tensor) -> list:
        """
        Greedy CTC decoding.
        Args:
            logits: (T, B, num_classes)
        Returns:
            list of lists of class indices (without blank/repeats).
        """
        preds = logits.argmax(dim=2)  # (T, B)
        batch_results = []
        for b in range(preds.shape[1]):
            seq = preds[:, b].tolist()
            decoded = []
            prev = -1
            for idx in seq:
                if idx != 0 and idx != prev:
                    decoded.append(idx)
                prev = idx
            batch_results.append(decoded)
        return batch_results

    def decode_beam_search(self, logits: torch.Tensor, beam_width: int = 10) -> list:
        """
        Prefix beam search decoding for CTC.
        Args:
            logits: (T, B, num_classes)
            beam_width: number of top hypotheses to maintain
        Returns:
            list of lists of class indices
        """
        import numpy as np
        log_probs = torch.log_softmax(logits, dim=-1).detach().cpu().numpy()  # (T, B, C)
        T, B, C = log_probs.shape
        batch_results = []

        for b in range(B):
            beams = {(): (0.0, -float('inf'))}
            for t in range(T):
                step_probs = log_probs[t, b]
                new_beams = {}
                top_indices = np.argsort(step_probs)[-min(beam_width * 2, C):]

                for prefix, (p_b, p_nb) in beams.items():
                    p_tot = np.logaddexp(p_b, p_nb)
                    # Extend with blank (index 0)
                    prob_blank = float(step_probs[0])
                    cur_b, cur_nb = new_beams.get(prefix, (-float('inf'), -float('inf')))
                    new_beams[prefix] = (np.logaddexp(cur_b, p_tot + prob_blank), cur_nb)

                    # Extend with non-blank
                    for c in top_indices:
                        if c == 0:
                            continue
                        prob_c = float(step_probs[c])
                        if prefix and prefix[-1] == c:
                            new_p = prefix + (int(c),)
                            cur_b, cur_nb = new_beams.get(new_p, (-float('inf'), -float('inf')))
                            new_beams[new_p] = (cur_b, np.logaddexp(cur_nb, p_b + prob_c))

                            cur_b, cur_nb = new_beams.get(prefix, (-float('inf'), -float('inf')))
                            new_beams[prefix] = (cur_b, np.logaddexp(cur_nb, p_nb + prob_c))
                        else:
                            new_p = prefix + (int(c),)
                            cur_b, cur_nb = new_beams.get(new_p, (-float('inf'), -float('inf')))
                            new_beams[new_p] = (cur_b, np.logaddexp(cur_nb, p_tot + prob_c))

                sorted_beams = sorted(
                    new_beams.items(),
                    key=lambda item: np.logaddexp(item[1][0], item[1][1]),
                    reverse=True
                )
                beams = dict(sorted_beams[:beam_width])

            best = max(beams.keys(), key=lambda p: np.logaddexp(beams[p][0], beams[p][1])) if beams else ()
            batch_results.append(list(best))

        return batch_results


class Vocabulary:
    """Maps characters to indices and back. Index 0 is CTC blank."""
    def __init__(self, charset: str):
        self.blank = 0
        self.chars = list(charset)
        self._char_to_idx = {c: i + 1 for i, c in enumerate(self.chars)}
        self._idx_to_char = {i + 1: c for i, c in enumerate(self.chars)}
        self.num_classes = len(self.chars) + 1  # +1 for blank

    def encode(self, text: str) -> list:
        """Encode text string to list of class indices."""
        return [self._char_to_idx.get(c, 0) for c in text]

    def decode(self, indices: list) -> str:
        """Decode list of class indices to text string."""
        return "".join(self._idx_to_char.get(i, "") for i in indices)
