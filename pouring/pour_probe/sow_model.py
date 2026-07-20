"""Standalone transcription of the *Sound of Water* audio-pitch model (Bagad et al.,
arXiv 2411.11222) — wav2vec2 + absolute-time encoding + 64-bin axial/radial wavelength
heads.

Why transcribe instead of importing `third_party/SoundOfWater`: their `model.py` imports
`pytorch_lightning` and `shared.utils` (which pulls librosa, matplotlib, decord-specific
helpers). We only need a frozen forward pass, so we reimplement the two `nn.Module`s
with **identical attribute names** and load their checkpoint with `strict=True`. The
strict load is the correctness gate: any renamed/missing parameter raises rather than
silently degrading (a wrong time encoding produces plausible-looking garbage, which
would poison every downstream experiment).

Config is taken from their released demo (`demo/util.py`): backbone defaults
(use_time=True, d=512, rate=49, scale_factor=0.01, layer_norm=False), axial/radial
heads = Linear(768, 64), softmax activation.

Checkpoint: checkpoints/sow/dsr9mf13_ep100_step12423_real_finetuned_with_cosupervision.pth

Usage:
    from sow_model import load_sow_model, load_audio, predict_axial
    model = load_sow_model()
    wav = load_audio("video.mp4")            # (1, NS) float32 @16 kHz, normalized
    lam, feats = predict_axial(model, wav)   # lam (F,) cm @49 fps, feats (F,768)

Run as a script for the correctness gate:
    .venv/bin/python pouring/pour_probe/sow_model.py --check <video.mp4>
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
CKPT = ROOT / "checkpoints/sow/dsr9mf13_ep100_step12423_real_finetuned_with_cosupervision.pth"
SR = 16000
RATE = 49.0          # backbone output frame rate (frames per second)
N_BINS = 64
W_MAX = 100.0        # wavelength bin centres span [0, 100] cm


class TimeEncodingDiscreteSinusoidal(nn.Module):
    """Sinusoidal encoding of ABSOLUTE time (seconds), discretised at `rate` fps.

    Parameter-free, but reproduced exactly: the model was trained with these vectors
    added to the CNN features, so any deviation shifts the input distribution.
    """

    def __init__(self, d=512, v=10000, rate=49, scale_factor=0.01):
        super().__init__()
        self.d, self.v, self.rate, self.scale_factor = d, v, rate, scale_factor

    def forward(self, t):
        # t: (B, N) seconds -> (B, N, d)
        B, N = t.shape
        i = (t * self.rate).to(int)
        div = torch.exp(torch.arange(0, self.d, 2, dtype=torch.float, device=t.device)
                        * -(math.log(self.v) / self.d))
        pe = torch.zeros(B, N, self.d, device=t.device)
        pe[:, :, 0::2] = torch.sin(i[:, :, None].float() * div)
        pe[:, :, 1::2] = torch.cos(i[:, :, None].float() * div)
        return pe * self.scale_factor


class Wav2Vec2WithTimeEncoding(nn.Module):
    """wav2vec2-base-960h with the time encoding injected between the convolutional
    feature extractor and the feature projection (i.e. the transformer sees *when*
    each frame occurred — that is how the model tracks a rising pitch over a pour)."""

    def __init__(self, model_name="facebook/wav2vec2-base-960h", use_time=True,
                 d=512, v=10000, rate=49, scale_factor=0.01, layer_norm=False):
        super().__init__()
        from transformers import Wav2Vec2Model
        self.net = Wav2Vec2Model.from_pretrained(model_name)
        self.d, self.v, self.rate, self.sr, self.use_time = d, v, rate, SR, use_time
        self.time_encoding = (TimeEncodingDiscreteSinusoidal(d, v, rate, scale_factor)
                              if use_time else None)
        self.layer_norm = nn.LayerNorm(d) if layer_norm else nn.Identity()

    def forward(self, x, t):
        """x: (B,T,1,NS) audio clips; t: (B,T,2) clip [start,end] seconds -> (B,T,F,768).

        NOTE: the reference implementation builds its dense timestamps with
        `for i in range(B)` while `t` has already been flattened to (B*T, 2) — correct
        only for a single clip per item, which is how their demo runs it. We iterate the
        flattened axis, which is equivalent at T=1 and correct for T>1.
        """
        B, T, C, NS = x.shape
        assert C == 1, "Require a single-channel input."
        x = x.reshape(B * T, NS)
        t = t.reshape(B * T, 2)

        feats = self.net.feature_extractor(x).transpose(1, 2)      # (BT,F,512)
        if self.use_time:
            F_ = feats.shape[1]
            t_dense = torch.stack([torch.linspace(float(s), float(e), F_) for s, e in t])
            feats = feats + self.time_encoding(t_dense.to(feats.device))
        feats = self.layer_norm(feats)
        hidden, _ = self.net.feature_projection(feats)
        z = self.net.encoder(hidden, attention_mask=None, output_attentions=False,
                             output_hidden_states=False, return_dict=True)[0]
        return z.reshape(B, T, z.shape[1], z.shape[2])


class WavelengthWithTime(nn.Module):
    """Backbone + softmax classification heads over `axial_bins` wavelength bins.

    Trained with a KL objective against a soft target distribution, so the *expectation*
    over bin centres (not the argmax) is the intended readout — see `axial_wavelength`.
    Only the plain-inference parts of the reference LightningModule are reproduced;
    the training/optimizer machinery is dropped.
    """

    def __init__(self, backbone, feat_dim=768, axial_bins=N_BINS, radial_bins=N_BINS):
        super().__init__()
        self.backbone = backbone
        self.feat_dim = feat_dim
        self.intermediate_layers = nn.Identity()      # matches the released config
        self.axial_head = nn.Linear(feat_dim, axial_bins)
        self.radial_head = nn.Linear(feat_dim, radial_bins)
        self.act = nn.Softmax(dim=-1)

    def forward(self, x, t):
        h = self.intermediate_layers(self.backbone(x, t))
        return {"axial": self.act(self.axial_head(h)),
                "radial": self.act(self.radial_head(h)),
                "feats": h}


def load_sow_model(ckpt=CKPT, device="cuda"):
    """Build the model and load their checkpoint with strict=True (the gate)."""
    model = WavelengthWithTime(Wav2Vec2WithTimeEncoding())
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    model.load_state_dict(sd, strict=True)        # must not be relaxed
    return model.eval().to(device)


def bin_centres(device="cpu"):
    return torch.linspace(0, W_MAX, N_BINS, device=device)


def axial_wavelength(probs):
    """(F,64) softmax probs -> (F,) wavelength in cm, as the distribution's expectation.

    Soft, not argmax: one bin is ~1.6 cm of wavelength (~0.4 cm of water level), so an
    argmax readout is a staircase whose time-derivative is a spike comb — useless for
    flow. The model is trained with a soft target, so the expectation is also the
    faithful readout.
    """
    return (probs.float().cpu() @ bin_centres()).numpy()


def load_audio(path, sr=SR):
    """Load a media file's audio as a (1, NS) float32 tensor, wav2vec2-normalized."""
    from decord import AudioReader
    ar = AudioReader(str(path), sample_rate=sr, mono=True)
    wav = np.asarray(ar[:].asnumpy()).reshape(-1).astype(np.float32)
    # Wav2Vec2Processor for base-960h = zero-mean / unit-variance waveform normalization
    wav = (wav - wav.mean()) / (wav.std() + 1e-7)
    return torch.from_numpy(wav)[None]


@torch.no_grad()
def predict_axial(model, wav, device="cuda", chunk_s=None):
    """(1,NS) waveform -> (lambda_cm (F,), feats (F,768)) at ~49 fps.

    The whole clip is passed as ONE window so the absolute-time encoding matches how the
    model was trained on full pours (their demo does the same). `chunk_s` splits long
    audio into sequential clips carrying their true start/end times, for memory.
    """
    NS = wav.shape[-1]
    dur = NS / SR
    if chunk_s is None:
        spans = [(0.0, dur)]
    else:
        spans = [(s, min(dur, s + chunk_s)) for s in np.arange(0, dur, chunk_s)]
    lams, feats = [], []
    for s, e in spans:
        seg = wav[:, int(s * SR):int(e * SR)]
        if seg.shape[-1] < SR // 10:
            continue
        x = seg.reshape(1, 1, 1, -1).to(device)              # (B=1,T=1,C=1,NS)
        t = torch.tensor([[[s, e]]], dtype=torch.float32)
        out = model(x, t)
        lams.append(axial_wavelength(out["axial"][0, 0]))
        feats.append(out["feats"][0, 0].float().cpu().numpy())
    return np.concatenate(lams), np.concatenate(feats)


def _check(video):
    """Correctness gate: strict load + lambda must DECREASE over a pour.

    Physics: as the vessel fills, the resonating air column shortens, so the resonant
    wavelength falls and the pitch rises. A model whose time encoding is wired wrong
    still emits smooth plausible curves — the monotone direction is the real test.
    """
    print(f"[gate] strict=True state-dict load ...")
    model = load_sow_model()
    print("[gate]   OK — no missing/unexpected keys")
    wav = load_audio(video)
    lam, feats = predict_axial(model, wav)
    n = len(lam)
    h1, h2 = lam[: n // 2].mean(), lam[n // 2:].mean()
    from scipy.stats import spearmanr
    rho = spearmanr(np.arange(n), lam).statistic
    print(f"[gate] {Path(video).name}: {n} frames ({n / RATE:.1f}s), "
          f"lambda {lam[0]:.1f} -> {lam[-1]:.1f} cm")
    print(f"[gate]   first-half mean {h1:.2f} cm, second-half mean {h2:.2f} cm")
    print(f"[gate]   Spearman(t, lambda) = {rho:+.3f}  "
          f"({'DECREASING as expected' if rho < -0.5 else 'NOT decreasing — investigate'})")
    print(f"[gate] feats {feats.shape}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="a pouring video/audio file")
    _check(ap.parse_args().check)
