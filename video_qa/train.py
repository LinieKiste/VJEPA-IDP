"""Three-stage visual instruction tuning for V-JEPA 2 + LLM (Appendix E).

Usage:
    python video_qa/train.py --config video_qa/configs/stage1.yaml
    python video_qa/train.py --config video_qa/configs/stage1.yaml --dry_run

Stage 1 trains only the projector; stages 2-3 additionally train LoRA adapters
on the LLM. Each stage loads the projector (and LoRA) from the previous stage
via the `projector.init_from` / `llm.lora_init_from` config keys.
"""

import argparse
import math
import os
import sys

# This machine's torch is built for CUDA 13.2 (cu132), but bitsandbytes only
# ships prebuilt binaries up to cuda130. CUDA 13.x minor versions are
# compatible, so point bitsandbytes at its cuda130 binary. Must be set before
# bitsandbytes is first imported. Remove once a cu132 bnb wheel exists.
os.environ.setdefault("BNB_CUDA_VERSION", "130")

import torch
import yaml
from box import Box
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collate import collate_fn
from dataset import VideoQADataset
from model import ENCODER_EMBED_DIM, Projector, VideoQAModel, build_encoder

_VJEPA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vjepa2")
if _VJEPA_ROOT not in sys.path:
    sys.path.insert(0, _VJEPA_ROOT)
from src.utils.logging import AverageMeter, CSVLogger  # noqa: E402


def build_llm(cfg, device):
    tokenizer = AutoTokenizer.from_pretrained(cfg.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_cfg = None
    if cfg.get("load_in_4bit", True):
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    llm = AutoModelForCausalLM.from_pretrained(
        cfg.name,
        quantization_config=quant_cfg,
        dtype=torch.bfloat16,
        device_map={"": device} if quant_cfg else None,
    )
    llm.config.pad_token_id = tokenizer.pad_token_id

    if cfg.get("use_lora", False):
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

        if quant_cfg:
            llm = prepare_model_for_kbit_training(llm, use_gradient_checkpointing=True)
        lora = LoraConfig(
            r=cfg.get("lora_r", 64),
            lora_alpha=cfg.get("lora_alpha", 128),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        llm = get_peft_model(llm, lora)
        if cfg.get("lora_init_from"):
            from peft import set_peft_model_state_dict
            sd = torch.load(cfg.lora_init_from, map_location="cpu")
            set_peft_model_state_dict(llm, sd)
        llm.print_trainable_parameters()
    else:
        for p in llm.parameters():
            p.requires_grad = False

    return llm, tokenizer


def build_model(config, device):
    enc_cfg = config.encoder
    encoder = build_encoder(
        enc_cfg.arch, enc_cfg.checkpoint, enc_cfg.img_size, enc_cfg.num_frames,
    ).to(device).to(torch.bfloat16)

    llm, tokenizer = build_llm(config.llm, device)

    projector = Projector(
        in_dim=ENCODER_EMBED_DIM[enc_cfg.arch],
        hidden_dim=config.projector.hidden_dim,
        out_dim=llm.config.hidden_size,
    ).to(device)
    if config.projector.get("init_from"):
        projector.load_state_dict(torch.load(config.projector.init_from, map_location="cpu"))

    model = VideoQAModel(
        encoder, projector, llm,
        spatial_pool_stride=enc_cfg.spatial_pool_stride,
    ).to(device)
    return model, tokenizer


def run_dry(model, tokenizer, config, device):
    """One forward pass on a synthetic batch — no dataset files required.

    Reuses the real tokenization (VideoQADataset._tokenize) and collator so the
    splice path is exercised exactly as in training, but with random pixels.
    """
    enc = config.encoder
    ds = VideoQADataset.__new__(VideoQADataset)
    ds.tok, ds.system_prompt = tokenizer, "You are a helpful assistant."

    convs = [
        [{"from": "human", "value": "<video>\nWhat is the person doing?"},
         {"from": "gpt", "value": "The person is pouring water into a glass."}],
        [{"from": "human", "value": "<video>\nDescribe the scene briefly."},
         {"from": "gpt", "value": "A kitchen counter with someone making tea."}],
    ]
    samples = []
    for c in convs:
        ids, labels = ds._tokenize(c)
        samples.append({
            "input_ids": ids,
            "labels": labels,
            "pixel_values": torch.randn(3, enc.num_frames, enc.img_size, enc.img_size),
        })
    batch = {k: v.to(device) for k, v in collate_fn(samples).items()}

    model.eval()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = model(**batch)
    print(f"[dry_run] input_ids {tuple(batch['input_ids'].shape)}, "
          f"pixel_values {tuple(batch['pixel_values'].shape)}, "
          f"loss {out.loss.item():.4f}")
    print("[dry_run] OK — encoder, projector, and LLM forward all ran.")


def cosine_warmup(step, total, warmup):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.5 * (1 + math.cos(math.pi * progress))


def main(config_path, dry_run=False):
    with open(config_path) as f:
        config = Box(yaml.safe_load(f))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(config.output_dir, exist_ok=True)

    model, tokenizer = build_model(config, device)

    # Dry run: validate encoder -> projector -> LLM forward on a synthetic batch
    # without needing the dataset to be downloaded yet.
    if dry_run:
        run_dry(model, tokenizer, config, device)
        return

    dataset = VideoQADataset(
        json_path=config.data.json,
        media_root=config.data.media_root,
        tokenizer=tokenizer,
        num_frames=config.encoder.num_frames,
        frame_step=config.encoder.frame_step,
        img_size=config.encoder.img_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.optim.batch_size,
        shuffle=True,
        num_workers=config.optim.get("num_workers", 4),
        collate_fn=collate_fn,
        drop_last=True,
    )

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in trainable)
    print(f"Trainable params: {n_params/1e6:.1f}M")
    optimizer = torch.optim.AdamW(
        trainable, lr=config.optim.lr, weight_decay=config.optim.get("weight_decay", 0.0)
    )

    grad_accum = config.optim.get("grad_accum", 1)
    total_steps = (len(loader) // grad_accum) * config.optim.epochs
    warmup = int(total_steps * config.optim.get("warmup_ratio", 0.03))

    logger = CSVLogger(os.path.join(config.output_dir, "train_log.csv"),
                       ("%d", "step"), ("%.5f", "loss"), ("%.6e", "lr"))
    loss_meter = AverageMeter()

    model.train()
    model.encoder.eval()
    step = 0
    for epoch in range(config.optim.epochs):
        for it, batch in enumerate(loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
                loss = out.loss / grad_accum
            loss.backward()
            loss_meter.update(out.loss.item())

            if (it + 1) % grad_accum == 0:
                lr = config.optim.lr * cosine_warmup(step, total_steps, warmup)
                for g in optimizer.param_groups:
                    g["lr"] = lr
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % config.optim.get("log_every", 10) == 0:
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"loss {loss_meter.avg:.4f} lr {lr:.2e}")
                    logger.log(step, loss_meter.avg, lr)
                    loss_meter = AverageMeter()

    save_checkpoint(model, config)


def save_checkpoint(model, config):
    out = config.output_dir
    torch.save(model.projector.state_dict(), os.path.join(out, "projector.pt"))
    if config.llm.get("use_lora", False):
        from peft import get_peft_model_state_dict
        torch.save(get_peft_model_state_dict(model.llm),
                   os.path.join(out, "lora_adapters.pt"))
    print(f"Saved checkpoint to {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    main(args.config, args.dry_run)
