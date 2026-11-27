"""QLoRA of Apertus v1.5 8B on the precomputed VQ codes, for two T4 GPUs (fp16, no bf16 on T4).

    torchrun --nproc_per_node 2 train.py --model-dir M --data-dir D --codes codes --out out --deadline UNIX_TIME

What it does:
- phase 1 of src/AGENTS.md by default: trains on common.splits.training_rows, never on the coverage set;
- loss only on the answer, and only on the class values of human labels (prelabels are -100);
- rare classes repeated (up to 3x), left/right perspectives mirrored on images without text;
- before training, measures the activations in fp16 and scales the MLP if they would overflow;
- after a few steps, measures the step time and fits the number of steps into the time left before --deadline,
  so that the run ends, with its adapters saved, inside the Kaggle quota;
- saves the LoRA adapters several times (for checkpoint averaging) and at the end.
"""

import argparse
import json
import math
import os
import pickle
import shutil
import time
import types
from pathlib import Path

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LambdaLR
from transformers import Trainer, TrainerCallback, TrainingArguments

import apertus_lora as L

LORA_TARGETS = r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|up_proj|down_proj)$"
FP16_SAFE = 2.0e4  # largest activation we let into an fp16 matmul (fp16 max is 65504)


def log(*parts):
    if int(os.environ.get("RANK", 0)) == 0:
        print(time.strftime("%H:%M:%S"), *parts, flush=True)


def broadcast_int(value):
    if dist.is_available() and dist.is_initialized():
        tensor = torch.tensor([int(value)], device=torch.device("cuda", torch.cuda.current_device()))
        dist.broadcast(tensor, 0)
        return int(tensor.item())
    return int(value)


# ---------------------------------------------------------------- model


def load_model(args, local_rank):
    from transformers import BitsAndBytesConfig
    from transformers.models.apertus1p5.modeling_apertus1p5 import Apertus1p5ForConditionalGeneration

    if args.tiny:  # a random miniature Apertus, to test the plumbing on a CPU
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(args.model_dir)
        config.text_config.update({"hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
                                   "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16})
        model = Apertus1p5ForConditionalGeneration(config)
        model.model.vision_tokenizer = None
        model.model.audio_tokenizer = None
        return model.to(torch.float32)
    if args.precision == "bf16":  # A100/L40S/H100: the 17 GB model as it was trained, no quantization
        model = Apertus1p5ForConditionalGeneration.from_pretrained(
            args.model_dir, dtype=torch.bfloat16, device_map={"": local_rank}, attn_implementation=L.ATTENTION
        )
        model.model.vision_tokenizer = None
        model.model.audio_tokenizer = None
        torch.cuda.empty_cache()
        return model
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.float16,
        llm_int8_skip_modules=["lm_head", "vision_tokenizer", "audio_tokenizer", "embed_tokens"],
    )
    model = Apertus1p5ForConditionalGeneration.from_pretrained(
        args.model_dir,
        quantization_config=quantization,
        dtype=torch.float16,
        device_map={"": local_rank},
        attn_implementation=L.ATTENTION,
    )
    # The codes are precomputed: the image and audio tokenizers are never called (1.2 GB per GPU freed).
    model.model.vision_tokenizer = None
    model.model.audio_tokenizer = None
    for module in model.modules():  # xIELU keeps its scalars in the bf16 of the config; T4 has no bf16
        if type(module).__name__ == "XIELUActivation":
            module.to(torch.float16)
    torch.cuda.empty_cache()
    return model


def patch_mlp(model, scale_holder):
    """xIELU in fp32, and the MLP output computed as scale * down(act / scale) so no fp16 value overflows.

    down_proj is linear, so the result is the same as down(act) whenever nothing overflows.
    """

    def forward(self, x):
        activation = self.act_fn(self.up_proj(x).float())
        scale = scale_holder["scale"]
        if scale != 1.0:
            activation = activation / scale
        out = self.down_proj(activation.to(x.dtype))
        return out * scale if scale != 1.0 else out

    count = 0
    for module in model.modules():
        if type(module).__name__ == "Apertus1p5TextMLP":
            module.forward = types.MethodType(forward, module)
            count += 1
    assert count, "no Apertus1p5TextMLP found to patch"
    return count


@torch.no_grad()
def calibrate(model, examples, pad_id, device, autocast, dtype=torch.float16):
    """Largest |activation| after xIELU and largest |hidden state| over a few real examples, and the loss."""
    peaks = {"act": 0.0, "hidden": 0.0}
    hooks = []
    for module in model.modules():
        name = type(module).__name__
        if name == "XIELUActivation":
            hooks.append(module.register_forward_hook(
                lambda m, i, o: peaks.__setitem__("act", max(peaks["act"], float(o.abs().max())))))
        elif name == "Apertus1p5TextDecoderLayer":
            hooks.append(module.register_forward_hook(
                lambda m, i, o: peaks.__setitem__(
                    "hidden", max(peaks["hidden"], float((o[0] if isinstance(o, tuple) else o).float().abs().max())))))
    losses = []
    for start in range(0, len(examples), 2):
        batch = {k: v.to(device) for k, v in L.collate(examples[start : start + 2], pad_id).items()}
        with torch.autocast("cuda", dtype=dtype, enabled=autocast):
            losses.append(answer_loss(model, batch).item())
    for hook in hooks:
        hook.remove()
    return peaks, losses


def answer_loss(model, batch):
    """Cross entropy on the answer tokens only. Left padding puts every answer at the end, so only the last
    `keep` logits are computed instead of seq_len x 131,072."""
    labels = batch["labels"]
    length = labels.shape[1]
    first = int((labels != -100).float().argmax(dim=1).min())
    keep = length - first + 1
    # With labels the model shifts them itself and keeps the compact text-only head (131,072 logits).
    return model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        position_ids=batch["position_ids"],
        labels=labels[:, length - keep :],
        logits_to_keep=keep,
        use_cache=False,
    ).loss


def upload(repo, folder, path_in_repo):
    """Copies a folder to the private model repo, so a crash of the machine loses nothing. Never fatal."""
    if not repo:
        return
    try:
        from huggingface_hub import HfApi

        # README.md: the model card PEFT writes carries the local base-model path, which the Hub rejects as
        # invalid metadata (2026-11-27: the final adapter of prototype-256 was lost this way).
        HfApi().upload_folder(repo_id=repo, folder_path=str(folder), path_in_repo=path_in_repo,
                              ignore_patterns=["README.md"], commit_message=f"upload {path_in_repo}")
        log(f"uploaded {path_in_repo} to {repo}")
    except Exception as error:  # the run goes on; the files stay on the machine
        log(f"upload of {path_in_repo} failed: {type(error).__name__}: {str(error)[:200]}")


# ---------------------------------------------------------------- trainer


class AnswerTrainer(Trainer):
    def __init__(self, *args, lr_state=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_accepts_loss_kwargs = False  # we return the mean loss; the Trainer divides by accumulation
        self.lr_state = lr_state

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        loss = answer_loss(model, inputs)
        return (loss, None) if return_outputs else loss

    def create_scheduler(self, num_training_steps, optimizer=None):
        """Linear warmup, then cosine down to 0 at lr_state["total"], which the budget callback can lower."""
        state = self.lr_state

        def factor(step):
            if step < state["warmup"]:
                return (step + 1) / state["warmup"]
            progress = min(1.0, (step - state["warmup"]) / max(1, state["total"] - state["warmup"]))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        if self.lr_scheduler is None:
            self.lr_scheduler = LambdaLR(optimizer or self.optimizer, factor)
        return self.lr_scheduler


class BudgetCallback(TrainerCallback):
    """Fits the run into the time before the deadline, stops on a broken loss, and keeps adapter snapshots."""

    def __init__(self, args, lr_state, steps_per_epoch, out_dir):
        self.deadline = args.deadline
        self.reserve = args.reserve_minutes * 60
        self.lr_state = lr_state
        self.steps_per_epoch = steps_per_epoch
        self.out_dir = out_dir
        self.save_every = None
        self.times = {}
        self.bad_logs = 0
        self.probe = (5, 25)
        self.min_save_every = args.min_save_every
        self.upload_repo = args.upload_repo
        self.upload_prefix = args.upload_prefix

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        self.times[step] = time.time()
        if step == self.probe[1] and self.probe[0] in self.times:
            per_step = (self.times[step] - self.times[self.probe[0]]) / (self.probe[1] - self.probe[0])
            left = self.deadline - time.time() - self.reserve
            fit = step + int(left / per_step) if self.deadline else self.lr_state["total"]
            total = broadcast_int(max(step + 1, min(self.lr_state["total"], fit)))
            self.lr_state["total"] = total
            half_epoch = max(1, self.steps_per_epoch // 2)
            self.save_every = max(self.min_save_every, min(half_epoch, total // 4))
            log(f"{per_step:.2f} s per step; {total} steps fit the time left "
                f"({total / self.steps_per_epoch:.2f} epochs); adapter snapshot every {self.save_every} steps")
        stop = step >= self.lr_state["total"] or (self.deadline and time.time() > self.deadline - self.reserve)
        if broadcast_int(stop):
            control.should_training_stop = True
        elif self.save_every and step % self.save_every == 0:
            control.should_save = True

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            source = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            target = self.out_dir / "adapters" / f"step-{state.global_step:05d}"
            target.mkdir(parents=True, exist_ok=True)
            for name in ("adapter_model.safetensors", "adapter_config.json"):
                if (source / name).exists():
                    shutil.copy2(source / name, target / name)
            log(f"adapter snapshot {target}")
            upload(self.upload_repo, target, f"{self.upload_prefix}/adapters/{target.name}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        loss = (logs or {}).get("loss")
        if loss is not None:
            self.bad_logs = self.bad_logs + 1 if not math.isfinite(loss) else 0
            if self.bad_logs >= 3:
                log("the loss is not finite three logs in a row: stopping")
                control.should_training_stop = True


# ---------------------------------------------------------------- main


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--codes", required=True, help="folder with the codes_*.pkl files of encode.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--part", default="training", choices=("training", "all", "prototype_train"))
    parser.add_argument("--side", type=int, default=256)
    parser.add_argument("--epochs", type=float, default=3.0, help="upper bound; the deadline may cut it")
    parser.add_argument("--deadline", type=float, default=0.0, help="unix time by which the adapters must be saved")
    parser.add_argument("--reserve-minutes", type=float, default=15.0)
    parser.add_argument("--max-steps", type=int, default=0, help="for a quick test")
    parser.add_argument("--limit", type=int, default=0, help="only the first N rows, for a quick test")
    parser.add_argument("--min-save-every", type=int, default=50)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--batch", type=int, default=4, help="per GPU")
    parser.add_argument("--accumulation", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--no-context", action="store_true")
    parser.add_argument("--precision", default="fp16-nf4", choices=("fp16-nf4", "bf16"),
                        help="fp16-nf4 for T4 (QLoRA), bf16 for A100/L40S/H100 (plain LoRA)")
    parser.add_argument("--no-grad-ckpt", action="store_true", help="faster when the GPU memory allows it")
    parser.add_argument("--upload-repo", default="", help="private HF model repo that receives every snapshot")
    parser.add_argument("--upload-prefix", default="run")
    parser.add_argument("--tiny", action="store_true", help="random miniature model on CPU, to test the code")
    args = parser.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoConfig, AutoTokenizer, set_seed

    set_seed(args.seed)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    cuda = torch.cuda.is_available() and not args.tiny
    if cuda:
        torch.cuda.set_device(local_rank)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # data
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    config = AutoConfig.from_pretrained(args.model_dir)
    sequences = L.Sequences(tokenizer, config.image_token_offset)
    codes = {}
    for path in sorted(Path(args.codes).glob(f"codes_{args.part}_{args.side}_shard*.pkl")):
        codes.update(pickle.loads(path.read_bytes()))
    rows = L.select_rows(L.read_train(args.data_dir), args.part)
    if args.limit:
        rows = rows.head(args.limit)
    records = [r for r in rows.to_dict("records") if r["image_id"] in codes]
    missing = len(rows) - len(records)
    records_ok = [r for r in records if not L.is_mirrorable(r) or L.code_key(r["image_id"], True) in codes]
    examples = L.build_examples(records_ok, codes, sequences, use_context=not args.no_context)
    lengths = sorted(len(e["input_ids"]) for e in examples)
    log(f"{len(rows)} rows ({args.part}), {missing} without codes, {len(examples)} examples per epoch, "
        f"length median {lengths[len(lengths) // 2]} max {lengths[-1]}")

    # model
    model = load_model(args, local_rank)
    scale_holder = {"scale": 1.0}
    half = torch.float16 if args.precision == "fp16-nf4" else torch.bfloat16
    if args.precision == "fp16-nf4":
        log(f"patched {patch_mlp(model, scale_holder)} MLPs")
    model.config.use_cache = False
    if not args.no_grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()

    device = torch.device("cuda", local_rank) if cuda else torch.device("cpu")
    probe = examples[:: max(1, len(examples) // 8)][:8]
    peaks, losses = calibrate(model, probe, tokenizer.pad_token_id, device, autocast=cuda, dtype=half)
    if not math.isfinite(peaks["act"]):
        raise SystemExit("fp16 overflow already inside up_proj: the T4 cannot run this model in fp16 as it is")
    if peaks["act"] > FP16_SAFE and args.precision == "fp16-nf4":
        scale_holder["scale"] = float(2 ** math.ceil(math.log2(peaks["act"] / FP16_SAFE)))
        _, losses = calibrate(model, probe, tokenizer.pad_token_id, device, autocast=cuda, dtype=half)
    log(f"fp16 check: max |xIELU| {peaks['act']:.0f}, max |hidden| {peaks['hidden']:.0f}, MLP scale "
        f"{scale_holder['scale']}, loss before training {sum(losses) / len(losses):.3f}")
    if not all(math.isfinite(x) for x in losses) or (args.precision == "fp16-nf4" and peaks["hidden"] > 6.0e4):
        raise SystemExit("fp16 overflow before training: the T4 cannot run this model in fp16 as it is")

    lora = LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=0.0, bias="none", target_modules=LORA_TARGETS)
    model = get_peft_model(model, lora)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert all(p.dtype == torch.float32 for p in model.parameters() if p.requires_grad), "LoRA weights must be fp32"
    log(f"LoRA r{args.rank}: {trainable / 1e6:.1f} M trainable parameters")

    steps_per_epoch = math.ceil(len(examples) / (args.batch * args.accumulation * world))
    nominal = args.max_steps or math.ceil(steps_per_epoch * args.epochs)
    lr_state = {"total": nominal, "warmup": max(1, round(args.warmup_ratio * min(nominal, steps_per_epoch * 2)))}
    log(f"{steps_per_epoch} steps per epoch, at most {nominal} steps, warmup {lr_state['warmup']}")

    training_args = TrainingArguments(
        output_dir=str(out_dir / "trainer"),
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.accumulation,
        num_train_epochs=math.ceil(args.epochs),
        max_steps=nominal,
        learning_rate=args.lr,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=cuda and args.precision == "fp16-nf4",
        bf16=cuda and args.precision == "bf16",
        logging_steps=10,
        save_strategy="no",
        save_total_limit=1,
        save_only_model=False,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
        label_names=["labels"],
        ddp_find_unused_parameters=False,
        disable_tqdm=True,
        use_cpu=not cuda,
    )
    trainer = AnswerTrainer(
        model=model,
        args=training_args,
        train_dataset=examples,
        data_collator=lambda batch: L.collate(batch, tokenizer.pad_token_id),
        lr_state=lr_state,
        callbacks=[BudgetCallback(args, lr_state, steps_per_epoch, out_dir)],
    )
    started = time.time()
    trainer.train()
    minutes = (time.time() - started) / 60

    if trainer.is_world_process_zero():
        final = out_dir / "adapters" / "final"
        model.save_pretrained(str(final))
        summary = {
            "part": args.part, "side": args.side, "rows": len(rows), "examples_per_epoch": len(examples),
            "steps": trainer.state.global_step, "steps_per_epoch": steps_per_epoch,
            "epochs": trainer.state.global_step / steps_per_epoch, "minutes": round(minutes, 1),
            "mlp_scale": scale_holder["scale"], "fp16_peaks": peaks, "loss_before": losses,
            "log_history": trainer.state.log_history, "args": vars(args), "system_prompt": L.SYSTEM_PROMPT,
            "peak_memory_gb": torch.cuda.max_memory_reserved() / 1024**3 if cuda else None,
        }
        (out_dir / "train_summary.json").write_text(json.dumps(summary, indent=2, default=str))
        upload(args.upload_repo, final, f"{args.upload_prefix}/adapters/final")
        log(f"done: {trainer.state.global_step} steps in {minutes:.0f} min, adapters in {final}")


if __name__ == "__main__":
    main()
