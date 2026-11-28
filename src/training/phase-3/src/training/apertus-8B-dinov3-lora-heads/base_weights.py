"""Train selected decoder blocks through FP32 master weights and persist their updates."""

import re

import torch

PATCH_NAME = "apertus_blocks.pt"
LAYER = re.compile(r"(?:^|\.)layers\.(\d+)\.")


def block_parameters(decoder, depth, count, *, adapters=False):
    if not 0 <= count <= depth:
        raise ValueError(f"Cannot select {count} of {depth} decoder blocks.")
    selected = {}
    for name, parameter in decoder.named_parameters():
        match = LAYER.search(name)
        if match and depth - count <= int(match[1]) < depth and ("lora_" in name) == adapters:
            selected[name] = parameter
    if count and not adapters and {int(LAYER.search(k)[1]) for k in selected} != set(range(depth - count, depth)):
        raise ValueError("Some requested Apertus blocks were not found.")
    return selected


def restore_blocks(decoder, folder, settings):
    count = settings.get("unfrozen_blocks", 0)
    if not count:
        return
    expected = block_parameters(decoder, settings["layers"], count)
    state = torch.load(folder / PATCH_NAME, map_location="cpu", weights_only=True, mmap=True)
    if state.keys() != expected.keys():
        raise ValueError("Apertus checkpoint does not contain exactly the declared base block weights.")
    with torch.no_grad():
        for name, parameter in expected.items():
            value = state[name]
            if value.shape != parameter.shape or value.dtype != torch.float32:
                raise ValueError(f"Invalid FP32 block weight for {name}.")
            parameter.copy_(value)


class BaseMasters:
    def __init__(self, decoder, depth):
        self.decoder, self.depth = decoder, depth
        self.base, self.master = {}, {}

    def expand(self, count):
        requested = block_parameters(self.decoder, self.depth, count)
        if not self.base.keys() <= requested.keys():
            raise ValueError("Progressive unfreezing cannot discard an active block.")
        new = []
        for name, parameter in requested.items():
            if name in self.base:
                continue
            if parameter.requires_grad:
                raise ValueError(f"Base weight was already active: {name}.")
            parameter.requires_grad_(True)
            master = torch.nn.Parameter(parameter.detach().to(dtype=torch.float32, copy=True))
            self.base[name], self.master[name] = parameter, master
            new.append(master)
        for parameter in block_parameters(self.decoder, self.depth, count, adapters=True).values():
            parameter.requires_grad_(False)
            parameter.grad = None
        return new

    def transfer_gradients(self):
        for name, base in self.base.items():
            self.master[name].grad = None if base.grad is None else base.grad.detach().float()
            base.grad = None

    @torch.no_grad()
    def sync(self):
        for name, base in self.base.items():
            base.copy_(self.master[name])

    def save(self, folder):
        if self.master:
            torch.save({name: weight.detach().cpu() for name, weight in self.master.items()}, folder / PATCH_NAME)
