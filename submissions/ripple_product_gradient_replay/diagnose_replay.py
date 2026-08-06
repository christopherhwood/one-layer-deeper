"""Check that detached activation replay exactly decomposes endpoint gradients."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import torch

from benchmark import ModelSpec, TokenLossBatch
from benchmark.batches import prepare_batch
from benchmark.manifest import load_manifest
from data import infer_max_seq_len, infer_vocab_size, make_dataloaders


ROOT = Path(__file__).resolve().parents[2]
SUBMISSION_PATH = Path(__file__).with_name("submission.py")


def _load_submission_module():
    spec = importlib.util.spec_from_file_location("gradient_replay_probe", SUBMISSION_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load submission module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _loss_batch(module, model, batch):
    input_ids, labels, attention_mask, target_positions = prepare_batch(
        batch, torch.device("cpu")
    )
    logits, auxiliary = model(input_ids, attention_mask=attention_mask)
    if target_positions is None:
        token_logits = logits[:, :-1].float()
        token_labels = labels[:, 1:]
    else:
        rows = torch.arange(logits.shape[0])[:, None]
        token_logits = logits[rows, target_positions.clamp_min(0)].float()
        token_labels = labels
    valid = token_labels != -100
    loss = module.token_training_loss(
        TokenLossBatch(
            logits=token_logits,
            labels=token_labels,
            valid_mask=valid,
            target_positions=target_positions,
            auxiliary=auxiliary,
        )
    )
    return loss, logits, auxiliary


def _t1_batch(model, dataloader):
    rows = []
    for batch in dataloader:
        input_ids, _, attention_mask, _ = prepare_batch(batch, torch.device("cpu"))
        with torch.no_grad():
            _, _, t_values = model._parse(input_ids, attention_mask)
        selected = torch.nonzero(t_values == 1).flatten()
        for index in selected.tolist():
            rows.append({name: value[index] for name, value in batch.items()})
            if len(rows) == 8:
                return {
                    name: torch.stack([row[name] for row in rows])
                    for name in rows[0]
                }
    raise RuntimeError("generated training data did not provide eight T=1 rows")


def main() -> None:
    torch.manual_seed(74)
    module = _load_submission_module()
    manifest = load_manifest(ROOT / "benchmark/manifests/local_cpu_10s.json")
    model_spec = ModelSpec(
        vocab_size=infer_vocab_size(manifest.data),
        max_seq_len=infer_max_seq_len(manifest.data),
        maximum_model_state_elements=manifest.model_state.maximum_elements,
    )
    model = module.build_model(model_spec)
    model.train()
    model.normalize_replay = False
    batch = _t1_batch(model, make_dataloaders(manifest.data)["train"])

    global_loss, global_logits, _ = _loss_batch(module, model, batch)
    global_loss.backward()
    state_gradients = tuple(
        state.grad.detach().clone() for state in model._gradient_states
    )
    global_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.square.named_parameters()
        if parameter.grad is not None
    }

    model.zero_grad(set_to_none=True)
    model._saved_state_gradients = state_gradients
    model.loss_mode = "replay"
    replay_loss, replay_logits, _ = _loss_batch(module, model, batch)
    replay_loss.backward()

    excluded_prefixes = (
        "output_norm.",
        "output.",
        "edge_norm.",
        "edge_head.",
        "start_scores",
        "end_scores",
    )
    dot = torch.zeros(())
    global_norm = torch.zeros(())
    replay_norm = torch.zeros(())
    maximum_error = 0.0
    compared = 0
    for name, parameter in model.square.named_parameters():
        if name.startswith(excluded_prefixes):
            continue
        expected = global_gradients[name]
        actual = parameter.grad
        if actual is None:
            raise RuntimeError(f"missing replay gradient for {name}")
        difference = (actual - expected).abs()
        maximum_error = max(maximum_error, float(difference.max()))
        dot += (actual * expected).sum()
        global_norm += expected.square().sum()
        replay_norm += actual.square().sum()
        compared += 1
    cosine = dot / (global_norm.sqrt() * replay_norm.sqrt()).clamp_min(1e-30)
    relative_l2 = (global_norm + replay_norm - 2.0 * dot).clamp_min(0).sqrt()
    relative_l2 = relative_l2 / global_norm.sqrt().clamp_min(1e-30)
    value_error = float((global_logits - replay_logits).abs().max().detach())

    print(f"compared_tensors={compared}")
    print(f"global_loss={float(global_loss):.9f}")
    print(f"replay_surrogate={float(replay_loss):.9f}")
    print(f"forward_max_abs_error={value_error:.9e}")
    print(f"gradient_cosine={float(cosine):.9f}")
    print(f"gradient_relative_l2={float(relative_l2):.9e}")
    print(f"gradient_max_abs_error={maximum_error:.9e}")
    if value_error > 1e-7 or float(cosine) < 0.999999 or float(relative_l2) > 1e-5:
        raise AssertionError("activation replay did not reproduce the endpoint gradient")


if __name__ == "__main__":
    main()
