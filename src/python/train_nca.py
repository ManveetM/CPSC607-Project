from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR

from nca_model import (
    NCAConfig,
    NeuralCellularAutomata,
    create_procedural_target,
    load_target_image,
    make_seed,
    pad_target,
    rgba_to_rgb,
)
from unity_export import build_unity_export_payload


class SamplePool:
    def __init__(self, seed_state: torch.Tensor, pool_size: int):
        self.x = seed_state.repeat(pool_size, 1, 1, 1)
        self.pool_size = pool_size

    def sample(self, batch_size: int) -> tuple[np.ndarray, torch.Tensor]:
        indices = np.random.choice(self.pool_size, batch_size, replace=False)
        return indices, self.x[indices].clone()

    def commit(self, indices: np.ndarray, batch: torch.Tensor) -> None:
        self.x[indices] = batch.detach()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a 2D Growing Neural Cellular Automata model.")
    parser.add_argument("--output-dir", default="outputs/growing_nca")
    parser.add_argument("--target-shape", choices=["disk", "ring", "diamond", "plus", "square"], default="disk")
    parser.add_argument("--target-image", default=None, help="Optional RGBA target image path.")
    parser.add_argument("--target-size", type=int, default=40)
    parser.add_argument("--target-padding", type=int, default=16)
    parser.add_argument("--channel-n", type=int, default=16)
    parser.add_argument("--hidden-n", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--pool-size", type=int, default=256)
    parser.add_argument("--train-steps", type=int, default=2000)
    parser.add_argument("--min-iter", type=int, default=48)
    parser.add_argument("--max-iter", type=int, default=96)
    parser.add_argument("--damage-n", type=int, default=0)
    parser.add_argument("--damage-mode", choices=["circle", "dropout"], default="circle")
    parser.add_argument("--dropout-p-min", type=float, default=0.05)
    parser.add_argument("--dropout-p-max", type=float, default=0.50)
    parser.add_argument("--fire-rate", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--lr-decay-step", type=int, default=2000)
    parser.add_argument("--lr-decay-factor", type=float, default=0.1)
    parser.add_argument("--preview-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=500)
    parser.add_argument("--rollout-steps", type=int, default=96)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--eval-horizons", type=int, nargs="+", default=[96, 200, 400])
    parser.add_argument("--eval-rollouts", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "cuda":
        return torch.device("cuda")
    if device_arg == "mps":
        return torch.device("mps")
    if device_arg == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def configure_reproducibility(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_target(args: argparse.Namespace, config: NCAConfig) -> torch.Tensor:
    if args.target_image:
        return load_target_image(args.target_image, config.target_size)
    return create_procedural_target(config.target_size, args.target_shape)


def loss_per_sample(x: torch.Tensor, target_rgba: torch.Tensor) -> torch.Tensor:
    diff = x[:, :4] - target_rgba.unsqueeze(0)
    return diff.square().mean(dim=(1, 2, 3))


def normalize_gradients(parameters: list[torch.nn.Parameter]) -> None:
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.norm()
        if grad_norm > 0:
            parameter.grad.div_(grad_norm + 1e-8)


def make_circle_damage_masks(count: int, height: int, width: int, device: torch.device) -> torch.Tensor:
    x = torch.linspace(-1.0, 1.0, width, device=device).view(1, 1, width)
    y = torch.linspace(-1.0, 1.0, height, device=device).view(1, height, 1)
    centers = torch.empty((count, 2, 1, 1), device=device).uniform_(-0.5, 0.5)
    radii = torch.empty((count, 1, 1), device=device).uniform_(0.1, 0.4)
    norm_x = (x - centers[:, 0]) / radii
    norm_y = (y - centers[:, 1]) / radii
    circle = (norm_x.square() + norm_y.square() < 1.0).to(torch.float32)
    return (1.0 - circle).unsqueeze(1)


def make_dropout_damage_masks(count: int, height: int, width: int, device: torch.device, p_min: float, p_max: float) -> torch.Tensor:
    if not 0.0 <= p_min <= 1.0 or not 0.0 <= p_max <= 1.0:
        raise ValueError(f"Dropout probabilities must be within [0, 1], got p_min={p_min}, p_max={p_max}")

    if p_min > p_max:
        raise ValueError(f"dropout-p-min must be <= dropout-p-max, got {p_min} > {p_max}")

    damage_probs = torch.empty((count, 1, 1, 1), device=device).uniform_(p_min, p_max)
    keep_mask = (torch.rand((count, 1, height, width), device=device) > damage_probs).to(torch.float32)
    return keep_mask


def apply_damage(batch: torch.Tensor, args: argparse.Namespace, damage_count: int) -> None:
    if damage_count <= 0:
        return

    if args.damage_mode == "circle":
        damage_masks = make_circle_damage_masks(
            damage_count,
            batch.shape[-2],
            batch.shape[-1],
            batch.device,
        )
    elif args.damage_mode == "dropout":
        damage_masks = make_dropout_damage_masks(
            damage_count,
            batch.shape[-2],
            batch.shape[-1],
            batch.device,
            args.dropout_p_min,
            args.dropout_p_max,
        )
    else:
        raise ValueError(f"Unsupported damage mode: {args.damage_mode}")

    batch[-damage_count:] *= damage_masks


def evaluate_model(
    model: NeuralCellularAutomata,
    config: NCAConfig,
    target_rgba: torch.Tensor,
    device: torch.device,
    horizons: list[int],
    rollout_count: int,
) -> dict[str, float]:
    model.eval()
    metrics: dict[str, float] = {}

    with torch.no_grad():
        for horizon in horizons:
            losses = []
            for _ in range(rollout_count):
                state = make_seed(1, config, device)
                for _ in range(horizon):
                    state = model(state)
                losses.append(float(loss_per_sample(state, target_rgba).item()))
            metrics[f"eval_loss_{horizon}"] = float(np.mean(losses))

    metrics["eval_loss_mean"] = float(np.mean([metrics[f"eval_loss_{horizon}"] for horizon in horizons]))
    return metrics


def rollout_frames(model: NeuralCellularAutomata, config: NCAConfig, steps: int, device: torch.device) -> list[np.ndarray]:
    model.eval()
    frames: list[np.ndarray] = []
    state = make_seed(1, config, device)
    with torch.no_grad():
        for step_index in range(steps + 1):
            rgb = rgba_to_rgb(state[:, :4])[0].permute(1, 2, 0).cpu().numpy()
            frames.append(np.uint8(np.clip(rgb, 0.0, 1.0) * 255.0))
            if step_index < steps:
                state = model(state)
    return frames


def save_preview(frames: list[np.ndarray], target_rgba: torch.Tensor, output_path: Path) -> None:
    target_rgb = rgba_to_rgb(target_rgba.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy()
    frame_count = min(8, len(frames))
    sample_indices = np.linspace(0, len(frames) - 1, frame_count).astype(int)
    tiles = [np.uint8(np.clip(target_rgb, 0.0, 1.0) * 255.0)]
    tiles.extend(frames[i] for i in sample_indices)
    preview = np.concatenate(tiles, axis=1)
    Image.fromarray(preview).save(output_path)


def save_gif(frames: list[np.ndarray], output_path: Path, duration_ms: int = 80) -> None:
    pil_frames = [Image.fromarray(frame) for frame in frames]
    pil_frames[0].save(
        output_path,
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def export_weights(model: NeuralCellularAutomata, output_path: Path) -> dict:
    conv1 = model.update_net[0]
    conv2 = model.update_net[2]
    payload = {
        "channel_n": model.config.channel_n,
        "hidden_n": model.config.hidden_n,
        "fire_rate": model.config.fire_rate,
        "alpha_channel": model.config.alpha_channel,
        "living_threshold": model.config.living_threshold,
        "identity_kernel": model.identity_kernel.detach().cpu().tolist(),
        "dx_kernel": model.dx_kernel.detach().cpu().tolist(),
        "dy_kernel": model.dy_kernel.detach().cpu().tolist(),
        "conv1_weight": conv1.weight.detach().cpu().tolist(),
        "conv1_bias": conv1.bias.detach().cpu().tolist(),
        "conv2_weight": conv2.weight.detach().cpu().tolist(),
        "conv2_bias": conv2.bias.detach().cpu().tolist(),
    }
    output_path.write_text(json.dumps(payload))
    return payload


def build_run_config(args: argparse.Namespace, config: NCAConfig, device: torch.device) -> dict:
    return {
        "model_config": asdict(config),
        "training_config": {
            "output_dir": args.output_dir,
            "target_shape": args.target_shape,
            "target_image": args.target_image,
            "batch_size": args.batch_size,
            "pool_size": args.pool_size,
            "train_steps": args.train_steps,
            "min_iter": args.min_iter,
            "max_iter": args.max_iter,
            "damage_n": args.damage_n,
            "damage_mode": args.damage_mode,
            "dropout_p_min": args.dropout_p_min,
            "dropout_p_max": args.dropout_p_max,
            "learning_rate": args.learning_rate,
            "lr_decay_step": args.lr_decay_step,
            "lr_decay_factor": args.lr_decay_factor,
            "preview_every": args.preview_every,
            "save_every": args.save_every,
            "rollout_steps": args.rollout_steps,
            "eval_every": args.eval_every,
            "eval_horizons": args.eval_horizons,
            "eval_rollouts": args.eval_rollouts,
            "seed": args.seed,
            "device": str(device),
        },
    }


def main() -> None:
    args = parse_args()
    configure_reproducibility(args.seed)
    device = resolve_device(args.device)

    config = NCAConfig(
        channel_n=args.channel_n,
        hidden_n=args.hidden_n,
        fire_rate=args.fire_rate,
        target_size=args.target_size,
        target_padding=args.target_padding,
    )

    output_dir = Path(args.output_dir)
    preview_dir = output_dir / "previews"
    checkpoint_dir = output_dir / "checkpoints"
    preview_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    target_rgba = load_target(args, config).to(device)
    padded_target = pad_target(target_rgba, config.target_padding).to(device)
    target_preview = Path(output_dir / "target.png")
    Image.fromarray(
        np.uint8(
            rgba_to_rgb(target_rgba.unsqueeze(0))[0].permute(1, 2, 0).cpu().numpy().clip(0.0, 1.0) * 255.0
        )
    ).save(target_preview)

    model = NeuralCellularAutomata(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = None
    if 0 < args.lr_decay_step < args.train_steps:
        scheduler = MultiStepLR(optimizer, milestones=[args.lr_decay_step], gamma=args.lr_decay_factor)

    base_seed = make_seed(1, config, device)
    pool = SamplePool(base_seed, args.pool_size)

    loss_log_path = output_dir / "loss.csv"
    eval_log_path = output_dir / "eval.csv"
    config_path = output_dir / "config.json"
    best_eval_path = checkpoint_dir / "best_eval.pt"
    best_eval_metrics_path = output_dir / "best_eval_metrics.json"
    config_path.write_text(json.dumps(build_run_config(args, config, device), indent=2))

    best_eval_score = float("inf")

    with loss_log_path.open("w", newline="") as loss_file:
        writer = csv.writer(loss_file)
        writer.writerow(["step", "loss", "learning_rate"])

        with eval_log_path.open("w", newline="") as eval_file:
            eval_writer = csv.writer(eval_file)
            eval_headers = ["step", "eval_loss_mean"]
            eval_headers.extend([f"eval_loss_{horizon}" for horizon in args.eval_horizons])
            eval_writer.writerow(eval_headers)

            for step in range(1, args.train_steps + 1):
                model.train()
                indices, batch = pool.sample(args.batch_size)
                with torch.no_grad():
                    ranking = torch.argsort(loss_per_sample(batch, padded_target), descending=True)
                    batch = batch[ranking]
                    indices = indices[ranking.cpu().numpy()]
                    batch[:1] = base_seed

                    if args.damage_n > 0:
                        damage_count = min(args.damage_n, batch.shape[0] - 1)
                        if damage_count > 0:
                            apply_damage(batch, args, damage_count)

                iter_n = random.randint(args.min_iter, args.max_iter)

                evolved = batch
                for _ in range(iter_n):
                    evolved = model(evolved)

                loss = loss_per_sample(evolved, padded_target).mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                normalize_gradients(list(model.parameters()))
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()

                pool.commit(indices, evolved)
                current_lr = optimizer.param_groups[0]["lr"]
                writer.writerow([step, float(loss.item()), current_lr])

                if step % 25 == 0 or step == 1:
                    print(f"step {step:04d} | loss {loss.item():.6f} | lr {current_lr:.6g} | rollout_steps {iter_n}")

                if step % args.eval_every == 0 or step == args.train_steps:
                    eval_metrics = evaluate_model(
                        model,
                        config,
                        padded_target,
                        device,
                        args.eval_horizons,
                        args.eval_rollouts,
                    )
                    eval_writer.writerow([step, eval_metrics["eval_loss_mean"], *[eval_metrics[f"eval_loss_{h}"] for h in args.eval_horizons]])
                    print(
                        f"eval step {step:04d} | mean {eval_metrics['eval_loss_mean']:.6f} | "
                        + " ".join([f"{h}:{eval_metrics[f'eval_loss_{h}']:.6f}" for h in args.eval_horizons])
                    )

                    if eval_metrics["eval_loss_mean"] < best_eval_score:
                        best_eval_score = eval_metrics["eval_loss_mean"]
                        best_payload = {
                            "config": asdict(config),
                            "state_dict": model.state_dict(),
                            "step": step,
                            "eval_metrics": eval_metrics,
                        }
                        torch.save(best_payload, best_eval_path)
                        best_eval_metrics_path.write_text(json.dumps({"step": step, **eval_metrics}, indent=2))

                if step % args.preview_every == 0 or step == args.train_steps:
                    frames = rollout_frames(model, config, args.rollout_steps, device)
                    save_preview(frames, padded_target, preview_dir / f"preview_{step:04d}.png")

                if step % args.save_every == 0 or step == args.train_steps:
                    checkpoint_path = checkpoint_dir / f"model_step_{step:04d}.pt"
                    torch.save({"config": asdict(config), "state_dict": model.state_dict(), "step": step}, checkpoint_path)

    final_frames = rollout_frames(model, config, args.rollout_steps, device)
    save_gif(final_frames, output_dir / "final_growth.gif")
    weights_payload = export_weights(model, output_dir / "weights.json")
    unity_payload = build_unity_export_payload(
        weights_payload,
        config_data=build_run_config(args, config, device),
        run_name=output_dir.name,
    )
    (output_dir / "unity_model.json").write_text(json.dumps(unity_payload, indent=2))

    print(f"Training complete. Outputs written to {output_dir}")


if __name__ == "__main__":
    main()
