import argparse
import ast
from pathlib import Path

import torch


def parse_config(config_path):
    info = {}
    if not config_path.exists():
        return info

    for line in config_path.read_text(encoding='utf-8', errors='ignore').splitlines():
        if ': ' not in line:
            continue
        key, value = line.split(': ', 1)
        info[key.strip()] = value.strip()
    return info


def parse_list(value, fallback):
    if not value:
        return fallback
    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
    except Exception:
        pass
    return fallback


def to_tensor(obj):
    if isinstance(obj, torch.Tensor):
        return obj
    return torch.as_tensor(obj)


def format_loss(epoch_idx, loss_row):
    if loss_row.ndim == 0:
        return f"[Loss: {loss_row.item():.4f}]"

    if loss_row.numel() == 0:
        return "[Loss: N/A]"

    if loss_row.numel() == 1:
        return f"[Loss: {loss_row[0].item():.4f}]"

    parts = [f"[Loss-{i + 1}: {value.item():.4f}]" for i, value in enumerate(loss_row[:-1])]
    parts.append(f"[Total: {loss_row[-1].item():.4f}]")
    return ''.join(parts)


def format_psnr(epoch_idx, psnr_row, data_names, scales):
    if psnr_row.ndim == 0:
        return f"[PSNR: {psnr_row.item():.3f}]"

    if psnr_row.ndim == 1:
        return ''.join(
            f"[PSNR-{i + 1}: {value.item():.3f}]"
            for i, value in enumerate(psnr_row)
        )

    if psnr_row.ndim == 2:
        parts = []
        for idx_data in range(psnr_row.size(0)):
            data_name = data_names[idx_data] if idx_data < len(data_names) else f"Data{idx_data + 1}"
            for idx_scale in range(psnr_row.size(1)):
                scale = scales[idx_scale] if idx_scale < len(scales) else f"S{idx_scale + 1}"
                parts.append(f"[{data_name} x{scale}] PSNR: {psnr_row[idx_data, idx_scale].item():.3f}")
        return ' | '.join(parts)

    return f"[PSNR tensor shape: {tuple(psnr_row.shape)}]"


def main():
    parser = argparse.ArgumentParser(description="Read experiment PSNR/Loss logs by epoch.")
    parser.add_argument("experiment_dir", type=str, help="Path to one experiment directory")
    args = parser.parse_args()

    exp_dir = Path(args.experiment_dir)
    psnr_path = exp_dir / "psnr_log.pt"
    loss_path = exp_dir / "loss_log.pt"
    config_path = exp_dir / "config.txt"

    if not psnr_path.exists() and not loss_path.exists():
        raise FileNotFoundError(f"No psnr_log.pt or loss_log.pt found in: {exp_dir}")

    config = parse_config(config_path)
    data_names = parse_list(config.get("data_test"), [])
    scales = parse_list(config.get("scale"), [])

    psnr_log = torch.load(psnr_path, map_location="cpu") if psnr_path.exists() else None
    loss_log = torch.load(loss_path, map_location="cpu") if loss_path.exists() else None

    psnr_log = to_tensor(psnr_log) if psnr_log is not None else None
    loss_log = to_tensor(loss_log) if loss_log is not None else None

    n_epochs = 0
    if psnr_log is not None:
        n_epochs = max(n_epochs, psnr_log.size(0))
    if loss_log is not None:
        n_epochs = max(n_epochs, loss_log.size(0))

    print(f"Experiment: {exp_dir}")
    if config:
        if data_names:
            print(f"Test datasets: {data_names}")
        if scales:
            print(f"Scales: {scales}")
    print()

    for epoch_idx in range(n_epochs):
        parts = [f"[Epoch {epoch_idx + 1}]"]

        if loss_log is not None and epoch_idx < loss_log.size(0):
            parts.append(format_loss(epoch_idx, loss_log[epoch_idx].reshape(-1)))
        else:
            parts.append("[Loss: N/A]")

        if psnr_log is not None and epoch_idx < psnr_log.size(0):
            parts.append(format_psnr(epoch_idx, psnr_log[epoch_idx], data_names, scales))
        else:
            parts.append("[PSNR: N/A]")

        print("\t".join(parts))


if __name__ == "__main__":
    main()
