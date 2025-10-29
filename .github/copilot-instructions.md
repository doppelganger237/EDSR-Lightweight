# AI agent guide for this repo

This repo is a PyTorch implementation of EDSR and lightweight variants for single-image super-resolution. Source lives in `src/`, datasets under `dataset/`, and runs log to `experiment/<run-name>/`.

## Architecture in 30 seconds
- Entry point: `src/main.py` wires args → data loaders → model → loss → `Trainer` loop. Video inference goes through `src/videotester.py` when `--data_test video`.
- CLI/options: `src/option.py` defines all args; `src/template.py` mutates args for presets (e.g., `--template EDSR_paper`, `MDSR`, `RCAN`). Scales support `"2+3+4"`.
- Models: `src/model/__init__.py` imports `model.<args.model.lower()>` and expects a `make_model(args)` that returns an `nn.Module`.
  - Baseline: `src/model/edsr.py` (MeanShift head/tail, residual body, tolerant `load_state_dict` that ignores tail mismatches between scales).
  - Lightweight: `src/model/edsr_variants.py` exposes `ULRNet` with toggles `--use_dwconv`, `--use_attention` (module name is `EDSR_VARIANTS`).
  - Other examples: `vdsr.py`, `ddbpn.py`, `rdn.py`, `rcan.py`, `mdsr.py`.
- Data: `src/data/` with `SRData` base (`srdata.py`) and datasets (`div2k.py`, `benchmark.py`). Patch extraction and augmentation live in `data/common.py`.
- Training loop: `src/trainer.py` handles loop, epoch scheduling, device pick (CUDA/MPS/CPU), PSNR/SSIM eval, result saving. Optim/scheduler and logging live in `src/utility.py`.

## Workflows that matter
- Training (from `src/`):
  - Set dataset root in `--dir_data` (default `../dataset`). First run can pre-decode pngs: `--ext sep-reset` (then reuse with `--ext sep`). For test-only, prefer default `--ext img`.
  - Typical EDSR baseline x2: `python main.py --model EDSR --scale 2 --save EDSR_Baseline_x2 --epochs 300`.
  - Paper setting: add `--n_resblocks 32 --n_feats 256 --res_scale 0.1` or `--template EDSR_paper`.
  - Lightweight variant: `python main.py --model EDSR_VARIANTS --scale 2 --use_dwconv --use_attention --save sr_dwconv_att_x2`.
- Evaluation:
  - With pretrained: `--pre_train download` (uses URLs embedded in model) or `--pre_train <path.pt>` and `--test_only`.
  - Benchmarks: `--data_test Set5+Set14+B100+Urban100` and `--data_range 801-900` for DIV2K splits.
  - Results/images saved to `experiment/<run>/results-<dataset>/`; logs/PSNR/SSIM and plots also stored under the run dir.
- Benchmarking (params/FLOPs/FPS): run `src/benchmark.py` (uses `thop`). Select device auto (CUDA > MPS > CPU).

## Project conventions and patterns
- Model plugin contract: file `src/model/<name>.py` must export `make_model(args)`. Put optional `url` map for `--pre_train download`. Implement `load_state_dict` to ignore `tail` shape differences across scales.
- Device handling: training/inference selects CUDA or Apple MPS automatically; DataLoader sets `pin_memory` only for CUDA.
- Multi-scale: `args.scale` is a list; training fixes scale index 0; testing iterates all.
- Metrics: PSNR crops border by `scale` for benchmarks (Y-channel conversion inside PSNR when `benchmark=True`), else `scale+6`. SSIM is implemented in `utility.calc_ssim` with Gaussian window.
- Experiments: by default saved under `../experiment/<save>/` relative to `src/`. Checkpoints: `model_latest.pt`, `model_best.pt` (when best PSNR); optimizer, loss logs, and plots persisted.
- Loss config: string like `1*L1`, `1*MSE`, or mixes with `VGGxx`/`GAN` (see `src/loss/`).

## Integration points and examples
- Add a new model variant: create `src/model/myvariant.py` with `make_model(args)`, then run with `--model MYVARIANT` (import key is lowercased file name). Reuse `utility.quantize`, `utility.calc_psnr/ssim` and respect `args.rgb_range`.
- Use chop/self-ensemble: add `--chop` for memory-efficient tiling; add `--self_ensemble` for x8 test-time augment average.
- Video: set `--data_test video --dir_demo <dir>` to route through `videotester.py`.

## Gotchas the code depends on
- Paths are relative to `src/`: experiment dir is `..` from there; ensure you run inside `src/`.
- For DIV2K caching, clean or regenerate when switching `--ext` modes (`sep-reset` regenerates `.pt` binaries under `<dataset>/bin`).
- `EDSR_VARIANTS` behavior controlled by `--use_dwconv`/`--use_attention` flags; default `False` unless provided.
- When resuming, `--resume -1|0|N` interacts with `--pre_train` and `experiment/<run>/model_*.pt` (see `Model.load`).
