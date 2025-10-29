import torch
import torch.nn as nn
import time
import csv
from torchinfo import summary
import model
from option import args
import utility
import os
from datetime import datetime
import warnings
args.n_feats = 64
args.n_resblocks = 8

# Initialize checkpoint
checkpoint = utility.checkpoint(args)


def benchmark(model_name, scale=2, input_size=96,
              n_warmup=10, n_runs=100,
              use_dwconv=False, use_attention=False, use_fullres=False):
    """
    对单个模型进行基准测试：参数量、FLOPs 与推理耗时（FPS）。
    use_fullres: 是否使用 HR=1280×720 口径计算 FLOPs，默认为 False。
    返回一个结果字典，或在失败时返回 None。
    """
    # 配置全局 args（与 main.py 保持一致）

    args.model = model_name
    args.use_dwconv = use_dwconv
    args.use_attention = use_attention
    args.scale = [scale]
    try:
        # 参数量统计（载入模型一次）
        net_params = model.Model(args, checkpoint)
        
        num_params = sum(p.numel() for p in net_params.parameters() if p.requires_grad)
        del net_params

        # 计算 LR 输入尺寸
        if use_fullres:
            lr_h, lr_w = 720 // args.scale[0], 1280 // args.scale[0]
        else:
            lr_h = lr_w = input_size

        # FLOPs 统计（使用 torchinfo.summary）
        net_flops = model.Model(args, checkpoint)
        net_flops.eval()
        x = torch.randn(1, args.n_colors, lr_h, lr_w).to(net_flops.device)
        try:
            info = summary(net_flops.model, input_data=x, verbose=0)
            flops = 2 * info.total_mult_adds  # MACs -> FLOPs
        except Exception as e:
            warnings.warn(f"FLOPs profiling failed: {e}")
            flops = 0.0
        del net_flops

        # -------------------------
        # 稳定推理时间测量（多轮采样）
        # -------------------------
        import numpy as np
        net_infer = model.Model(args, checkpoint)
        net_infer.eval()
        dummy_input = torch.randn(1, 3, lr_h, lr_w).to(net_infer.device)

        def sync():
            if net_infer.device.type == "cuda":
                torch.cuda.synchronize()
            elif net_infer.device.type == "mps":
                torch.mps.synchronize()

        with torch.no_grad():
            # 预热阶段
            for _ in range(n_warmup):
                try:
                    _ = net_infer(dummy_input, 0)
                except TypeError:
                    _ = net_infer(dummy_input)
            sync()

            # 多轮测试
            times = []
            for _ in range(n_runs):
                sync()
                start = time.time()
                try:
                    _ = net_infer(dummy_input, 0)
                except TypeError:
                    _ = net_infer(dummy_input)
                sync()
                end = time.time()
                times.append((end - start) * 1000.0)

        # 去除异常值（3σ原则）
        times = np.array(times)
        mean_t = np.mean(times)
        std_t = np.std(times)
        times = times[np.abs(times - mean_t) < 3 * std_t]
        median_time = np.median(times)
        fps = 1000.0 / median_time if median_time > 0 else 0.0

        # 打印简要信息
        print("=" * 65)
        print(f"{'Model':<20}: {model_name}_x{scale} (DWConv={use_dwconv}, Attention={use_attention})")
        print(f"{'Parameters':<20}: {num_params / 1e3:10.4f} K")
        if use_fullres:
            print(f"{'FLOPs':<20}: {flops / 1e9:10.4f} G (HR=1280×720, scale×{args.scale[0]})")
        else:
            print(f"{'FLOPs':<20}: {flops / 1e9:10.4f} G (input: {lr_w}×{lr_h})")
        print(f"{'Inference Time (median)':<20}: {median_time:10.3f} ms")
        print(f"{'FPS':<20}: {fps:10.2f}")
        print("=" * 65)

        return {
            "Model": model_name,
            "Use_DWConv": use_dwconv,
            "Use_Attention": use_attention,
            "Params (K)": num_params / 1e3,
            "FLOPs (G)": flops / 1e9 if flops else 0.0,
            "FPS": fps,
            "Inference Time (ms)": median_time,
        }

    except Exception as e:
        warnings.warn(f"Benchmark failed for model '{model_name}' with error: {e}")
        return None






def benchmark_all(save_csv=True):
    results = []

    configs = [
        dict(model_name="EDSR", use_dwconv=False, use_attention=False),
        dict(model_name="EDSR_VARIANTS", use_dwconv=True, use_attention=False),
        dict(model_name="EDSR_VARIANTS", use_dwconv=False, use_attention=True),
        dict(model_name="EDSR_VARIANTS", use_dwconv=True, use_attention=True),
    ]

    start_all = time.time()
    for cfg in configs:
        result = benchmark(
            model_name=cfg["model_name"],
            scale=2,
            input_size=96,
            use_dwconv=cfg["use_dwconv"],
            use_attention=cfg["use_attention"],
            use_fullres=False
        )
        if result is not None:
            results.append(result)
    end_all = time.time()

    total_duration = end_all - start_all

    # 保存结果（按时间戳写入并更新 benchmark_latest.csv）
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{results_dir}/benchmark_results_{now_str}.csv"
    latest_path = f"{results_dir}/benchmark_latest.csv"

    fieldnames = ["Model", "Use_DWConv", "Use_Attention",
                  "Params (K)", "FLOPs (G)",
                  "FPS", "Inference Time (ms)"]
    for path in [csv_path, latest_path]:
        with open(path, "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                # Format Params (K) as integer with thousand separators
                params_formatted = f"{row.get('Params (K)', 0):,.0f}"
                # Format FLOPs (G) with 4 decimal places
                flops_formatted = f"{row.get('FLOPs (G)', 0):.4f}"
                out_row = row.copy()
                out_row["Params (K)"] = params_formatted
                out_row["FLOPs (G)"] = flops_formatted
                writer.writerow({k: out_row.get(k, "") for k in fieldnames})

    print(f"\n✅ Results saved to {csv_path}")
    print(f"   Total time: {total_duration:.2f}s\n")

    # 打印表格
    header = ["Model", "DWConv", "Attention", "Params (K)", "FLOPs (G)", "FPS", "Inference Time (ms)"]
    print("-" * 100)
    print(" | ".join(f"{h:^15}" for h in header))
    print("-" * 100)
    for r in results:
        params_fmt = f"{r['Params (K)']:,.0f}"
        flops_fmt = f"{r['FLOPs (G)']:.4f}"
        print(f"{r['Model']:<15} | "
              f"{r['Use_DWConv']!s:^15} | {r['Use_Attention']!s:^15} | "
              f"{params_fmt:^15} | {flops_fmt:^15} | "
              f"{r['FPS']:^15.2f} | "
              f"{r['Inference Time (ms)']:^15.2f}")
    print("-" * 100)



if __name__ == "__main__":
    benchmark_all(save_csv=True)