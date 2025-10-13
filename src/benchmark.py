import torch
import torch.nn as nn
import time
import csv
from thop import profile
import model
from option import args
import utility
import os
from datetime import datetime
import warnings

# Initialize checkpoint
checkpoint = utility.checkpoint(args)


def select_device(preferred_device=None):
    """
    自动选择设备（优先级：CUDA > MPS > CPU），或使用用户指定的 preferred_device。
    """
    if preferred_device is not None:
        if preferred_device == 'cuda' and torch.cuda.is_available():
            return 'cuda'
        if preferred_device == 'mps' and getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            return 'mps'
        if preferred_device == 'cpu':
            return 'cpu'
    # 自动选择
    if torch.cuda.is_available():
        return 'cuda'
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def benchmark(model_name, scale=2, input_size=96, device='cpu', n_warmup=10, n_runs=50,
              use_dwconv=False, use_ca=False, use_sa=False):
    """
    对单个模型进行基准测试：参数量、FLOPs 与推理耗时（FPS）。
    返回一个结果字典，或在失败时返回 None。
    """
    # 配置全局 args（与 main.py 保持一致）
    args.model = model_name
    args.scale = [scale]
    args.n_feats = 64
    args.n_resblocks = 16
    args.res_scale = 1
    args.n_colors = 3
    args.use_dwconv = use_dwconv
    args.use_ca = use_ca
    args.use_sa = use_sa

    try:
        # 参数量统计（载入模型一次）
        net_params = model.Model(args, checkpoint).to(device)
        num_params = sum(p.numel() for p in net_params.parameters() if p.requires_grad)
        del net_params

        # FLOPs 统计（使用 thop profile）
        net_flops = model.Model(args, checkpoint).to(device)
        net_flops.eval()
        module_for_profile = getattr(net_flops, 'model', net_flops).to(device)
        x = torch.randn(1, args.n_colors, input_size, input_size).to(device)
        try:
            flops, _ = profile(module_for_profile, inputs=(x,), verbose=False)
        except Exception as e:
            warnings.warn(f"FLOPs profiling failed: {e}")
            flops = 0.0
        del net_flops

        # 推理时间测量（多次平均）
        net_infer = model.Model(args, checkpoint).to(device)
        net_infer.eval()
        dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
        with torch.no_grad():
            # 预热
            for _ in range(n_warmup):
                try:
                    _ = net_infer(dummy_input, 0)
                except TypeError:
                    # 某些 model.forward 可能不需要 idx_scale
                    _ = net_infer(dummy_input)
            if device == "mps":
                torch.mps.synchronize()
            start = time.time()
            for _ in range(n_runs):
                try:
                    _ = net_infer(dummy_input, 0)
                except TypeError:
                    _ = net_infer(dummy_input)
            if device == "mps":
                torch.mps.synchronize()
            end = time.time()

        avg_time = (end - start) / n_runs * 1000.0  # 毫秒
        fps = 1000.0 / avg_time if avg_time > 0 else 0.0

        # 打印简要信息
        print("=" * 65)
        print(f"{'Model':<20}: {model_name}_x{scale} (DWConv={use_dwconv}, CA={use_ca}, SA={use_sa})")
        print(f"{'Parameters':<20}: {num_params / 1e3:10.4f} K")
        print(f"{'FLOPs':<20}: {flops / 1e9:10.4f} G (input: {input_size}x{input_size})")
        print(f"{'Avg Inference Time':<20}: {avg_time:10.2f} ms")
        print(f"{'FPS':<20}: {fps:10.2f}")
        print("=" * 65)

        return {
            "Model": model_name,
            "Use_DWConv": use_dwconv,
            "Use_CA": use_ca,
            "Use_SA": use_sa,
            "Params (K)": round(num_params / 1e3, 4),
            "FLOPs (G)": round(flops / 1e9, 4) if flops else 0.0,
            "FPS": round(fps, 2),
            "Inference Time (ms)": round(avg_time, 2)
        }

    except Exception as e:
        warnings.warn(f"Benchmark failed for model '{model_name}' with error: {e}")
        return None


def read_previous_results(filepath):
    """
    更稳健的 CSV 读取函数，优先用 pandas.read_csv 自动解析（支持 UTF-8 BOM、Excel、NUL、空行等），
    失败时退回 csv.DictReader 方案。
    返回结果字典：{(model, use_dwconv, use_ca, use_sa): {...}}
    """
    import re
    import codecs
    import warnings
    prev_results = {}

    def safe_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return 0.0

    def str_to_bool(s):
        if s is None:
            return False
        return str(s).strip().lower() in ("true", "1", "yes")

    if not os.path.isfile(filepath):
        print(f"⚠️  File not found: {filepath}")
        return {}

    # Try pandas first for robust CSV parsing
    try:
        import pandas as pd
        # Try several encodings
        encodings = ["utf-8-sig", "utf-8", "gbk", "latin1"]
        for enc in encodings:
            try:
                # read_csv handles BOM, Excel, NUL, etc.
                df = pd.read_csv(filepath, encoding=enc, engine="python")
                # Drop empty columns/rows
                df = df.dropna(how='all')
                if df.empty:
                    continue
                # Check for required columns
                if not any(col in df.columns for col in ["Model", "FPS", "FLOPs (G)", "FLOPs"]):
                    continue
                # Some older files may have "FLOPs" instead of "FLOPs (G)"
                # Standardize column names
                col_map = {}
                for c in df.columns:
                    if c.strip().lower() == "flops":
                        col_map[c] = "FLOPs (G)"
                    if c.strip().lower() == "params (m)":
                        col_map[c] = "Params (K)"
                if col_map:
                    df = df.rename(columns=col_map)
                parsed = 0
                for _, row in df.iterrows():
                    model = str(row.get("Model", "")).strip()
                    if not model:
                        continue
                    key = (
                        model,
                        str_to_bool(row.get("Use_DWConv")),
                        str_to_bool(row.get("Use_CA")),
                        str_to_bool(row.get("Use_SA")),
                    )
                    prev_results[key] = {
                        "Params (K)": safe_float(row.get("Params (K)")),
                        "FLOPs (G)": safe_float(row.get("FLOPs (G)")),
                        "FPS": safe_float(row.get("FPS")),
                        "Inference Time (ms)": safe_float(row.get("Inference Time (ms)")),
                    }
                    parsed += 1
                if parsed > 0:
                    print(f"✅ Successfully parsed {parsed} rows from {filepath} (encoding={enc}, pandas)")
                    return prev_results
            except Exception as e:
                continue
    except ImportError:
        warnings.warn("pandas not available, fallback to csv.DictReader.")
    except Exception as e:
        warnings.warn(f"pandas CSV parse failed: {e}, fallback to csv.DictReader.")

    # Fallback: csv.DictReader with robust cleaning
    encodings = ["utf-8-sig", "utf-8", "gbk", "latin1"]
    separators = [",", ";", "\t"]
    for enc in encodings:
        try:
            with codecs.open(filepath, "r", encoding=enc, errors="ignore") as f:
                content = f.read()
            # 清理不可见字符和 Excel 导出的 NUL
            content = re.sub(r"[\x00-\x1f\x7f]", "", content).strip()
            if not content:
                continue
            lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
            if not lines:
                continue
            header = lines[0]
            if not any(k in header for k in ["Model", "FPS", "FLOPs"]):
                continue
            for sep in separators:
                try:
                    reader = csv.DictReader(lines, delimiter=sep)
                    if not reader.fieldnames:
                        continue
                    parsed = 0
                    for row in reader:
                        if not row.get("Model"):
                            continue
                        key = (
                            row.get("Model", "").strip(),
                            str_to_bool(row.get("Use_DWConv")),
                            str_to_bool(row.get("Use_CA")),
                            str_to_bool(row.get("Use_SA")),
                        )
                        prev_results[key] = {
                            "Params (K)": safe_float(row.get("Params (K)")),
                            "FLOPs (G)": safe_float(row.get("FLOPs (G)")),
                            "FPS": safe_float(row.get("FPS")),
                            "Inference Time (ms)": safe_float(row.get("Inference Time (ms)")),
                        }
                        parsed += 1
                    if parsed > 0:
                        print(f"✅ Successfully parsed {parsed} rows from {filepath} (encoding={enc}, sep='{sep}')")
                        return prev_results
                except Exception:
                    continue
        except Exception:
            continue
    print(f"⚠️  No valid rows parsed from {filepath} (all methods failed). File may be empty or malformed.")
    return prev_results


def format_change(current, previous, higher_is_better=True):
    """
    Formats the percentage change between current and previous values.
    If previous is zero or None, returns '-'.
    """
    if previous is None or previous == 0:
        return "-"
    try:
        change = (current - previous) / previous * 100.0
    except Exception:
        return "-"
    arrow = "↑" if (change > 0 and higher_is_better) or (change < 0 and not higher_is_better) else "↓"
    return f"{arrow}{abs(change):.2f}%"


def benchmark_all(save_csv=True, preferred_device=None):
    device = select_device(preferred_device)
    print(f"Using device: {device}\n")
    results = []

    configs = [
        dict(model_name="EDSR", use_dwconv=False, use_ca=False, use_sa=False),
        dict(model_name="EDSR_VARIANTS", use_dwconv=True, use_ca=False, use_sa=False),
        dict(model_name="EDSR_VARIANTS", use_dwconv=False, use_ca=True, use_sa=False),
        dict(model_name="EDSR_VARIANTS", use_dwconv=False, use_ca=False, use_sa=True),
        dict(model_name="EDSR_VARIANTS", use_dwconv=True, use_ca=True, use_sa=False),
        dict(model_name="EDSR_VARIANTS", use_dwconv=True, use_ca=True, use_sa=True),
    ]

    start_all = time.time()
    for cfg in configs:
        result = benchmark(
            model_name=cfg["model_name"],
            scale=2,
            input_size=96,
            device=device,
            use_dwconv=cfg["use_dwconv"],
            use_ca=cfg["use_ca"],
            use_sa=cfg["use_sa"]
        )
        if result is not None:
            results.append(result)
    end_all = time.time()

    total_duration = end_all - start_all

    # 收集 results 目录下的所有有效 CSV 文件（排除 macOS 资源分叉文件与最新结果文件）
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    csv_files = [
        os.path.join(results_dir, f)
        for f in os.listdir(results_dir)
        if f.endswith(".csv") and not f.startswith("._") and f != "benchmark_latest.csv"
    ]
    # 按修改时间倒序排序（最新在前）
    csv_files.sort(key=os.path.getmtime, reverse=True)
    prev_csv_path = None
    if len(csv_files) > 1:
        # 使用倒数第二个作为上一轮结果
        prev_csv_path = csv_files[1]
    elif len(csv_files) == 1:
        prev_csv_path = csv_files[0]
    else:
        prev_csv_path = None

    # debug 输出
    print(f"Found csv files (most recent first): {csv_files}")
    print(f"Using previous CSV for comparison: {prev_csv_path}")

    prev_results = {}
    if prev_csv_path:
        prev_results = read_previous_results(prev_csv_path)
        print(f"Loaded previous results entries: {len(prev_results)}")

    # 计算变化
    for r in results:
        key = (r["Model"], bool(r["Use_DWConv"]), bool(r["Use_CA"]), bool(r["Use_SA"]))
        prev = None
        if key in prev_results:
            prev = prev_results[key]
        else:
            # 模糊匹配（先找同模型名）
            for k, v in prev_results.items():
                if k[0] == r["Model"]:
                    prev = v
                    break

        prev_fps = None
        prev_time = None
        if prev:
            prev_fps = float(prev.get("FPS", 0)) if prev.get("FPS") is not None else None
            prev_time = float(prev.get("Inference Time (ms)", 0)) if prev.get("Inference Time (ms)") is not None else None

        r["FPS Δ"] = format_change(r["FPS"], prev_fps, True) if prev_fps is not None else "-"
        r["Inference Time Δ"] = format_change(r["Inference Time (ms)"], prev_time, False) if prev_time is not None else "-"

    # 保存结果（按时间戳写入并更新 benchmark_latest.csv）
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = f"{results_dir}/benchmark_results_{now_str}.csv"
    latest_path = f"{results_dir}/benchmark_latest.csv"

    fieldnames = ["Model", "Use_DWConv", "Use_CA", "Use_SA",
                  "Params (K)", "FLOPs (G)", "FPS", "Inference Time (ms)",
                  "FPS Δ", "Inference Time Δ"]
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
    header = ["Model", "DWConv", "CA", "SA", "Params (K)", "FLOPs (G)", "FPS", "FPS Δ", "Inference Time (ms)", "Inference Time Δ"]
    print("-" * 120)
    print(" | ".join(f"{h:^15}" for h in header))
    print("-" * 120)
    for r in results:
        params_fmt = f"{r['Params (K)']:,.0f}"
        flops_fmt = f"{r['FLOPs (G)']:.4f}"
        print(f"{r['Model']:<15} | "
              f"{r['Use_DWConv']!s:^15} | {r['Use_CA']!s:^15} | {r['Use_SA']!s:^15} | "
              f"{params_fmt:^15} | {flops_fmt:^15} | "
              f"{r['FPS']:^15.2f} | {r['FPS Δ']:^15} | "
              f"{r['Inference Time (ms)']:^15.2f} | {r['Inference Time Δ']:^15}")
    print("-" * 120)


if __name__ == "__main__":
    benchmark_all(save_csv=True)