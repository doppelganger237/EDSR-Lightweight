import torch
import torch.nn as nn
import time
from thop import profile
import model
from option import args
import utility

# 初始化 checkpoint
checkpoint = utility.checkpoint(args)

def benchmark(model_name, scale=2, input_size=96, device='cpu', n_warmup=10, n_runs=50):
    # 构造 args
    args.model = model_name
    args.scale = [scale]
    args.n_feats = 64
    args.n_resblocks = 16
    args.res_scale = 1
    args.n_colors = 3

    # ================= 参数量 =================
    net_params = model.Model(args, checkpoint).to(device)
    num_params = sum(p.numel() for p in net_params.parameters() if p.requires_grad)
    del net_params

    # ================= FLOPs =================
    net_flops = model.Model(args, checkpoint).to(device)
    net_flops.eval()
    module_for_profile = getattr(net_flops, 'model', net_flops).to(device)
    module_for_profile.eval()
    x = torch.randn(1, args.n_colors, input_size, input_size).to(device)
    flops, flop_params = profile(module_for_profile, inputs=(x,), verbose=False)
    del net_flops

    # ================= 推理时间 =================
    net_infer = model.Model(args, checkpoint).to(device)
    net_infer.eval()
    dummy_input = torch.randn(1, 3, input_size, input_size).to(device)
    with torch.no_grad():
        # 预热
        for _ in range(n_warmup):
            _ = net_infer(dummy_input, 0)

        # 正式计时
        torch.mps.synchronize() if device == "mps" else None
        start = time.time()
        for _ in range(n_runs):
            _ = net_infer(dummy_input, 0)
        torch.mps.synchronize() if device == "mps" else None
        end = time.time()

    avg_time = (end - start) / n_runs * 1000  # 毫秒
    fps = 1000.0 / avg_time

    # ================= 输出 =================
    print("============================================================")
    print(f"Model: {model_name}_x{scale}")
    print(f"Params: {num_params/1e6:.4f} M ({num_params:,} parameters)")
    print(f"FLOPs: {flops/1e9:.4f} G (for input {input_size}x{input_size})")
    print(f"Avg Inference Time: {avg_time:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print("============================================================")


if __name__ == "__main__":
    # 示例： baseline EDSR 和 DWConv 版本对比
    benchmark(model_name="EDSR", scale=2, input_size=96, device="mps")
    #benchmark(model_name="EDSR_DWConv", scale=2, input_size=96, device="mps")