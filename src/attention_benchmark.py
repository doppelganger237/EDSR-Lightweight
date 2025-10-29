# attention_benchmark_ulrnet.py
import torch
from model.edsr_variants import ULRNet
from option import args

args.scale = [2]
args.patch_size = 64
args.batch_size = 32
args.n_threads = 0
args.use_attention = True
args.n_colors = 3  # 确保是3通道

def count_attention_params(module, mode=None):
    total = 0
    cls_name = module.__class__.__name__.lower()
    if mode == 'ul':
        # For UL mode, sum all parameters of the module directly
        total += sum(p.numel() for p in module.parameters() if p.requires_grad)
    else:
        if any(name in cls_name for name in ['ca', 'ccaattention', 'eca', 'sa', 'esa', 'lsa', 'dila', 'stage']):
            total += sum(p.numel() for p in module.parameters() if p.requires_grad)
    return total

def benchmark_network_attention(args, input_size=None):
    if input_size is None:
        input_size = args.patch_size

    att_modes = ['ca', 'cca', 'eca', 'sa', 'esa', 'lsa', 'dila', 'stage', 'ul']
    results = []

    for mode in att_modes:
        args.att_mode = mode
        model = ULRNet(args)
        model.eval()

        total_params = 0
        for m in model.modules():
            total_params += count_attention_params(m, mode=mode)

        total_params /= 1e3  # 转换为千参数

        results.append((mode, total_params))

    # 打印结果
    print(f"{'Attention Mode':<15} | {'Params (K)':<10}")
    print("-"*30)
    for mode, p in results:
        print(f"{mode:<15} | {p:<10.2f}")

if __name__ == "__main__":
    benchmark_network_attention(args)