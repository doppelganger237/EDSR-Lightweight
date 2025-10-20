import os
import torch
import torch.nn as nn
import platform

import utility
import data
import model
import loss
from option import args
from trainer import Trainer

# Optional distributed imports
import torch.distributed as dist

torch.backends.cudnn.benchmark = True

# Seed will be adjusted per rank if running distributed
def set_seed(seed, rank=0):
    torch.manual_seed(seed + rank)


def main():
    # --- 环境与设备检测 ---
    is_mps = torch.backends.mps.is_available()
    is_cuda = torch.cuda.is_available()
    on_mac = platform.system() == "Darwin"
    on_windows = platform.system() == "Windows"

    # 获取分布式环境变量
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    local_rank = int(os.environ.get('LOCAL_RANK', os.environ.get('RANK', '0')))
    use_ddp = (world_size > 1) and is_cuda  # 仅在 CUDA 环境启用 DDP

    # --- 设备与 DDP 初始化 ---
    if use_ddp:
        try:
            dist.init_process_group(backend='nccl')
            torch.cuda.set_device(local_rank)
            device = torch.device(f'cuda:{local_rank}')
            args.distributed = True
            args.world_size = world_size
            args.rank = int(os.environ.get('RANK', '0'))
            print(f"[DDP] init: rank={args.rank}, local_rank={local_rank}, world_size={world_size}")
        except Exception as e:
            print(f"[DDP] Initialization failed ({e}), fallback to single GPU mode")
            use_ddp = False
            device = torch.device('cuda' if is_cuda else 'cpu')
    else:
        if is_mps:
            device = torch.device('mps')
            print("[INFO] Using Apple MPS backend")
        elif is_cuda:
            device = torch.device('cuda')
            print("[INFO] Using CUDA backend")
        else:
            device = torch.device('cpu')
            print("[INFO] Using CPU backend")

        args.distributed = False
        args.world_size = 1
        args.rank = 0

    set_seed(args.seed, args.rank)

    checkpoint = utility.checkpoint(args)

    if args.data_test == ['video']:
        from videotester import VideoTester
        _model = model.Model(args, checkpoint)

        # move / wrap model for distributed or single-GPU
        if use_ddp:
            _model = _model.to(device)
            _model = nn.parallel.DistributedDataParallel(_model, device_ids=[local_rank])
        else:
            if isinstance(_model, nn.Module):
                _model = _model.to(device)
            else:
                for attr in ('model', 'net', 'module', 'network', 'generator', 'body'):
                    if hasattr(_model, attr):
                        mod = getattr(_model, attr)
                        if isinstance(mod, nn.Module):
                            setattr(_model, attr, mod.to(device))
                            break

        t = VideoTester(args, _model, checkpoint)
        t.test()
        return

    if checkpoint.ok:
        # Data loader
        loader = data.Data(args)

        # Create model and loss
        _model = model.Model(args, checkpoint)
        _loss = loss.Loss(args, checkpoint) if not args.test_only else None

        # Move / wrap model for distributed training if needed
        if use_ddp:
            _model = _model.to(device)
            _model = nn.parallel.DistributedDataParallel(_model, device_ids=[local_rank])
        else:
            if isinstance(_model, nn.Module):
                _model = _model.to(device)
            else:
                for attr in ('model', 'net', 'module', 'network', 'generator', 'body'):
                    if hasattr(_model, attr):
                        mod = getattr(_model, attr)
                        if isinstance(mod, nn.Module):
                            setattr(_model, attr, mod.to(device))
                            break

        t = Trainer(args, loader, _model, _loss, checkpoint)

        # Training loop
        while not t.terminate():
            t.train()
            t.test()

        # Only rank 0 should finalize checkpoint to avoid races
        if not use_ddp or args.rank == 0:
            checkpoint.done()


if __name__ == '__main__':
    main()
