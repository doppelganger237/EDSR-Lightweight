import os
import math
import time
import datetime
from concurrent.futures import ThreadPoolExecutor

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import numpy as np
import imageio

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lrs

import torch.nn.functional as F

class timer():
    def __init__(self):
        self.acc = 0
        self.tic()

    def tic(self):
        self.t0 = time.time()

    def toc(self, restart=False):
        diff = time.time() - self.t0
        if restart: self.t0 = time.time()
        return diff

    def hold(self):
        self.acc += self.toc()

    def release(self):
        ret = self.acc
        self.acc = 0

        return ret

    def reset(self):
        self.acc = 0

def bg_target(queue):
    # kept for backward compatibility; unused when using ThreadPoolExecutor
    while False:
        break

class checkpoint():
    def __init__(self, args):
        self.args = args
        self.ok = True
        self.log = torch.Tensor()
        now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')

        if not args.load:
            if not args.save:
                args.save = now
            self.dir = os.path.join('..', 'experiment', args.save)
        else:
            self.dir = os.path.join('..', 'experiment', args.load)
            if os.path.exists(self.dir):
                self.log = torch.load(self.get_path('psnr_log.pt'))
                print('Continue from epoch {}...'.format(len(self.log)))
            else:
                args.load = ''

        if args.reset:
            os.system('rm -rf ' + self.dir)
            args.load = ''

        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.get_path('model'), exist_ok=True)
        for d in args.data_test:
            os.makedirs(self.get_path('results-{}'.format(d)), exist_ok=True)

        open_type = 'a' if os.path.exists(self.get_path('log.txt'))else 'w'
        self.log_file = open(self.get_path('log.txt'), open_type)
        with open(self.get_path('config.txt'), open_type) as f:
            f.write(now + '\n\n')
            for arg in vars(args):
                f.write('{}: {}\n'.format(arg, getattr(args, arg)))
            f.write('\n')

        self.n_processes = 8

    def get_path(self, *subdir):
        return os.path.join(self.dir, *subdir)

    def save(self, trainer, epoch, is_best=False):
        trainer.model.save(self.get_path('model'), epoch, is_best=is_best)
        trainer.loss.save(self.dir)
        trainer.loss.plot_loss(self.dir, epoch)

        self.plot_psnr(epoch)
        trainer.optimizer.save(self.dir)
        torch.save(self.log, self.get_path('psnr_log.pt'))

    def add_log(self, log):
        self.log = torch.cat([self.log, log])

    def write_log(self, log, refresh=False):
        print(log)
        try:
            self.log_file.write(log + '\n')
            self.log_file.flush()
        except Exception:
            pass
        if refresh:
            try:
                self.log_file.close()
            except Exception:
                pass
            self.log_file = open(self.get_path('log.txt'), 'a')

    def done(self):
        try:
            if hasattr(self, 'executor'):
                for f in getattr(self, '_futures', []):
                    try:
                        f.result()
                    except Exception:
                        pass
                self.executor.shutdown(wait=True)
        except Exception:
            pass
        try:
            self.log_file.close()
        except Exception:
            pass

    def plot_psnr(self, epoch):
        # 根据 self.log 实际长度动态生成横轴
        n_epochs = self.log.size(0)
        axis = np.arange(1, n_epochs + 1)

        for idx_data, d in enumerate(self.args.data_test):
            label = f'SR on {d}'
            fig = plt.figure()
            plt.title(label)
            for idx_scale, scale in enumerate(self.args.scale):
                # 安全检查，防止索引越界或维度不匹配
                if self.log.ndim == 3:
                    y_data = self.log[:, idx_data, idx_scale].numpy()
                elif self.log.ndim == 2:
                    y_data = self.log[:, idx_data].numpy()
                else:
                    y_data = self.log.numpy()

                # 确保横纵长度一致
                min_len = min(len(axis), len(y_data))
                plt.plot(axis[:min_len], y_data[:min_len], label=f'Scale {scale}')

            plt.legend()
            plt.xlabel('Epochs')
            plt.ylabel('PSNR')
            plt.grid(True)
            plt.savefig(self.get_path(f'test_{d}.pdf'))
            plt.close(fig)

    

    def begin_background(self):
        # Use a thread pool on macOS to avoid spawn/multiprocessing errors
        # Keep a list of futures to ensure we can wait for all IO tasks
        self.executor = ThreadPoolExecutor(max_workers=4)
        self._futures = []

    def end_background(self):
        # Wait for all scheduled image write tasks to finish and shutdown executor
        if hasattr(self, 'executor'):
            # wait for any submitted futures
            for f in getattr(self, '_futures', []):
                try:
                    f.result()
                except Exception:
                    pass
            self.executor.shutdown(wait=True)

    def save_results(self, dataset, filename, save_list, scale):
        if self.args.save_results:
            filename = self.get_path(
                'results-{}'.format(dataset.dataset.name),
                '{}_x{}_'.format(filename, scale)
            )

            postfix = ('SR', 'LR', 'HR')
            for v, p in zip(save_list, postfix):
                normalized = v[0].mul(255 / self.args.rgb_range)
                tensor_cpu = normalized.byte().permute(1, 2, 0).cpu()
                save_path = '{}{}.png'.format(filename, p)
                # schedule asynchronous write using thread pool
                if hasattr(self, 'executor'):
                    future = self.executor.submit(imageio.imwrite, save_path, tensor_cpu.numpy())
                    self._futures.append(future)
                else:
                    # fallback to synchronous write
                    imageio.imwrite(save_path, tensor_cpu.numpy())

def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)

def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    """
    Calculate PSNR.
    """
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range

    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.dim() == 4 and diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()

    return -10 * math.log10(mse)

def make_optimizer(args, target):
    '''
        make optimizer and scheduler together
    '''
    # optimizer
    trainable = filter(lambda x: x.requires_grad, target.parameters())
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'RMSprop':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['eps'] = args.epsilon

    # scheduler
    milestones = list(map(lambda x: int(x), args.decay.split('-')))
    kwargs_scheduler = {'milestones': milestones, 'gamma': args.gamma}
    scheduler_class = lrs.MultiStepLR

    class CustomOptimizer(optimizer_class):
        def __init__(self, *args, **kwargs):
            super(CustomOptimizer, self).__init__(*args, **kwargs)

        def _register_scheduler(self, scheduler_class, **kwargs):
            self.scheduler = scheduler_class(self, **kwargs)

        def save(self, save_dir):
            torch.save(self.state_dict(), self.get_dir(save_dir))

        def load(self, load_dir, epoch=1):
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            ckpt = torch.load(self.get_dir(load_dir), map_location=device)
            self.load_state_dict(ckpt)
            if epoch > 1:
                for _ in range(epoch): self.scheduler.step()
                current_lr = ckpt['param_groups'][0]['lr']
                for g in self.param_groups:
                    g['lr'] = current_lr
                self.scheduler._last_lr = [current_lr]

        def get_dir(self, dir_path):
            return os.path.join(dir_path, 'optimizer.pt')

        def schedule(self):
            self.scheduler.step()

        def get_lr(self):
            return self.scheduler.get_last_lr()[0]

        def get_last_epoch(self):
            return self.scheduler.last_epoch
    
    optimizer = CustomOptimizer(trainable, **kwargs_optimizer)
    optimizer._register_scheduler(scheduler_class, **kwargs_scheduler)
    return optimizer

def calc_ssim(sr, hr, scale, rgb_range=255):
    """
    Calculate Structural Similarity Index (SSIM).
    """
    if hr.nelement() == 1:
        return 0

    # Normalize to [0,1]
    sr = sr / rgb_range
    hr = hr / rgb_range

    # Convert to grayscale if 3 channels, else keep first channel
    if sr.size(1) == 3:
        sr_gray = 0.2989 * sr[:, 0, :, :] + 0.5870 * sr[:, 1, :, :] + 0.1140 * sr[:, 2, :, :]
        hr_gray = 0.2989 * hr[:, 0, :, :] + 0.5870 * hr[:, 1, :, :] + 0.1140 * hr[:, 2, :, :]
    else:
        sr_gray = sr[:, 0, :, :]
        hr_gray = hr[:, 0, :, :]

    # Crop edges consistent with calc_psnr strategy
    if scale > 0:
        sr_gray = sr_gray[..., scale:-scale, scale:-scale]
        hr_gray = hr_gray[..., scale:-scale, scale:-scale]

    # Define 11x11 Gaussian kernel
    def gaussian(window_size=11, sigma=1.5):
        coords = torch.arange(window_size).float() - (window_size - 1) / 2
        g = torch.exp(-(coords**2) / (2 * sigma**2))
        g /= g.sum()
        return g.unsqueeze(1) @ g.unsqueeze(0)  # 2D gaussian kernel

    kernel = gaussian().to(sr.device)
    kernel = kernel.unsqueeze(0).unsqueeze(0)  # shape (1,1,11,11)

    mu1 = F.conv2d(sr_gray.unsqueeze(1), kernel, padding=5)
    mu2 = F.conv2d(hr_gray.unsqueeze(1), kernel, padding=5)

    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(sr_gray.unsqueeze(1)**2, kernel, padding=5) - mu1_sq
    sigma2_sq = F.conv2d(hr_gray.unsqueeze(1)**2, kernel, padding=5) - mu2_sq
    sigma12 = F.conv2d((sr_gray.unsqueeze(1) * hr_gray.unsqueeze(1)), kernel, padding=5) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean().item()

if __name__ == '__main__':
    ckpt = torch.load('experiment/bffn_128/optimizer.pt',map_location='cpu')
    print("保存时 lr:", ckpt['param_groups'][0]['lr'])
    print("当前学习率:", ckpt['param_groups'][0]['lr'])
    print("优化器已走步数:", ckpt['state'][0]['step'])
    print("state字典里一共有多少个参数:", len(ckpt['state']))