import os
import math
from decimal import Decimal

import utility

import torch
import torch.nn.utils as utils
from tqdm import tqdm

#from torch.cuda.amp import autocast, GradScaler
from torch.amp import autocast, GradScaler


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.optimizer = utility.make_optimizer(args, self.model)

        # AMP 设置（可选）
        self.use_amp = args.use_amp
        if self.use_amp:
            self.scaler = GradScaler(self.model.device.type)
        else:
            self.scaler = None

        # 打印 AMP 状态（只执行一次）
        self.ckp.write_log(f"AMP enabled: {self.use_amp}")

        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        self.error_last = 1e8

    def train(self):
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()
            self.optimizer.zero_grad()
            
            if self.use_amp:
                # 使用自动混合精度
                with autocast(device_type=self.model.device.type):
                    sr = self.model(lr, 0)
                    loss = self.loss(sr, hr)
                self.scaler.scale(loss).backward()
                if self.args.gclip > 0:
                    self.scaler.unscale_(self.optimizer)
                    utils.clip_grad_value_(self.model.parameters(), self.args.gclip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # 标准精度训练
                sr = self.model(lr, 0)
                loss = self.loss(sr, hr)
                loss.backward()
                if self.args.gclip > 0:
                    utils.clip_grad_value_(self.model.parameters(), self.args.gclip)
                self.optimizer.step()

            # self.optimizer.zero_grad()
            # sr = self.model(lr, 0)
            # loss = self.loss(sr, hr)
            # loss.backward()
            # if self.args.gclip > 0:
            #     utils.clip_grad_value_(self.model.parameters(), self.args.gclip)
            # self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        self.ckp.write_log('\nEvaluation:')
        # 每个 epoch 新增一行 log（PSNR 和 SSIM 各一份）
        self.ckp.add_log(torch.zeros(1, len(self.loader_test), len(self.scale)))
        if not hasattr(self.ckp, 'log_ssim'):
            self.ckp.log_ssim = torch.zeros_like(self.ckp.log)
        else:
            # 累积一行以保存新 epoch 的 SSIM
            new_ssim_log = torch.zeros(1, len(self.loader_test), len(self.scale))
            self.ckp.log_ssim = torch.cat([self.ckp.log_ssim, new_ssim_log], dim=0)
        self.model.eval()

        timer_test = utility.timer()
        if self.args.save_results: self.ckp.begin_background()
        for idx_data, d in enumerate(self.loader_test):
            for idx_scale, scale in enumerate(self.scale):
                d.dataset.set_scale(idx_scale)
                for lr, hr, filename in tqdm(d, ncols=80):
                    lr, hr = self.prepare(lr, hr)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range)

                    save_list = [sr]
                    self.ckp.log[-1, idx_data, idx_scale] += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range, dataset=d
                    )
                    ssim_val = utility.calc_ssim(sr, hr, scale, self.args.rgb_range, dataset=d)
                    self.ckp.log_ssim[-1, idx_data, idx_scale] += ssim_val
                    if self.args.save_gt:
                        save_list.extend([lr, hr])

                    if self.args.save_results:
                        self.ckp.save_results(d, filename[0], save_list, scale)

                self.ckp.log[-1, idx_data, idx_scale] /= len(d)
                self.ckp.log_ssim[-1, idx_data, idx_scale] /= len(d)
                best_psnr = self.ckp.log.max(0)
                best_ssim = self.ckp.log_ssim.max(0)
                self.ckp.write_log(
                    '[{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {}) | SSIM: {:.4f} (Best: {:.4f} @epoch {})'.format(
                        d.dataset.name,
                        scale,
                        self.ckp.log[-1, idx_data, idx_scale],
                        best_psnr[0][idx_data, idx_scale],
                        best_psnr[1][idx_data, idx_scale] + 1,
                        self.ckp.log_ssim[-1, idx_data, idx_scale],
                        best_ssim[0][idx_data, idx_scale],
                        best_ssim[1][idx_data, idx_scale] + 1
                    )
                )

        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')
        self.ckp.write_log(f"[Checkpoint] Epoch {epoch}: model saved")

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best_psnr[1][0, 0] + 1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        if self.args.cpu:
            device = torch.device('cpu')
        else:
            if torch.backends.mps.is_available():
                device = torch.device('mps')
            elif torch.cuda.is_available():
                device = torch.device('cuda')
            else:
                device = torch.device('cpu')
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs
