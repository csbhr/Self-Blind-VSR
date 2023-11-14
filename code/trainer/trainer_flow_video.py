import decimal
import os
import torch
import torch.optim as optim
from tqdm import tqdm
from utils import data_utils
from trainer.trainer import Trainer
from loss import kernel_loss


class Trainer_Flow_Video(Trainer):
    def __init__(self, args, loader, my_model, my_loss, ckp):
        super(Trainer_Flow_Video, self).__init__(args, loader, my_model, my_loss, ckp)
        print("Using Trainer_Flow_Video")
        assert args.n_sequence == 5, \
            "Just support args.n_sequence=5; but get args.n_sequence={}".format(args.n_sequence)

        self.ksize = args.ksize
        self.kernel_loss_boundaries = kernel_loss.BoundariesLoss(k_size=self.ksize)
        self.kernel_loss_sparse = kernel_loss.SparsityLoss()
        self.kernel_loss_center = kernel_loss.CentralizedLoss(k_size=self.ksize, scale_factor=1. / args.scale)
        self.l1_loss = torch.nn.L1Loss()

        self.downdata_psnr_log = []
        self.cycle_psnr_log = []
        self.mid_loss_log = []
        self.cycle_loss_log = []
        self.kloss_boundaries_log = []
        self.kloss_sparse_log = []
        self.kloss_center_log = []

        if args.load != '.':
            mid_logs = torch.load(os.path.join(ckp.dir, 'mid_logs.pt'))
            self.downdata_psnr_log = mid_logs['downdata_psnr_log']
            self.cycle_psnr_log = mid_logs['cycle_psnr_log']
            self.mid_loss_log = mid_logs['mid_loss_log']
            self.cycle_loss_log = mid_logs['cycle_loss_log']
            self.kloss_boundaries_log = mid_logs['kloss_boundaries_log']
            self.kloss_sparse_log = mid_logs['kloss_sparse_log']
            self.kloss_center_log = mid_logs['kloss_center_log']

    def make_optimizer(self):
        kwargs = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        optimizer = optim.Adam([{"params": self.model.get_model().in_conv.parameters()},
                                {"params": self.model.get_model().extra_feat.parameters()},
                                {"params": self.model.get_model().fusion_conv.parameters()},
                                {"params": self.model.get_model().recons_net.parameters()},
                                {"params": self.model.get_model().upsample_layers.parameters()},
                                {"params": self.model.get_model().out_conv.parameters()},
                                {"params": self.model.get_model().kernel_net.parameters()},
                                {"params": self.model.get_model().flow_net.parameters(), "lr": 1e-6}],
                               **kwargs)
        print(optimizer)
        return optimizer

    def train(self):
        print("Now training")
        self.scheduler.step()
        self.loss.step()
        epoch = self.scheduler.last_epoch + 1
        lr = self.scheduler.get_lr()[0]
        self.ckp.write_log('Epoch {:3d} with Lr {:.2e}'.format(epoch, decimal.Decimal(lr)))
        self.loss.start_log()
        self.model.train()
        self.ckp.start_log()
        mid_loss_sum = 0.
        cycle_loss_sum = 0.
        kloss_boundaries_sum = 0.
        kloss_sparse_sum = 0.
        kloss_center_sum = 0.

        for batch, (input, _, kernel, _) in enumerate(self.loader_train):

            input = input.to(self.device)

            output_dict, mid_loss = self.model({'x': input, 'mode': 'train'})
            input_center = output_dict['input']
            input_center_cycle = output_dict['input_cycle']
            recons_down = output_dict['recons_down']
            est_kernel = output_dict['est_kernel']

            self.optimizer.zero_grad()

            loss = self.loss(recons_down, input_center)

            cycle_loss = self.l1_loss(input_center_cycle, input_center)
            cycle_loss_sum = cycle_loss_sum + cycle_loss.item()
            loss = loss + cycle_loss

            w1, w2, w3 = 0.5, 0.04, 1

            kloss_boundaries = self.kernel_loss_boundaries(est_kernel)
            kloss_boundaries_sum = kloss_boundaries_sum + kloss_boundaries.item()
            loss = loss + w1 * kloss_boundaries

            kloss_sparse = self.kernel_loss_sparse(est_kernel)
            kloss_sparse_sum = kloss_sparse_sum + kloss_sparse.item()
            loss = loss + w2 * kloss_sparse

            kloss_center = self.kernel_loss_center(est_kernel)
            kloss_center_sum = kloss_center_sum + kloss_center.item()
            loss = loss + w3 * kloss_center

            if mid_loss:  # mid loss is the loss during the model
                loss = loss + self.args.mid_loss_weight * mid_loss
                mid_loss_sum = mid_loss_sum + mid_loss.item()
            loss.backward()
            self.optimizer.step()

            self.ckp.report_log(loss.item())

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\tLoss : [total: {:.4f}]{}[cycle: {:.4f}][boundaries: {:.4f}][sparse: {:.4f}][center: {:.4f}][mid: {:.4f}]'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.ckp.loss_log[-1] / (batch + 1),
                    self.loss.display_loss(batch),
                    cycle_loss_sum / (batch + 1),
                    kloss_boundaries_sum / (batch + 1),
                    kloss_sparse_sum / (batch + 1),
                    kloss_center_sum / (batch + 1),
                    mid_loss_sum / (batch + 1)
                ))

        self.loss.end_log(len(self.loader_train))
        self.mid_loss_log.append(mid_loss_sum / len(self.loader_train))
        self.cycle_loss_log.append(cycle_loss_sum / len(self.loader_train))
        self.kloss_boundaries_log.append(kloss_boundaries_sum / len(self.loader_train))
        self.kloss_sparse_log.append(kloss_sparse_sum / len(self.loader_train))
        self.kloss_center_log.append(kloss_center_sum / len(self.loader_train))

    def test(self):
        epoch = self.scheduler.last_epoch + 1
        self.ckp.write_log('\nEvaluation:')
        self.model.eval()
        self.ckp.start_log(train=False)
        cycle_psnr_list = []
        downdata_psnr_list = []

        with torch.no_grad():
            tqdm_test = tqdm(self.loader_test, ncols=80)
            for idx_img, (input, gt, kernel, filename) in enumerate(tqdm_test):

                filename = filename[self.args.n_sequence // 2][0]

                input = input.to(self.device)
                gt = gt[:, self.args.n_sequence // 2, :, :, :].to(self.device)

                output_dict, _ = self.model({'x': input, 'mode': 'val'})
                input_center = output_dict['input']
                input_down_center = output_dict['input_down']
                input_center_cycle = output_dict['input_cycle']
                recons = output_dict['recons']
                recons_down = output_dict['recons_down']
                est_kernel = output_dict['est_kernel']

                cycle_PSNR = data_utils.calc_psnr(input_center, input_center_cycle, rgb_range=self.args.rgb_range, is_rgb=True)
                downdata_PSNR = data_utils.calc_psnr(input_center, recons_down, rgb_range=self.args.rgb_range, is_rgb=True)
                PSNR = data_utils.calc_psnr(gt, recons, rgb_range=self.args.rgb_range, is_rgb=True)
                self.ckp.report_log(PSNR, train=False)
                cycle_psnr_list.append(cycle_PSNR)
                downdata_psnr_list.append(downdata_PSNR)

                if self.args.save_images:
                    gt, input_center, recons, input_center_cycle, input_down_center, recons_down = data_utils.postprocess(
                        gt, input_center, recons, input_center_cycle, input_down_center, recons_down,
                        rgb_range=self.args.rgb_range,
                        ycbcr_flag=False,
                        device=self.device)

                    gt_kernel = self.auto_crop_kernel(kernel[:, self.args.n_sequence // 2, :, :, :])
                    gt_kernel = self.process_kernel(gt_kernel)

                    est_kernel = self.process_kernel(est_kernel)

                    save_list = [gt, input_center, recons, input_center_cycle,
                                 input_down_center, recons_down, est_kernel, gt_kernel]
                    self.ckp.save_images(filename, save_list, epoch)

            self.ckp.end_log(len(self.loader_test), train=False)
            best = self.ckp.psnr_log.max(0)
            self.ckp.write_log('[{}]\taverage Cycle-PSNR: {:.3f} Down-PSNR: {:.3f} PSNR: {:.3f} (Best: {:.3f} @epoch {})'.format(
                self.args.data_test,
                sum(cycle_psnr_list) / len(cycle_psnr_list),
                sum(downdata_psnr_list) / len(downdata_psnr_list),
                self.ckp.psnr_log[-1],
                best[0], best[1] + 1))
            self.cycle_psnr_log.append(sum(cycle_psnr_list) / len(cycle_psnr_list))
            self.downdata_psnr_log.append(sum(downdata_psnr_list) / len(downdata_psnr_list))
            if not self.args.test_only:
                self.ckp.save(self, epoch, is_best=(best[1] + 1 == epoch))
                self.ckp.plot_log(self.cycle_psnr_log, filename='cycle_psnr.pdf', title='Cycle PSNR')
                self.ckp.plot_log(self.downdata_psnr_log, filename='downdata_psnr.pdf', title='DownData PSNR')
                self.ckp.plot_log(self.mid_loss_log, filename='mid_loss.pdf', title='Mid Loss')
                self.ckp.plot_log(self.cycle_loss_log, filename='cycle_loss.pdf', title='Cycle Loss')
                self.ckp.plot_log(self.kloss_boundaries_log, filename='kloss_boundaries.pdf', title='Kernel Boundaries Loss')
                self.ckp.plot_log(self.kloss_sparse_log, filename='kloss_sparse.pdf', title='Kernel Sparse Loss')
                self.ckp.plot_log(self.kloss_center_log, filename='kloss_center.pdf', title='Kernel Center Loss')
                torch.save({
                    'downdata_psnr_log': self.downdata_psnr_log,
                    'cycle_psnr_log': self.cycle_psnr_log,
                    'mid_loss_log': self.mid_loss_log,
                    'cycle_loss_log': self.cycle_loss_log,
                    'kloss_boundaries_log': self.kloss_boundaries_log,
                    'kloss_sparse_log': self.kloss_sparse_log,
                    'kloss_center_log': self.kloss_center_log,
                }, os.path.join(self.ckp.dir, 'mid_logs.pt'))

    def auto_crop_kernel(self, kernel):
        end = 0
        for i in range(kernel.size()[2]):
            if kernel[0, 0, end, 0] == -1:
                break
            end += 1
        kernel = kernel[:, :, :end, :end]
        return kernel

    def process_kernel(self, kernel):
        mi = torch.min(kernel)
        ma = torch.max(kernel)
        kernel = (kernel - mi) / (ma - mi)
        kernel = torch.cat([kernel, kernel, kernel], dim=1)
        kernel = kernel.mul(255.).clamp(0, 255).round()
        return kernel
