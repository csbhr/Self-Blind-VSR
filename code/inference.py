import os
import torch
import glob
import numpy as np
import imageio
import cv2
import math
import time
import argparse
from model.pwc_recons import PWC_Recons


class Logger:
    def __init__(self, result_dir, filename='inference_log.txt'):
        self.log_file_path = os.path.join(result_dir, filename)
        open_type = 'a' if os.path.exists(self.log_file_path) else 'w'
        self.log_file = open(self.log_file_path, open_type)

    def write_log(self, log):
        print(log)
        self.log_file.write(log + '\n')


class Inference:
    def __init__(self, args):

        self.input_path = args.input_path
        self.GT_path = args.gt_path
        self.model_path = args.model_path
        self.result_path = args.result_path
        self.border = args.border
        self.save_image = args.save_image
        self.n_seq = args.n_seq
        self.scale = args.scale
        self.size_must_mode = args.size_must_mode
        self.device = args.device

        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
            print('mkdir: {}'.format(self.result_path))

        time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        infer_flag = args.infer_flag if args.infer_flag != '.' else time_str
        self.result_path = os.path.join(self.result_path, 'infer_{}'.format(infer_flag))
        if not os.path.exists(self.result_path):
            os.mkdir(self.result_path)
            print('mkdir: {}'.format(self.result_path))

        self.logger = Logger(self.result_path, 'inference_log_{}.txt'.format(time_str))

        self.logger.write_log('Inference - {}'.format(infer_flag))
        self.logger.write_log('input_path: {}'.format(self.input_path))
        self.logger.write_log('gt_path: {}'.format(self.GT_path))
        self.logger.write_log('model_path: {}'.format(self.model_path))
        self.logger.write_log('result_path: {}'.format(self.result_path))
        self.logger.write_log('border: {}'.format(self.border))
        self.logger.write_log('save_image: {}'.format(self.save_image))
        self.logger.write_log('n_seq: {}'.format(self.n_seq))
        self.logger.write_log('scale: {}'.format(self.scale))
        self.logger.write_log('size_must_mode: {}'.format(self.size_must_mode))
        self.logger.write_log('device: {}'.format(self.device))

        self.net = PWC_Recons(
            n_colors=3, n_sequence=5, extra_RBS=3, recons_RBS=20, n_feat=128, scale=4, ksize=13, device=self.device
        )
        self.net.load_state_dict(torch.load(self.model_path), strict=True)
        self.net = self.net.to(self.device)
        self.logger.write_log('Loading model from {}'.format(self.model_path))
        self.net.eval()

    def infer(self):
        with torch.no_grad():
            total_psnr = {}
            total_ssim = {}
            videos = sorted(os.listdir(self.input_path))
            for v in videos:
                video_psnr = []
                video_ssim = []
                input_frames = sorted(glob.glob(os.path.join(self.input_path, v, "*.png")))
                gt_frames = sorted(glob.glob(os.path.join(self.GT_path, v, "*.png")))
                input_seqs = self.gene_seq(input_frames, n_seq=self.n_seq)
                gt_seqs = self.gene_seq(gt_frames, n_seq=self.n_seq)
                for in_seq, gt_seq in zip(input_seqs, gt_seqs):
                    start_time = time.time()
                    filename = os.path.basename(in_seq[self.n_seq // 2]).split('.')[0]
                    inputs = [imageio.imread(p) for p in in_seq]
                    gt = imageio.imread(gt_seq[self.n_seq // 2])

                    h, w, c = inputs[self.n_seq // 2].shape
                    new_h, new_w = h - h % self.size_must_mode, w - w % self.size_must_mode
                    inputs = [im[:new_h, :new_w, :] for im in inputs]
                    gt = gt[:new_h * self.scale, :new_w * self.scale, :]

                    in_tensor = self.numpy2tensor(inputs).to(self.device)
                    preprocess_time = time.time()
                    output_dict, _ = self.net({'x': in_tensor, 'mode': 'infer'})
                    output = output_dict['recons']
                    forward_time = time.time()
                    output_img = self.tensor2numpy(output)

                    psnr, ssim = self.get_PSNR_SSIM(output_img, gt)
                    video_psnr.append(psnr)
                    video_ssim.append(ssim)
                    total_psnr[v] = video_psnr
                    total_ssim[v] = video_ssim

                    if self.save_image:
                        if not os.path.exists(os.path.join(self.result_path, v)):
                            os.mkdir(os.path.join(self.result_path, v))
                        imageio.imwrite(os.path.join(self.result_path, v, '{}.png'.format(filename)), output_img)
                    postprocess_time = time.time()

                    self.logger.write_log(
                        '> {}-{} PSNR={:.5}, SSIM={:.4} pre_time:{:.3}s, forward_time:{:.3}s, post_time:{:.3}s, total_time:{:.3}s'
                            .format(v, filename, psnr, ssim,
                                    preprocess_time - start_time,
                                    forward_time - preprocess_time,
                                    postprocess_time - forward_time,
                                    postprocess_time - start_time))

            sum_psnr = 0.
            sum_ssim = 0.
            n_img = 0
            for k in total_psnr.keys():
                self.logger.write_log("# Video:{} AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(
                    k, sum(total_psnr[k]) / len(total_psnr[k]), sum(total_ssim[k]) / len(total_ssim[k])))
                sum_psnr += sum(total_psnr[k])
                sum_ssim += sum(total_ssim[k])
                n_img += len(total_psnr[k])
            self.logger.write_log("# Total AVG-PSNR={:.5}, AVG-SSIM={:.4}".format(sum_psnr / n_img, sum_ssim / n_img))

    def gene_seq(self, img_list, n_seq):
        if self.border:
            half = n_seq // 2
            img_list_temp = img_list[:half]
            img_list_temp.extend(img_list)
            img_list_temp.extend(img_list[-half:])
            img_list = img_list_temp
        seq_list = []
        for i in range(len(img_list) - 2 * (n_seq // 2)):
            seq_list.append(img_list[i:i + n_seq])
        return seq_list

    def numpy2tensor(self, input_seq, rgb_range=1.):
        tensor_list = []
        for img in input_seq:
            img = np.array(img).astype('float64')
            np_transpose = np.ascontiguousarray(img.transpose((2, 0, 1)))  # HWC -> CHW
            tensor = torch.from_numpy(np_transpose).float()  # numpy -> tensor
            tensor.mul_(rgb_range / 255)  # (0,255) -> (0,1)
            tensor_list.append(tensor)
        stacked = torch.stack(tensor_list).unsqueeze(0)
        return stacked

    def tensor2numpy(self, tensor, rgb_range=1.):
        rgb_coefficient = 255 / rgb_range
        img = tensor.mul(rgb_coefficient).clamp(0, 255).round()
        img = img[0].data
        img = np.transpose(img.cpu().numpy(), (1, 2, 0)).astype(np.uint8)
        return img

    def get_PSNR_SSIM(self, output, gt, crop_border=4):
        cropped_output = output[crop_border:-crop_border, crop_border:-crop_border, :]
        cropped_GT = gt[crop_border:-crop_border, crop_border:-crop_border, :]
        psnr = self.calc_PSNR(cropped_GT, cropped_output)
        ssim = self.calc_SSIM(cropped_GT, cropped_output)
        return psnr, ssim

    def calc_PSNR(self, img1, img2):
        '''
        img1 and img2 have range [0, 255]
        '''
        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        mse = np.mean((img1 - img2) ** 2)
        if mse == 0:
            return float('inf')
        return 20 * math.log10(255.0 / math.sqrt(mse))

    def calc_SSIM(self, img1, img2):
        '''calculate SSIM
        the same outputs as MATLAB's
        img1, img2: [0, 255]
        '''

        def ssim(img1, img2):
            C1 = (0.01 * 255) ** 2
            C2 = (0.03 * 255) ** 2

            img1 = img1.astype(np.float64)
            img2 = img2.astype(np.float64)
            kernel = cv2.getGaussianKernel(11, 1.5)
            window = np.outer(kernel, kernel.transpose())

            mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
            mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
            mu1_sq = mu1 ** 2
            mu2_sq = mu2 ** 2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
            sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
            sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

            ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
                                                                    (sigma1_sq + sigma2_sq + C2))
            return ssim_map.mean()

        if not img1.shape == img2.shape:
            raise ValueError('Input images must have the same dimensions.')
        if img1.ndim == 2:
            return ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError('Wrong input image dimensions.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CDVD-TSP-Inference')

    parser.add_argument('--save_image', action='store_true', default=True, help='save image if true')
    parser.add_argument('--border', action='store_true', help='restore border images of video if true')

    parser.add_argument('--input_path', type=str, default='../dataset/input', help='the path of input')
    parser.add_argument('--gt_path', type=str, default='../dataset/gt', help='the path of input')
    parser.add_argument('--model_path', type=str, default='../pretrain_models.pt', help='the path of pretrain model')
    parser.add_argument('--result_path', type=str, default='../infer_results', help='the path of result')
    parser.add_argument('--infer_flag', type=str, default='.', help='the flag of this reference')
    parser.add_argument('--quick_test', type=str, default='.',
                        help='Gaussian_REDS4/Gaussian_Vid4/Gaussian_SPMCS/Realistic_REDS4')

    parser.add_argument('--n_seq', type=int, default=5, help='the number of sequence of video')
    parser.add_argument('--scale', type=int, default=4, help='the upsample scale')
    parser.add_argument('--size_must_mode', type=int, default=1, help='the input size must mode')
    parser.add_argument('--device', type=str, default='cuda', help='device')

    args = parser.parse_args()

    if args.quick_test == 'Gaussian_REDS4':
        args.scale = 4
        args.input_path = '../dataset/REDS4_BlurDown_Gaussian/LR_blurdown_x4'
        args.gt_path = '../dataset/REDS4_BlurDown_Gaussian/HR'
        args.model_path = '../pretrain_models/self_blind_vsr_gaussian.pt'
        args.infer_flag = 'Gaussian_REDS4'
    elif args.quick_test == 'Gaussian_Vid4':
        args.scale = 4
        args.input_path = '../dataset/Vid4_BlurDown_Gaussian/LR_blurdown_x4'
        args.gt_path = '../dataset/Vid4_BlurDown_Gaussian/HR'
        args.model_path = '../pretrain_models/self_blind_vsr_gaussian.pt'
        args.infer_flag = 'Gaussian_Vid4'
    elif args.quick_test == 'Gaussian_SPMCS':
        args.scale = 4
        args.input_path = '../dataset/SPMCS_BlurDown_Gaussian/LR_blurdown_x4'
        args.gt_path = '../dataset/SPMCS_BlurDown_Gaussian/HR'
        args.model_path = '../pretrain_models/self_blind_vsr_gaussian.pt'
        args.infer_flag = 'Gaussian_SPMCS'
    elif args.quick_test == 'Realistic_REDS4':
        args.scale = 4
        args.input_path = '../dataset/REDS4_BlurDown_Realistic/LR_blurdown_x4'
        args.gt_path = '../dataset/REDS4_BlurDown_Realistic/HR'
        args.model_path = '../pretrain_models/self_blind_vsr_realistic.pt'
        args.infer_flag = 'Realistic_REDS4'

    Infer = Inference(args)
    Infer.infer()
