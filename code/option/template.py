def set_template(args):
    if args.template == 'Self_Blind_VSR_Gaussian':
        args.task = "FlowVideoSR"
        args.model = "PWC_Recons"
        args.save = "Self_Blind_VSR_Gaussian"
        args.data_train = 'REDS_BLURDOWN_GAUSSIAN'
        args.dir_data = '../dataset/REDS/train'
        args.data_test = 'REDS_HRLR'
        args.dir_data_test = '../dataset/REDS4_BlurDown_Gaussian'
        args.HR_in = False
        args.scale = 4
        args.patch_size = 160
        args.n_sequence = 5
        args.n_frames_per_video = 50
        args.n_feat = 128
        args.extra_RBS = 3
        args.recons_RBS = 20
        args.ksize = 13
        args.loss = '1*L1'
        args.lr = 1e-4
        args.lr_decay = 100
        args.save_middle_models = True
        args.save_images = False
        args.epochs = 500
        args.batch_size = 8
        # args.resume = True
        # args.load = args.save
        # args.test_only = True
    elif args.template == 'Self_Blind_VSR_Realistic':
        args.task = "FlowVideoSR"
        args.model = "PWC_Recons"
        args.save = "Self_Blind_VSR_Realistic"
        args.data_train = 'REDS_BLURDOWN_REALISTIC'
        args.dir_data = '../dataset/REDS/train'
        args.data_test = 'REDS_HRLR'
        args.dir_data_test = '../dataset/REDS4_BlurDown_Realistic'
        args.HR_in = False
        args.scale = 4
        args.patch_size = 160
        args.n_sequence = 5
        args.n_frames_per_video = 50
        args.n_feat = 128
        args.extra_RBS = 3
        args.recons_RBS = 20
        args.ksize = 13
        args.loss = '1*L1'
        args.lr = 1e-4
        args.lr_decay = 100
        args.save_middle_models = True
        args.save_images = False
        args.epochs = 500
        args.batch_size = 8
        # args.resume = True
        # args.load = args.save
        # args.test_only = True
    else:
        raise NotImplementedError('Template [{:s}] is not found'.format(args.template))
