import torch

import data
import model
import loss
import option
from trainer.trainer_flow_video import Trainer_Flow_Video
from logger import logger

args = option.args
torch.manual_seed(args.seed)
chkp = logger.Logger(args)

if args.task == 'FlowVideoSR':
    print("Selected task: {}".format(args.task))
    model = model.Model(args, chkp)
    loss = loss.Loss(args, chkp) if not args.test_only else None
    loader = data.Data(args)
    t = Trainer_Flow_Video(args, loader, model, loss, chkp)
    while not t.terminate():
        t.train()
        t.test()
else:
    raise NotImplementedError('Task [{:s}] is not found'.format(args.task))

chkp.done()
