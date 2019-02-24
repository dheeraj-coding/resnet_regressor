import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.models as models
import os
import argparse

from utils.datasets import *

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser()
parser.add_argument("--n_cpu", type=int, default=0, help="number of cpu threads to use during batch generation")
parser.add_argument("--batch_size", type=int, default=16, help="size of each image batch")
parser.add_argument("--use_cuda", type=bool, default=True, help="whether to use cuda if available")
parser.add_argument("--epoch_start", type=int, default=0, help="number of epochs")
parser.add_argument("--epochs", type=int, default=30, help="number of epochs")
parser.add_argument(
    "--checkpoint_dir", type=str, default="checkpoints/", help="directory where model checkpoints are saved"
)

opt = parser.parse_args()

dataloader = torch.utils.data.DataLoader(
    ListDataset('data/swish/train.txt', img_size=224), batch_size=opt.batch_size, shuffle=True, num_workers=opt.n_cpu
)

model = models.resnet152(pretrained=False, num_classes=4)

cuda = torch.cuda.is_available() and opt.use_cuda
if cuda:
    model = model.cuda()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

for epoch in range(opt.epoch_start, opt.epochs):
    loss_ = []
    writer = SummaryWriter()
    for batch_i, (img_path, imgs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        imgs = Variable(imgs.type(Tensor))
        targets = Variable(targets.type(Tensor), requires_grad=False)
        targets = targets[:, 0, 1:]

        output = model(imgs)

        loss = criterion(output, targets)
        loss.backward()
        loss_.append(loss.item())
        print('Step: %d, loss: %f' % (batch_i, loss.item()))
        optimizer.step()

        writer.add_scalar('data/losses/x', loss.item(), batch_i)
    print('Epoch: %d, global loss: %f' % (epoch, np.asarray(loss_).mean()))
    torch.save(model.state_dict(), os.path.join(os.getcwd(),opt.checkpoint_dir,'{}.weights'.format(epoch)))
