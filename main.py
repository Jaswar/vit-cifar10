import torchvision as tv
import torch as th
import cv2 as cv
import numpy as np
from model import ViT
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import argparse
from utils import train_epoch, val_epoch, plot_progress


def main(args):
    device = 'cuda' if th.cuda.is_available() else 'cpu'

    train_transform = tv.transforms.Compose([tv.transforms.RandomCrop(size=args.image_size, padding=4),
                                             # tv.transforms.Resize(size=(args.image_size, args.image_size)),
                                             tv.transforms.ToTensor(),
                                             tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                     std=[0.229, 0.224, 0.225]),
                                             tv.transforms.RandomHorizontalFlip()])

    val_transform = tv.transforms.Compose([tv.transforms.ToTensor(),
                                           tv.transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                   std=[0.229, 0.224, 0.225])])

    # train_dataset = tv.datasets.ImageFolder(root='./data/imagenet-object-localization-challenge/ILSVRC/Data/CLS-LOC/train',
    #                                         transform=transform)
    train_dataset = tv.datasets.CIFAR10(root='./data', train=True, transform=train_transform, download=True)
    val_dataset = tv.datasets.CIFAR10(root='./data', train=False, transform=val_transform, download=True)

    train_loader = th.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader = th.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    model = ViT(args.image_size,
                len(train_dataset.classes),
                patch_size=args.patch_size,
                num_layers=args.num_layers,
                num_heads=args.num_heads,
                hidden_dim=args.hidden_dim,
                mlp_size=args.mlp_size,
                dropout=args.dropout)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.decay)
    loss_func = th.nn.CrossEntropyLoss()
    model = model.to(device)

    warmup_scheduler = th.optim.lr_scheduler.LambdaLR(optimizer,
                                                      lr_lambda=lambda ep: (ep + 1) / args.warmup,
                                                      verbose=True)
    cosine_scheduler = th.optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                               T_max=args.epochs,
                                                               verbose=True)
    lr_scheduler = th.optim.lr_scheduler.SequentialLR(optimizer,
                                                      schedulers=[warmup_scheduler, cosine_scheduler],
                                                      milestones=[args.warmup],
                                                      verbose=True)

    start_epoch = 1
    if args.checkpoint is not None:
        checkpoint = th.load(args.checkpoint)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch']

    print(f'Number of parameters: {sum(p.numel() for p in model.parameters())}')
    start = time.time()
    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []
    for epoch in range(start_epoch, args.epochs + 1):
        tl, ta = train_epoch(model, optimizer, loss_func, train_loader, device, start, epoch)
        train_losses.extend(tl)
        train_accuracies.extend(ta)

        vl, va = val_epoch(model, loss_func, val_loader, device, start, epoch)
        val_losses.extend(vl)
        val_accuracies.extend(va)

        lr_scheduler.step()

        if epoch % args.plot_freq == 0:
            plot_progress('plot_train', train_losses, train_accuracies)
            plot_progress('plot_val', val_losses, val_accuracies)

        # save model
        if epoch % args.save_freq == 0:
            checkpoint = {
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict()
            }
            th.save(checkpoint, f'./logs/checkpoint_{epoch}.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--epochs', default=300)
    parser.add_argument('--batch_size', default=128)
    parser.add_argument('--image_size', default=32)
    parser.add_argument('--warmup', default=10)
    parser.add_argument('--plot_freq', default=1)
    parser.add_argument('--save_freq', default=5)
    parser.add_argument('--checkpoint', default=None)

    parser.add_argument('--patch_size', default=8)
    parser.add_argument('--num_layers', default=7)
    parser.add_argument('--num_heads', default=12)
    parser.add_argument('--hidden_dim', default=384)
    parser.add_argument('--mlp_size', default=384)
    parser.add_argument('--dropout', default=0.2)

    parser.add_argument('--lr', default=0.0003)
    parser.add_argument('--decay', default=5e-5)

    args = parser.parse_args()
    main(args)
