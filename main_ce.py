from __future__ import print_function

import os
import sys
import argparse
import time
import math
import numpy as np
import matplotlib.pyplot as plt

import tensorboard_logger as tb_logger
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer, save_model

from sklearn.model_selection import StratifiedKFold
from torch.utils.data import Subset

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass

import random


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='30,60,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='effnet-b0')
    parser.add_argument('--dataset', type=str, default='pathmnist', help='dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')
    parser.add_argument('--tsne', action='store_true',default=True,
                        help='generate t-SNE visualization')
    
    opt = parser.parse_args()
    opt.seed = 28
    opt.early_stop_patience = 5

    # set the path according to the environment
    opt.data_folder = './datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.tb_path = './save/SupCon/{}_tensorboard'.format(opt.dataset)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = 'SupCE_{}_{}_lr_{}_{}'.\
        format(opt.dataset, opt.model, opt.learning_rate, opt.trial)

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    if opt.dataset == 'cifar10':
        opt.n_cls = 10
    elif opt.dataset == 'cifar100':
        opt.n_cls = 100
    elif opt.dataset == 'pathmnist':
        opt.n_cls = 9
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt

# set_loader for PathMNIST 
def set_loader(opt, fold_idx=None):
    from medmnist import PathMNIST
    from medmnist import INFO

    mean=(0.7405, 0.5330, 0.7058)
    std=(0.1237, 0.1767, 0.1244)

    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),  # 28x28 for PathMNIST
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if opt.dataset == 'pathmnist':
        full_dataset = PathMNIST(root=opt.data_folder, split='train', transform=train_transform, download=True)
        test_dataset = PathMNIST(root=opt.data_folder, split='test', transform=val_transform, download=True)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    # K-fold split
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=opt.seed)
    folds = list(skf.split(np.zeros(len(full_dataset)), full_dataset.labels.squeeze()))

    train_idx, val_idx = folds[fold_idx]

    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    g = torch.Generator()
    g.manual_seed(opt.seed)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True,
        worker_init_fn=seed_worker, generator=g
    )

    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=8, pin_memory=True
    )

    return train_loader, val_loader, test_loader


class SupCEEfficientNet(nn.Module):
    """encoder + classifier"""
    def __init__(self):
        super(SupCEEfficientNet, self).__init__()
        feat_dim = 1280
        num_classes = 9
        
        self.encoder = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)

        # Remove original classifier
        self.encoder.classifier = nn.Identity() 

        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


def set_model(opt):
    model = SupCEEfficientNet()
    criterion = torch.nn.CrossEntropyLoss()

    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = images.cuda(non_blocking=True)
        labels = labels.cuda(non_blocking=True)
        labels = labels.squeeze()
        bsz = labels.shape[0]
        
        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        output = model(images)

        loss = criterion(output, labels)

        # update metric
        losses.update(loss.item(), bsz)
        acc1, acc5 = accuracy(output, labels, topk=(1, 5))
        top1.update(acc1[0], bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1))
            sys.stdout.flush()

    return losses.avg, top1.avg


def validate(val_loader, model, criterion, opt):
    """validation"""
    model.eval()

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()

    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            labels = labels.squeeze()
            bsz = labels.shape[0]

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # update metric
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time,
                       loss=losses, top1=top1))

    print(' * Acc@1 {top1.avg:.3f}'.format(top1=top1))
    return losses.avg, top1.avg.cpu()


def plot_loss_curves(fold, epochs, train_losses, val_losses, train_accs, val_accs, save_folder):
    plt.figure(figsize=(12, 5))
    
    # Convert all inputs to numpy arrays if they're not already
    epochs = np.array(epochs)
    train_losses = np.array(train_losses)
    val_losses = np.array(val_losses)
    train_accs = np.array(train_accs)
    val_accs = np.array(val_accs)
    
    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', linewidth=2, label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', linewidth=2, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'FOLD {fold}: Training and Validation Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', linewidth=2, label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', linewidth=2, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'FOLD {fold}: Training and Validation Accuracy')
    plt.legend()
    
    # Save the figure
    plot_path = os.path.join(save_folder, f'fold{fold}_losses_accuracies_plot_ce.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f'Train+Val curves and accuracies saved to {save_folder}')


def main():
    set_seed(42)
    opt = parse_option()
    
    # Track metrics across all folds
    all_fold_metrics = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'test_acc': []
    }

    # Perform k-fold cross-validation
    for fold in range(3):
        print(f"\n=== Fold {fold+1}/5 ===")
        
        # Build fold-specific loaders
        train_loader, val_loader, test_loader = set_loader(opt, fold_idx=fold)

        # tensorboard
        logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)

        # Track early stopping metrics
        best_acc = 0
        best_val_loss = float('inf')
        epochs_no_improve = 0
        early_stop = False  # Flag to stop training

        # Re-initialize model for each fold
        model, criterion = set_model(opt)
        optimizer = set_optimizer(opt, model)

        fold_metrics = {'train_loss': [], 'val_loss': [],
                       'train_acc': [], 'val_acc': [],
                       'epoches':[]}

        # Training loop
        for epoch in range(1, opt.epochs + 1):
            # train for one epoch
            time1 = time.time()
            loss, train_acc = train(train_loader, model, criterion, optimizer, epoch, opt)
            time2 = time.time()
            print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

            # tensorboard logger
            logger.log_value('train_loss', loss, epoch)
            logger.log_value('train_acc', train_acc, epoch)
            logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)

            # evaluation
            val_loss, val_acc = validate(val_loader, model, criterion, opt)
            logger.log_value('val_loss', val_loss, epoch)
            logger.log_value('val_acc', val_acc, epoch)

            # ======== Early Stopping Check ========
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= opt.early_stop_patience:
                    early_stop = True

            # ======== Keep Best Model Checkpoint ========
            if val_acc > best_acc:
                best_acc = val_acc
                save_file = os.path.join(
                    opt.save_folder, f'fold{fold}_best.pth')
                save_model(model, optimizer, opt, epoch, save_file)
                
            # Store fold metrics
            fold_metrics['train_loss'].append(loss)
            fold_metrics['val_loss'].append(val_loss)
            fold_metrics['train_acc'].append(train_acc)
            fold_metrics['val_acc'].append(val_acc)
            fold_metrics['epoches'].append(epoch)

        # Final test evaluation
        test_loss, test_acc = validate(test_loader, model, criterion, opt)
        plot_loss_curves(fold, fold_metrics['epoches'], fold_metrics['train_loss'], fold_metrics['val_loss'], fold_metrics['train_acc'], fold_metrics['val_acc'], opt.save_folder)
       
        # Aggregate metrics
        for k in all_fold_metrics:
            if k in fold_metrics:
                all_fold_metrics[k].append(fold_metrics[k])
        all_fold_metrics['test_acc'].append(test_acc)

    # Calculate and report average metrics
    print("\nFinal Results:")
    for metric, values in all_fold_metrics.items():
        if metric == 'test_acc':
            avg = np.mean(values)
            std = np.std(values)
            print(f"Test Accuracy: {avg:.2f}% ± {std:.2f}%")  
        else:
            avg_values = np.mean(values, axis=0)
            print(f"Average {metric}:")
            for epoch, val in enumerate(avg_values):
                print(f"Epoch {epoch+1}: {val:.2f}")  


if __name__ == '__main__':
    main()