from __future__ import print_function

import sys
import argparse
import time
import math
import os
import torch
import torch.backends.cudnn as cudnn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt

from util import AverageMeter
from util import adjust_learning_rate, warmup_learning_rate, accuracy
from util import set_optimizer

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=128,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=50,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='10,20,30',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='effnet-b0')
    parser.add_argument('--dataset', type=str, default='pathmnist', help='dataset')
    parser.add_argument('--trial', type=str, default='0',
                        help='id for recording multiple runs')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='/home/cvteam1/cv-project/CV_Project_SupCon/save/SupCon/pathmnist_models/SimCLR_pathmnist_effnet_b0_lr_0.05_decay_0.0001_bsz_128_temp_0.07_trial_0/last.pth',
                        help='path to pre-trained model')

    opt = parser.parse_args()

    # set the path according to the environment
    opt.data_folder = './datasets/'
    
    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    if opt.cosine:
        opt.model_name = '{}_cosine'.format(opt.model_name)

    # Save path
    opt.method = opt.method = opt.ckpt.split('\\')[-2].split('_')[0]

    opt.model_path = './save/SupCon/{}_models'.format(opt.dataset)
    opt.model_name = '{}-linear_{}_{}_lr_{}_decay_{}_bsz_{}_trial_{}'.\
        format(opt.method, opt.dataset, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.trial)
    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # warm-up for large-batch training,
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

    if opt.dataset == 'pathmnist':
        opt.n_cls = 9
        opt.size = 28
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))

    return opt


class SupConEfficientNet(nn.Module):
    def __init__(self):
        super(SupConEfficientNet, self).__init__()
        self.encoder = efficientnet_b0(pretrained=True)
        self.encoder.classifier = nn.Identity()
        self.head = nn.Sequential(
            nn.Linear(1280, 1280),
            nn.ReLU(inplace=True),
            nn.Linear(1280, 512)
        )

    def forward(self, x):
        feat = self.encoder(x)
        feat = self.head(feat)
        feat = F.normalize(feat, dim=1)
        return feat

class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self):
        super(LinearClassifier, self).__init__()
        feat_dim = 512  # EfficientNet-B0 output
        num_classes = 9
        self.fc = nn.Linear(feat_dim, num_classes)

        # Apply He (Kaiming) initialization
        self._initialize_weights()


    def forward(self, features):
        return self.fc(features)
    
    def _initialize_weights(self):
        # Apply He initialization to weights
        init.kaiming_normal_(self.fc.weight, mode='fan_out', nonlinearity='relu')
        
        # Initialize biases to zeros (common practice)
        init.zeros_(self.fc.bias)


def set_model(opt):
    model = SupConEfficientNet()
    criterion = torch.nn.CrossEntropyLoss()

    for param in model.encoder.parameters():
        param.requires_grad = False

    classifier = LinearClassifier()

    print('model:',model)
    print('classifier:',classifier)
    #ckpt = torch.load(opt.ckpt, map_location='cpu')
    ckpt = torch.load(opt.ckpt, map_location='cuda:0',weights_only=False)
    state_dict = ckpt['model']

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        else:
            new_state_dict = {}
            for k, v in state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            state_dict = new_state_dict
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

        model.load_state_dict(state_dict)
    else:
        raise NotImplementedError('This code requires GPU')

    return model, classifier, criterion

# set_loader for PathMNIST 
def set_loader(opt):
    from medmnist import PathMNIST
    from medmnist import INFO

    # Mean and std (approximate, based on PathMNIST)
    mean=(0.7405, 0.5330, 0.7058)
    std=(0.1237, 0.1767, 0.1244)
    normalize = transforms.Normalize(mean=mean, std=std)
    
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=28, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    if opt.dataset == 'pathmnist':
        # Load the train and test splits directly from the PathMNIST dataset
        train_dataset = PathMNIST(root=opt.data_folder, split='train', transform=train_transform, download=True)
        test_dataset = PathMNIST(root=opt.data_folder, split='test', transform=val_transform, download=True)
        val_dataset = PathMNIST(root=opt.data_folder, split='val', transform=val_transform, download=True)
        
    else:
        raise ValueError('Dataset not supported: {}'.format(opt.dataset))
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=True,
        num_workers=opt.num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=256, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=256, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


def train(train_loader, model, classifier, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.eval()
    classifier.train()

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

        with torch.no_grad():
            features = model(images)  # This will call the full SupConEfficientNet forward
        output = classifier(features.detach())
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


def validate(val_loader, model, classifier, criterion, opt):
    """validation"""
    model.eval()
    classifier.eval()

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

            output = classifier(model(images))
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
    return losses.avg, top1.avg


def plot_loss_curves(epochs, train_losses, val_losses, train_accs, val_accs, save_folder):
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
    plt.title('Training and Validation Loss')
    plt.legend()
    
    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', linewidth=2, label='Training Accuracy')
    plt.plot(epochs, val_accs, 'r-', linewidth=2, label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    
    # Save the figure
    plot_path = os.path.join(save_folder, 'losses_accuracies_plot.png')
    plt.tight_layout()
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f'Train+Val curves and accuracies saved to {save_folder}')


def save_model(model, classifier, optimizer, opt, epoch, save_file):
    """
    Save the current model state with all necessary information
    """
    print('==> Saving model checkpoint...')
    state = {
        'model': model.state_dict(),
        'classifier': classifier.state_dict(),
        'optimizer': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, save_file)
    print(f'==> Model saved to {save_file}')


def main():
    best_acc = 0
    opt = parse_option()

    # build data loader
    train_loader, val_loader, test_loader = set_loader(opt)

    # build model and criterion
    model, classifier, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, classifier)

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    epochs = []
    
    # Initialize variable to store best model checkpoint path
    best_model_path = os.path.join(opt.save_folder, 'best_model.pth')
    best_epoch = 0

    # training routine
    for epoch in range(1, opt.epochs + 1):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        train_loss, train_acc = train(train_loader, model, classifier, criterion,
                          optimizer, epoch, opt)
        time2 = time.time()
        print('Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(
            epoch, time2 - time1, train_acc))

        # eval for one epoch
        val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
        
        # Save model if it's the best so far
        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch
            save_model(model, classifier, optimizer, opt, epoch, best_model_path)
            print(f'==> New best model saved (epoch {epoch}, accuracy: {best_acc:.2f}%)')

        # Save last model
        last_model_path = os.path.join(opt.save_folder, 'last_model.pth')
        save_model(model, classifier, optimizer, opt, epoch, last_model_path)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        epochs.append(epoch)

    plot_loss_curves(epochs, train_losses, val_losses, train_accs, val_accs, opt.save_folder)
    print('Best accuracy: {:.2f}% at epoch {}'.format(best_acc, best_epoch))
    
    # Evaluate best model on test set
    print('\n==> Evaluating best model on test set...')
    # Load the best model
    best_checkpoint = torch.load(best_model_path)
    model.load_state_dict(best_checkpoint['model'])
    classifier.load_state_dict(best_checkpoint['classifier'])
    
    # Evaluate on test set
    test_loss, test_acc = validate(test_loader, model, classifier, criterion, opt)
    print('Test accuracy with best model: {:.2f}%'.format(test_acc))


if __name__ == '__main__':
    main()
