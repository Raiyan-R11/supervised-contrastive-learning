from __future__ import print_function

import sys
import argparse
import os
import torch
import torch.backends.cudnn as cudnn
from torchvision.models import efficientnet_b0
from torchvision import transforms
from torch import nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from util import AverageMeter, accuracy

def parse_option():
    parser = argparse.ArgumentParser('argument for evaluation')

    parser.add_argument('--model_path', type=str, default='/home/cvteam1/cv-project/cv-project/save/SupCon/pathmnist_models/SupCon-linear_pathmnist_effnet-b0_lr_0.1_trial_512-featdim/best_model.pth',
                        help='path to the trained model checkpoint')
    parser.add_argument('--dataset', type=str, default='pathmnist', 
                        help='dataset name')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    
    opt = parser.parse_args()
    
    # set the path according to the environment
    opt.data_folder = './datasets/'
    
    # Dataset specific parameters
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
    def __init__(self, num_classes=9):
        super(LinearClassifier, self).__init__()
        feat_dim = 512  # EfficientNet-B0 output
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


def set_model(opt):
    model = SupConEfficientNet()
    classifier = LinearClassifier(num_classes=opt.n_cls)
    criterion = torch.nn.CrossEntropyLoss()

    if torch.cuda.is_available():
        model = model.cuda()
        classifier = classifier.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True
    else:
        raise NotImplementedError('This code requires GPU')

    # Load the saved model
    print(f"Loading model from {opt.model_path}")
    checkpoint = torch.load(opt.model_path, map_location='cpu')
    
    # Handle different checkpoint formats
    if 'model' in checkpoint:
        model_state_dict = checkpoint['model']
        # Handle potential DataParallel wrapper
        if list(model_state_dict.keys())[0].startswith('module.'):
            new_state_dict = {}
            for k, v in model_state_dict.items():
                k = k.replace("module.", "")
                new_state_dict[k] = v
            model_state_dict = new_state_dict
        model.load_state_dict(model_state_dict)
    else:
        print("Error: checkpoint does not contain 'model' key")
        sys.exit(1)
    
    if 'classifier' in checkpoint:
        classifier.load_state_dict(checkpoint['classifier'])
    else:
        print("Error: checkpoint does not contain 'classifier' key")
        sys.exit(1)
    
    return model, classifier, criterion


def set_loader(opt):
    from medmnist import PathMNIST
    
    # Mean and std (approximate, based on PathMNIST)
    mean = (0.7405, 0.5330, 0.7058)
    std = (0.1237, 0.1767, 0.1244)
    normalize = transforms.Normalize(mean=mean, std=std)
    
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    if opt.dataset == 'pathmnist':
        val_dataset = PathMNIST(root=opt.data_folder, split='test', transform=val_transform, download=True)
    else:
        raise ValueError('Dataset not supported: {}'.format(opt.dataset))
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True
    )
    
    return val_loader


def evaluate(val_loader, model, classifier, criterion, opt):
    """Evaluate the model on validation set and generate confusion matrix"""
    model.eval()
    classifier.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    
    # For confusion matrix
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for idx, (images, labels) in enumerate(val_loader):
            images = images.float().cuda()
            labels = labels.cuda().squeeze()
            bsz = labels.shape[0]

            # Forward pass
            features = model(images)
            output = classifier(features)
            loss = criterion(output, labels)
            
            # Get predictions
            _, preds = torch.max(output, 1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Update metrics
            losses.update(loss.item(), bsz)
            acc1, _ = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)

            if (idx + 1) % 10 == 0:
                print('Test: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})'.format(
                       idx + 1, len(val_loader), loss=losses, top1=top1))

    print('\n * Validation Accuracy: {top1.avg:.3f}%'.format(top1=top1))
    
    # Generate confusion matrix
    conf_matrix = confusion_matrix(all_targets, all_preds)
    return losses.avg, top1.avg, conf_matrix


def plot_confusion_matrix(conf_matrix, class_names, save_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    
    # Create ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    
    print(f'Confusion matrix saved to {save_path}')


def main():
    # Parse command line arguments
    opt = parse_option()
    
    # Set up data loader
    val_loader = set_loader(opt)
    
    # Build model and load saved weights
    model, classifier, criterion = set_model(opt)
    
    # Evaluate the model
    print("==> Evaluating model on validation set...")
    val_loss, val_acc, conf_matrix = evaluate(val_loader, model, classifier, criterion, opt)
    
    # Define class names for PathMNIST
    if opt.dataset == 'pathmnist':
        class_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    else:
        class_names = [str(i) for i in range(opt.n_cls)]
    
    # Save directory (use the directory of the model path)
    save_dir = os.path.dirname(opt.model_path)
    if save_dir == '':
        save_dir = '.'
    
    # Save confusion matrix
    cm_save_path = os.path.join(save_dir, 'confusion_matrix.png')
    plot_confusion_matrix(conf_matrix, class_names, cm_save_path)
    
    # Save confusion matrix as numpy array for further analysis
    np.save(os.path.join(save_dir, 'confusion_matrix.npy'), conf_matrix)
    
    print(f"Evaluation completed with accuracy: {val_acc:.2f}%")


if __name__ == '__main__':
    main()