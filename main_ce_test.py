import os
import sys
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from util import AverageMeter, accuracy

from medmnist import PathMNIST


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


def parse_option():
    parser = argparse.ArgumentParser('Model evaluation script')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='pathmnist', help='dataset')
    parser.add_argument('--data_folder', type=str, default='./data', help='path to data folder')
    
    # Model path
    parser.add_argument('--model_path', type=str, default='/home/cvteam1/cv-project/CV_Project_SupCon/save/SupCon/pathmnist_models/SupCE_pathmnist_effnet-b0_lr_0.1_0', 
                        help='path to folder containing model checkpoints')
    parser.add_argument('--output_folder', type=str, default='/home/cvteam1/cv-project/CV_Project_SupCon/save/SupCon/pathmnist_models/SupCE_pathmnist_effnet-b0_lr_0.1_0',
                        help='path to save evaluation results')
    
    # Other settings
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
    
    opt = parser.parse_args()
    opt.data_folder = './datasets/'

    # Create output folder if it doesn't exist
    if not os.path.exists(opt.output_folder):
        os.makedirs(opt.output_folder)
    
    return opt


def load_data(opt):
    """Load data for evaluation"""
    # Mean and std for PathMNIST
    mean = (0.7405, 0.5330, 0.7058)
    std = (0.1237, 0.1767, 0.1244)
    
    normalize = transforms.Normalize(mean=mean, std=std)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    
    # Load test split from PathMNIST
    test_dataset = PathMNIST(root=opt.data_folder, split='test', transform=transform, download=True)
    val_dataset = PathMNIST(root=opt.data_folder, split='val', transform=transform, download=True)
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True
    )
    
    class_names = ['ADI', 'BACK', 'DEB', 'LYM', 'MUC', 'MUS', 'NORM', 'STR', 'TUM']
    
    return test_loader, val_loader, class_names


def evaluate(data_loader, model, model_file, criterion, opt, class_names, data_type='Test'):
    """Evaluation function"""
    model.eval()
    
    # Metrics
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    
    # Store all predictions and targets for confusion matrix
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        end = time.time()
        for idx, (images, labels) in enumerate(data_loader):
            images = images.float().cuda()
            labels = labels.cuda()
            labels = labels.squeeze()
            bsz = labels.shape[0]
            
            # Forward pass
            output = model(images)
            loss = criterion(output, labels)
            
            # Get predictions
            _, predictions = torch.max(output, 1)
            
            # Store for confusion matrix
            all_predictions.extend(predictions.cpu().numpy())
            all_targets.extend(labels.cpu().numpy())
            
            # Update metrics
            losses.update(loss.item(), bsz)
            acc1, acc5 = accuracy(output, labels, topk=(1, 5))
            top1.update(acc1[0], bsz)
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            if idx % 10 == 0:
                print(f'{data_type}: [{idx}/{len(data_loader)}]\t'
                      f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                      f'Acc@1 {top1.val:.3f} ({top1.avg:.3f})')
    
    # Create and save confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - {data_type}')
    model_name = model_file.split('.')[0]
    # Save the confusion matrix plot
    cm_path = os.path.join(opt.output_folder, f'{model_name}_{data_type.lower()}_confusion_matrix.png')
    plt.tight_layout()
    print(f"Confusion matrix saved to {cm_path}")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    
    # Generate and save classification report
    clf_report = classification_report(all_targets, all_predictions, 
                                      target_names=class_names, output_dict=True)
    
    # Print summary
    print(f"\n{data_type} Results:")
    print(f"Loss: {losses.avg:.4f}")
    print(f"Accuracy: {top1.avg:.2f}%")
    
    # Calculate per-class metrics
    print("\nPer-class Performance:")
    for i, class_name in enumerate(class_names):
        precision = clf_report[class_name]['precision']
        recall = clf_report[class_name]['recall']
        f1 = clf_report[class_name]['f1-score']
        support = clf_report[class_name]['support']
        print(f"{class_name}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, Support={support}")
    
    return losses.avg, top1.avg, cm, clf_report


def main():
    opt = parse_option()
    
    # Load data
    test_loader, val_loader, class_names = load_data(opt)
    
    # Define loss function
    criterion = torch.nn.CrossEntropyLoss()
    
    # Find all model files
    model_files = [f for f in os.listdir(opt.model_path) if f.endswith('.pth')]
    
    if not model_files:
        print(f"No .pth model files found in {opt.model_path}")
        return
    
    # Evaluate each model
    results = []
    
    for model_file in model_files:
        model_path = os.path.join(opt.model_path, model_file)
        print(f"\nEvaluating model: {model_file}")
        
        # Load model
        checkpoint = torch.load(model_path, map_location='cpu')
        
        # Create model instance
        model = SupCEEfficientNet()
        
        # Handle different checkpoint formats
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
        
        # Move model to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            criterion = criterion.cuda()
            cudnn.benchmark = True
        
        # Evaluate on validation set
        print("\nEvaluating on validation set:")
        val_loss, val_acc, val_cm, val_report = evaluate(
            val_loader, model, model_file,criterion, opt, class_names, 'Validation')
        
        # Evaluate on test set
        print("\nEvaluating on test set:")
        test_loss, test_acc, test_cm, test_report = evaluate(
            test_loader, model,model_file, criterion, opt, class_names, 'Test')
        
        # Store results
        results.append({
            'model_name': model_file,
            'val_loss': val_loss,
            'val_acc': val_acc.item(),
            'test_loss': test_loss,
            'test_acc': test_acc.item(),
            'val_report': val_report,
            'test_report': test_report
        })
    
    # Summarize results
    print("\n===== Overall Results =====")
    results.sort(key=lambda x: x['test_acc'], reverse=True)
    
    for i, result in enumerate(results):
        print(f"\n{i+1}. {result['model_name']}")
        print(f"   Validation - Loss: {result['val_loss']:.4f}, Accuracy: {result['val_acc']:.2f}%")
        print(f"   Test - Loss: {result['test_loss']:.4f}, Accuracy: {result['test_acc']:.2f}%")
    
    # Save best model metrics to a separate file
    best_model = results[0]
    with open(os.path.join(opt.output_folder, 'best_model_metrics.txt'), 'w') as f:
        f.write(f"Best Model: {best_model['model_name']}\n")
        f.write(f"Validation Accuracy: {best_model['val_acc']:.2f}%\n")
        f.write(f"Test Accuracy: {best_model['test_acc']:.2f}%\n\n")
        
        f.write("Per-class Test Metrics:\n")
        for class_name in class_names:
            metrics = best_model['test_report'][class_name]
            f.write(f"{class_name}: ")
            f.write(f"Precision={metrics['precision']:.4f}, ")
            f.write(f"Recall={metrics['recall']:.4f}, ")
            f.write(f"F1={metrics['f1-score']:.4f}, ")
            f.write(f"Support={metrics['support']}\n")


if __name__ == '__main__':
    main()