from dataset.semi import SemiDataset
from dataset.semi import DatasetConsistency
from model.semseg.deeplabv2 import DeepLabV2
from model.semseg.deeplabv3plus import DeepLabV3Plus
from model.semseg.pspnet import PSPNet
from utils import count_params, meanIOU, color_map
import random
import argparse
from copy import deepcopy
import numpy as np
import os
from PIL import Image
import torch
from torch.nn import CrossEntropyLoss, DataParallel
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from loss import consistency_loss
from loss import consistency_loss_
from dataset.transform import crop, hflip, normalize, resize, blur, cutout
from dataset.transform import crop_img, hflip_img, resize_img
from torchvision import transforms
import torch.nn.functional as F
from utils import consistency_weight

MODE = None


def parse_args():
    parser = argparse.ArgumentParser(description='ST and ST++ Framework')

    # basic settings
    parser.add_argument('--data-root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['pascal', 'cityscapes', 'dataset1', 'dataset2', 'lisc'], default='pascal')
    parser.add_argument('--batch-size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--backbone', type=str, choices=['resnet50', 'resnet101'], default='resnet50')
    parser.add_argument('--model', type=str, choices=['deeplabv3plus', 'pspnet', 'deeplabv2'],
                        default='deeplabv3plus')

    # semi-supervised settings
    parser.add_argument('--labeled-id-path', type=str, required=True)
    parser.add_argument('--unlabeled-id-path', type=str, required=True)
    parser.add_argument('--pseudo-mask-path', type=str, required=True)
    parser.add_argument('--pseudo-consistency-mask-path', type=str, default = None)

    parser.add_argument('--save-path', type=str, required=True)

    # arguments for ST++
    parser.add_argument('--reliable-id-path', type=str)
    parser.add_argument('--plus', dest='plus', default=False, action='store_true',
                        help='whether to use ST++')

    args = parser.parse_args()
    return args


def main(args):
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    if not os.path.exists(args.pseudo_mask_path):
        os.makedirs(args.pseudo_mask_path)
    if args.plus and args.reliable_id_path is None:
        exit('Please specify reliable-id-path in ST++.')

    criterion = CrossEntropyLoss(ignore_index=255)

    valset = SemiDataset(args.dataset, args.data_root, 'val', None)
    valloader = DataLoader(valset, batch_size=4 if args.dataset == 'cityscapes' else 1,
                           shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    # <====================== Supervised training with labeled images (SupOnly) ======================>
    print('\n================> Total stage 1/%i: '
          'Supervised training on labeled images (SupOnly)' % (6 if args.plus else 3))

    global MODE
    MODE = 'train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size, args.labeled_id_path)
    trainset.ids = 2 * trainset.ids if len(trainset.ids) < 200 else trainset.ids
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)
    
    dataset = DatasetConsistency(args.dataset, args.data_root, 'consistency_training', args.crop_size, None, args.unlabeled_id_path)
    consistency_loader = DataLoader(dataset, batch_size=16, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    model, optimizer = init_basic_elems(args)
    print('\nParams: %.1fM' % count_params(model))

    best_model, checkpoints = train(model, trainloader, valloader, criterion, optimizer, args)#, consistency_loader)

    # <================================== Consistency_training on unlabeled images ==================================>
    # MODE = 'consistency_training'
    # print('\n================> Auxillary Stage 1.5: '
    #       'Consistency training on unlabeled images')
   
    # best_model, checkpoint = train_consistency(best_model, consistency_loader, valloader, criterion, optimizer, args)

    """
        ST framework without selective re-training
    """
    if not args.plus:
        # <============================= Pseudo label all unlabeled images =============================>
        print('\n\n\n================> Total stage 2/3: Pseudo labeling all unlabeled images')

        dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

        label(best_model, dataloader, args)

        # <======================== Re-training on labeled and unlabeled images ========================>
        print('\n\n\n================> Total stage 3/3: Re-training on labeled and unlabeled images')

        MODE = 'semi_train'

        trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                               args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
        trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                                 pin_memory=True, num_workers=16, drop_last=True)

        model, optimizer = init_basic_elems(args)

        train(model, trainloader, valloader, criterion, optimizer, args)

        return

    """
        ST++ framework with selective re-training
    """
    # <===================================== Select Reliable IDs =====================================>
    print('\n\n\n================> Total stage 2/6: Select reliable images for the 1st stage re-training')

    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, args.unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    select_reliable(checkpoints, dataloader, args)

    # <================================ Pseudo label reliable images =================================>
    print('\n\n\n================> Total stage 3/6: Pseudo labeling reliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'reliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args)

    # <================================== The 1st stage re-training ==================================>
    print('\n\n\n================> Total stage 4/6: The 1st stage re-training on labeled and reliable unlabeled images')

    MODE = 'semi_train'

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, cur_unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)

    best_model = train(model, trainloader, valloader, criterion, optimizer, args)

    # <=============================== Pseudo label unreliable images ================================>
    print('\n\n\n================> Total stage 5/6: Pseudo labeling unreliable images')

    cur_unlabeled_id_path = os.path.join(args.reliable_id_path, 'unreliable_ids.txt')
    dataset = SemiDataset(args.dataset, args.data_root, 'label', None, None, cur_unlabeled_id_path)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=4, drop_last=False)

    label(best_model, dataloader, args)

    # <================================== The 2nd stage re-training ==================================>
    print('\n\n\n================> Total stage 6/6: The 2nd stage re-training on labeled and all unlabeled images')

    trainset = SemiDataset(args.dataset, args.data_root, MODE, args.crop_size,
                           args.labeled_id_path, args.unlabeled_id_path, args.pseudo_mask_path)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True,
                             pin_memory=True, num_workers=16, drop_last=True)

    model, optimizer = init_basic_elems(args)

    train(model, trainloader, valloader, criterion, optimizer, args)


def init_basic_elems(args):
    model_zoo = {'deeplabv3plus': DeepLabV3Plus, 'pspnet': PSPNet, 'deeplabv2': DeepLabV2}
    #This is old code
    #model = model_zoo[args.model](args.backbone, 21 if args.dataset == 'pascal' else 19)
    
    #This is for dataset1 and dataset2
    model=model_zoo[args.model](args.backbone, 3)
    
    head_lr_multiple = 10.0
    if args.model == 'deeplabv2':
        assert args.backbone == 'resnet101'
        model.load_state_dict(torch.load('pretrained/deeplabv2_resnet101_coco_pretrained.pth'))
        head_lr_multiple = 1.0

    optimizer = SGD([{'params': model.backbone.parameters(), 'lr': args.lr},
                     {'params': [param for name, param in model.named_parameters()
                                 if 'backbone' not in name],
                      'lr': args.lr * head_lr_multiple}],
                    lr=args.lr, momentum=0.9, weight_decay=1e-4)

    model = DataParallel(model).cuda()

    return model, optimizer

def softmax_mse_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    #assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size() # (batch_size * num_classes * H * W)
    inputs = F.softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)

    if conf_mask:
        loss_mat = F.mse_loss(inputs, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.mean()
    else:
        return F.mse_loss(inputs, targets, reduction='mean') # take the mean over the batch_size

def softmax_kl_loss(inputs, targets, conf_mask=False, threshold=None, use_softmax=False):
    assert inputs.requires_grad == True and targets.requires_grad == False
    assert inputs.size() == targets.size()
    input_log_softmax = F.log_softmax(inputs, dim=1)
    if use_softmax:
        targets = F.softmax(targets, dim=1)
    
    if conf_mask:
        loss_mat = F.kl_div(input_log_softmax, targets, reduction='none')
        mask = (targets.max(1)[0] > threshold)
        loss_mat = loss_mat[mask.unsqueeze(1).expand_as(loss_mat)]
        if loss_mat.shape.numel() == 0: loss_mat = torch.tensor([0.]).to(inputs.device)
        return loss_mat.sum() / mask.shape.numel()
    else:
        return F.kl_div(input_log_softmax, targets, reduction='mean')

def train(model, trainloader , valloader, criterion, optimizer, args, consistency_loader = None):
    iters = 0
    total_iters = len(trainloader) * args.epochs
    iters_per_epoch = len(trainloader)

    previous_best = 0.0

    global MODE

    if MODE == 'train':
        checkpoints = []
    
    for epoch in range(args.epochs):
        print("\n==> Epoch %i, learning rate = %.4f\t\t\t\t\t previous best = %.2f" %
              (epoch, optimizer.param_groups[0]["lr"], previous_best))

        model.train()
        total_loss = 0.0
        if consistency_loader != None:
            tbar1 = tqdm(zip(trainloader, consistency_loader))
            for i, ((img, mask), (img_weak, img_strong)) in enumerate(tbar1):
                #confidence_threshold=0.95
                img, mask = img.cuda(), mask.cuda()
                img_weak, img_strong = img_weak.cuda(), img_strong.cuda()

                pred = model(img)
                loss_labeled = criterion(pred, mask)

                output_weak = model(img_weak).cpu()
                output_weak = F.softmax(output_weak)
                output_weak = torch.max(output_weak, dim=1)[0]
                output_weak = torch.where(output_weak > 0.7, output_weak, torch.zeros(output_weak.shape))

                output_strong = model(img_strong).cpu()
                output_strong = F.softmax(output_strong)
                output_strong = torch.max(output_strong, dim=1)[0]
                output_strong = torch.where(output_strong > 0.7, output_strong, torch.zeros(output_strong.shape))
                

                loss_unlabeled = softmax_mse_loss(output_strong, output_weak, True,0.9,False)
                w = consistency_weight(0.002, iters_per_epoch)(epoch, iters)
                loss_unlabeled = loss_unlabeled * w

                loss = loss_labeled + 0.01 * loss_unlabeled #+ loss_unlabeled  # Combine the losses if needed

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                iters += 1
                lr = args.lr * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

                tbar1.set_description('Loss: %.3f' % (total_loss / (i + 1)))
        else:
            tbar1 = tqdm(trainloader)
            for i, (img, mask) in enumerate(tbar1):
                img, mask = img.cuda(), mask.cuda()

                pred = model(img)
                if MODE == 'semi_train':
                    loss = softmax_kl_loss(pred, mask, True,0.7,False) 
                    w = consistency_weight(0.1, iters_per_epoch)(epoch, iters)
                    loss = loss * w
                else:
                    loss = criterion(pre, mask)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

                iters += 1
                lr = args.lr * (1 - iters / total_iters) ** 0.9
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * 1.0 if args.model == 'deeplabv2' else lr * 10.0

                tbar1.set_description('Loss: %.3f' % (total_loss / (i + 1)))
        #This is old code
        #metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
        metric = meanIOU(num_classes=3)
        model.eval()
        tbar = tqdm(valloader)

        with torch.no_grad():
            for img, mask, _ in tbar:
                img = img.cuda()
                pred = model(img)
                pred = torch.argmax(pred, dim=1)

                metric.add_batch(pred.cpu().numpy(), mask.numpy())
                mIOU = metric.evaluate()[-1]

                tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

        mIOU *= 100.0
        if mIOU > previous_best:
            if previous_best != 0:
                os.remove(os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, previous_best)))
            previous_best = mIOU
            torch.save(model.module.state_dict(),
                       os.path.join(args.save_path, '%s_%s_%.2f.pth' % (args.model, args.backbone, mIOU)))

            best_model = deepcopy(model)

        if MODE == 'train' and ((epoch + 1) in [args.epochs // 3, args.epochs * 2 // 3, args.epochs]):
            checkpoints.append(deepcopy(model))

    if MODE == 'train':
        return best_model, checkpoints

    return best_model

def val_inference(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    #This is old code
    #metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
    metric = meanIOU(num_classes=3)

    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            pred = model(img, True)
            pred = torch.argmax(pred, dim=1).cpu()

            # Create a new figure
            plt.figure(figsize=(10,10))

            plt.subplot(1, 3, 1)
            plt.imshow(img.cpu().detach().numpy().squeeze(0).transpose(1,2,0))
            plt.title('Image')

            # Plot `mask` on the left
            plt.subplot(1, 3, 2)
            plt.imshow(mask.numpy().squeeze(0), cmap='gray')
            plt.title('Mask')

            # Plot `pred` on the right
            plt.subplot(1, 3, 3)
            plt.imshow(pred.numpy().squeeze(0), cmap='gray')
            plt.title('Prediction')

            plt.savefig('output.png')
            
            # Display the figure
            plt.show()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]
            

            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)
            pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))
            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))

def select_reliable(models, dataloader, args):
    if not os.path.exists(args.reliable_id_path):
        os.makedirs(args.reliable_id_path)

    for i in range(len(models)):
        models[i].eval()
    tbar = tqdm(dataloader)

    id_to_reliability = []

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()

            preds = []
            for model in models:
                preds.append(torch.argmax(model(img), dim=1).cpu().numpy())

            mIOU = []
            for i in range(len(preds) - 1):
                #metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
                metric = meanIOU(num_classes=3)
                metric.add_batch(preds[i], preds[-1])
                mIOU.append(metric.evaluate()[-1])

            reliability = sum(mIOU) / len(mIOU)
            id_to_reliability.append((id[0], reliability))

    id_to_reliability.sort(key=lambda elem: elem[1], reverse=True)
    with open(os.path.join(args.reliable_id_path, 'reliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[:int(3/4 * len(id_to_reliability))]:
            f.write(elem[0] + '\n')
    with open(os.path.join(args.reliable_id_path, 'unreliable_ids.txt'), 'w') as f:
        for elem in id_to_reliability[int(3/4 * len(id_to_reliability)):]:
            f.write(elem[0] + '\n')

def label(model, dataloader, args):
    model.eval()
    tbar = tqdm(dataloader)

    global MODE
    #This is old code
    #metric = meanIOU(num_classes=21 if args.dataset == 'pascal' else 19)
    metric = meanIOU(num_classes=3)

    cmap = color_map(args.dataset)

    with torch.no_grad():
        for img, mask, id in tbar:
            img = img.cuda()
            pred = model(img, True)
            pred = torch.argmax(pred, dim=1).cpu()


            # # Create a new figure
            # plt.figure(figsize=(10,10))

            # plt.subplot(1, 3, 1)
            # plt.imshow(img.cpu().detach().numpy().squeeze(0).transpose(1,2,0))
            # plt.title('Image')

            # # Plot `mask` on the left
            # plt.subplot(1, 3, 2)
            # plt.imshow(mask.numpy().squeeze(0), cmap='gray')
            # plt.title('Mask')

            # # Plot `pred` on the right
            # plt.subplot(1, 3, 3)
            # plt.imshow(pred.numpy().squeeze(0), cmap='gray')
            # plt.title('Prediction')

            
            # # Display the figure
            # plt.show()

            metric.add_batch(pred.numpy(), mask.numpy())
            mIOU = metric.evaluate()[-1]
            
            pred = Image.fromarray(pred.squeeze(0).numpy().astype(np.uint8), mode='P')
            pred.putpalette(cmap)
            if args.pseudo_consistency_mask_path is None:
                pred.save('%s/%s' % (args.pseudo_mask_path, os.path.basename(id[0].split(' ')[1])))
            else:
                pred.save('%s/%s' % (args.pseudo_consistency_mask_path, os.path.basename(id[0].split(' ')[1])))
            
            tbar.set_description('mIOU: %.2f' % (mIOU * 100.0))


if __name__ == '__main__':
    args = parse_args()

    if args.epochs is None:
        args.epochs = {'pascal': 80, 'cityscapes': 240, 'dataset1': 100, 'dataset2': 100, 'lisc': 100}[args.dataset]
    if args.lr is None:
        args.lr = {'pascal': 0.001, 'cityscapes': 0.004, 'dataset1': 0.0009, 'dataset2': 0.0009, 'lisc': 0.0009}[args.dataset] / 16 * args.batch_size
    if args.crop_size is None:
        if args.dataset == 'dataset1':
            args.crop_size = {'pascal': 321, 'cityscapes': 721, 'dataset1': 128}[args.dataset]
        elif args.dataset == 'dataset2':
            args.crop_size = {'pascal': 321, 'cityscapes': 721, 'dataset2': 320}[args.dataset]
        elif args.dataset == 'lisc':
            args.crop_size = {'pascal': 321, 'cityscapes': 721, 'dataset2': 320, 'lisc': 128}[args.dataset]

    print()
    print(args)

    main(args)
