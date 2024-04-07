import argparse
import os
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
from data_loaders import cifar10,cifar100,imagenet100,svhn
from functions import *
import attack
from models import *
from utils import train, val

parser = argparse.ArgumentParser()
parser.add_argument('-j','--workers',default=8, type=int,metavar='N',help='number of data loading workers')
parser.add_argument('-b','--batch_size',default=256, type=int,metavar='N',help='mini-batch size')
parser.add_argument('--seed',default=42,type=int,help='seed for initializing training. ')
parser.add_argument('--optim', default='sgd',type=str,help='optimizer')
parser.add_argument('-suffix','--suffix',default='', type=str,help='suffix')
# model configuration
parser.add_argument('-data', '--dataset',default='cifar10',type=str,help='dataset')
parser.add_argument('-arch','--model',default='vgg11',type=str,help='model')
parser.add_argument('-T','--time',default=4, type=int,metavar='N',help='snn simulation time')
# training configuration
parser.add_argument('--epochs',default=200,type=int,metavar='N',help='number of total epochs to run')
parser.add_argument('-lr','--lr',default=0.1,type=float,metavar='LR', help='initial learning rate')
parser.add_argument('-dev','--device',default='0',type=str,help='device')
# adv training configuration
parser.add_argument('-special','--special', default='l2', type=str, help='[reg, l2]')
parser.add_argument('-beta','--beta',default=5e-4, type=float,help='regulation beta')
parser.add_argument('-atk','--attack',default='', type=str,help='attack')
parser.add_argument('-eps','--eps',default=8, type=float, metavar='N',help='attack eps')
parser.add_argument('-atk_m','--attack_mode',default='bptt', type=str,help='[bptt, bptr, '']')
# only PGD
parser.add_argument('-alpha','--alpha',default=2, type=float, metavar='N', help='pgd attack alpha')
parser.add_argument('-steps','--steps',default=4, type=int, metavar='N', help='pgd attack steps')

parser.add_argument('-enc','--encoding',default='rate',type=str,help='encoding')
parser.add_argument('-atk_enc','--atk_encoding',default='rate',type=str,help='attack encoding')
parser.add_argument('--resume', action='store_true', help='resume training from checkpoint')
parser.add_argument("--INdatapath", type=str, default="", help="Path to the ImageNet100 dataset zip file")
parser.add_argument("--cache_dataset", action="store_true", help="Cache the datasets for quicker initialization. It also serializes the transforms")
parser.add_argument("--distributed", action="store_true", help="Enable distributed training")
parser.add_argument("--TET", action="store_true", help="Enable distributed training")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cpu":
    raise RuntimeError("Please use a GPU.")
def main():
    global args
    if args.dataset.lower() == 'cifar10':
        num_labels = 10
        train_dataset, val_dataset, znorm = cifar10()
    elif args.dataset.lower() == 'svhn':
        num_labels = 10
        train_dataset, val_dataset, znorm = svhn()
    elif args.dataset.lower() == 'cifar100':
        num_labels = 100
        train_dataset, val_dataset, znorm = cifar100()
    elif args.dataset.lower() == 'imagenet100':
        num_labels = 100
        train_dataset, val_dataset, znorm = imagenet100(args.INdatapath, args.cache_dataset, args.distributed)

    log_dir = '%s-checkpoints'% (args.dataset)    
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_all(args.seed)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                                            num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size,
                                              shuffle=False, num_workers=args.workers, pin_memory=True)

    model = create_model(args.model.lower(), args.encoding, args.atk_encoding, 'normal', args.time, num_labels, znorm)
    model.to(device)
    model.set_simulation_time(args.time)

    if args.attack_mode == 'bptt':
        ff = BPTT_attack
    elif args.attack_mode == 'bptr':
        ff = BPTR_attack
    else:
        ff = None

    if args.attack.lower() == 'fgsm':
        atk = attack.FGSM(model, device, forward_function=ff, eps=args.eps / 255, T=args.time)
    elif args.attack.lower() == 'fgsmh':
        atk = attack.FGSMH(model, device, forward_function=ff, eps=args.eps / 255, T=args.time)    
    elif args.attack.lower() == 'pgdh':
        atk = attack.PGDH(model, device, forward_function=ff, eps=args.eps / 255, T=args.time)    
    elif args.attack.lower() == 'pgd':
        atk = attack.PGD(model, device, forward_function=ff, eps=args.eps / 255, alpha=args.alpha / 255, steps=args.steps, T=args.time)
    elif args.attack.lower() == 'gn':
        atk = attack.GN(model, device, eps=args.eps / 255)
    else:
        atk = None

    criterion = nn.CrossEntropyLoss().to(device)

    if args.optim.lower() == 'adam' and args.special == 'l2':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.beta)
    elif args.optim.lower() == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    elif args.optim.lower() == 'sgd' and args.special == 'l2':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.beta)
    elif args.optim.lower() == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=0)
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_acc = 0

    identifier = f'{args.model}_{args.encoding}_T{args.time}'

    if atk is not None:
        identifier += f'_{args.atk_encoding}'
        identifier += '_%s[%f][%s]' %(atk.__class__.__name__, atk.eps, args.attack_mode)
    else:
        identifier += '_clean'
    if args.special == 'reg':
        identifier += '_RAT'

    # identifier += '_%s[%f]'%(args.special, args.beta)
    identifier += args.suffix

    logger = get_logger(os.path.join(log_dir, '%s.log'%(identifier)))
    
    if args.resume:
        checkpoint = torch.load(os.path.join(log_dir, '%s.pth'%(identifier)))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_acc = checkpoint['best_acc']
        logger.info('Resuming training from checkpoint: {}'.format(os.path.join(log_dir, '%s.pth'%(identifier))))
    else:
        logger.info('start training!')
        start_epoch = 0
        best_acc = 0
    for epoch in range(start_epoch, args.epochs):
        loss, acc = train(model, device, train_loader, criterion, optimizer, args.time, atk=atk, beta=args.beta, parseval=(args.special == 'reg'),TET=args.TET)
        logger.info('Epoch:[{}/{}]\t loss={:.5f}\t acc={:.3f}'.format(epoch , args.epochs, loss, acc))
        scheduler.step()
        tmp = val(model, test_loader, device, args.time)
        logger.info('Epoch:[{}/{}]\t Test acc={:.3f}\n'.format(epoch , args.epochs, tmp))

        if best_acc < tmp:
            best_acc = tmp
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc
            }
            torch.save(checkpoint, os.path.join(log_dir, '%s.pth'%(identifier)))
        logger.info('Best Test acc={:.3f}'.format(best_acc))
        print(f'Best accuracy: {best_acc:.3f}')

if __name__ == "__main__":
    main()
