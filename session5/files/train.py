from __future__ import print_function
import numpy as np
import os
import scipy.io as sio
import torch
import argparse
from Networks import CNN_Net as Net
# from Networks import FC_Net as Net
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
from torch.optim.lr_scheduler import StepLR
from prepare_data import MyData
# import my_transform


def normalize_dataset(data):
    mean = data.mean((1, 2))
    std = data.std((1, 2))
    normalize = transforms.Normalize(mean=mean, std=std)
    return normalize(data.type(torch.float32))


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def evaluation(args, model, device, test_loader, optimizer, val_loss_min, epoch):
    model.eval()
    val_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    val_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        val_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # save model if validation loss has decreased
    if val_loss <= val_loss_min:
        if args.save_model:
            filename = 'model_epock_{0}_val_loss_{1}.pt'.format(epoch, val_loss)
            torch.save({'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()},
                       filename)
            # torch.save(model.state_dict())
            val_loss_min = val_loss
        return val_loss_min
    else:
        return None


def main():
    # argparse = argparse.parse_args()
    parser = argparse.ArgumentParser(description='PyTorch SVHN("http://ufldl.stanford.edu/housenumbers/") Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=14, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.4, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.05, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=True,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--weight', default=False,
                        help='path of pretrain weights')
    parser.add_argument('--resume', default=False,
                        help='path of resume weights , "./cnn_83.pt" OR "./FC_83.pt" OR False ')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs_train = {'batch_size': args.batch_size}
    kwargs_train.update({'num_workers': 1,
                         'shuffle': True,
                         'drop_last': True},
                        )
    kwargs_val = {'batch_size': args.test_batch_size}
    kwargs_val.update({'num_workers': 1,
                       'shuffle': True}
                      )

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=3, gamma=args.gamma)

    if args.weight:
        if os.path.isfile(args.weight):
            checkpoint = torch.load(args.weight)
            try:
                model.load_state_dict(checkpoint['state_dict'])
            except:
                model.load_state_dict(checkpoint)

    # args.resume = False
    if args.resume:
        if os.path.isfile(args.resume):
            # checkpoint = torch.load(args.resume, map_location=lambda storage, loc: storage.cuda())
            checkpoint = torch.load(args.resume)
            try:
                args.start_epoch = checkpoint['epoch']
                model.load_state_dict(checkpoint['state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer'])
            except:
                model.load_state_dict(checkpoint)

    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        normalize_dataset
    ])

    ##################################
    # custom_transform = my_transform.Compose([
    #     transform.RandScale([args.scale_min, args.scale_max]),
    #     transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean, ignore_label=args.ignore_label),
    #     transform.RandomGaussianBlur(),
    #     transform.RandomHorizontalFlip(),
    #     transform.Crop([args.train_h, args.train_w], crop_type='rand', padding=mean, ignore_label=args.ignore_label),
    #     transform.ToTensor(),
    #     transform.Normalize(mean=mean, std=std)])
    #####################################

    train_data = sio.loadmat("./data/train_32x32.mat")
    train_data["X"] = np.transpose(train_data["X"], (3, 0, 1, 2))
    train_data = [train_data["X"], np.int64(np.ravel(train_data["y"]))]
    MyData_train = MyData(data=train_data[0], label=train_data[1] - 1, transform=transform)

    val_data = sio.loadmat("./data/test_32x32.mat")
    val_data["X"] = np.transpose(val_data["X"], (3, 0, 1, 2))
    test_data = [val_data["X"], np.int64(np.ravel(val_data["y"]))]
    MyData_val = MyData(data=test_data[0], label=test_data[1] - 1, transform=transform)

    train_loader = torch.utils.data.DataLoader(MyData_train, **kwargs_train)
    val_loader = torch.utils.data.DataLoader(MyData_val, **kwargs_val)

    val_loss_min = np.Inf
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        scheduler.step()
        out_loss = evaluation(args, model, device, val_loader, optimizer, val_loss_min, epoch)
        if out_loss is not None:
            val_loss_min = out_loss


if __name__ == '__main__':
    main()
