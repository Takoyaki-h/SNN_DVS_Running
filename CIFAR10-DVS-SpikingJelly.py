import torch
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional, surrogate, neuron
from spikingjelly.activation_based.model import parametric_lif_net
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader, random_split
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
import matplotlib.pyplot as plt
from collections import Counter

# from torch.utils.tensorboard import SummaryWriter
import time
import os
import argparse
import datetime
from tqdm import tqdm

def main():
    # python -m spikingjelly.activation_based.examples.classify_dvsg -T 16 -device cuda:0 -b 16 -epochs 64 -data-dir /datasets/DVSGesture/ -amp -cupy -opt adam -lr 0.001 -j 8

    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=16, type=int, help='batch size')
    parser.add_argument('-epochs', default=32, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, default='/home/hp/DataSet/CIFAR10DVS/',
                        help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', default=True, help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy backend')
    parser.add_argument('-opt', type=str, default='adam', help='use which optimizer. SDG or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.001, type=float, help='learning rate')
    parser.add_argument('-channels', default=128, type=int, help='channels of CSNN')

    args = parser.parse_args()
    print(args)

    net = parametric_lif_net.CIFAR10DVSNet(channels=args.channels, spiking_neuron=neuron.LIFNode,
                                           surrogate_function=surrogate.ATan(), detach_reset=True)

    functional.set_step_mode(net, 'm')
    # if args.cupy:
    #     functional.set_backend(net, 'cupy', instance=neuron.LIFNode)

    print(net)

    net.to(args.device)

    cifar10dvs = CIFAR10DVS(root=args.data_dir, data_type='frame', frames_number=args.T,
                            split_by='number')
    # 定义训练集和测试集的划分比例
    train_size = int(0.8 * len(cifar10dvs))
    test_size = len(cifar10dvs) - train_size

    # 随机划分训练集和测试集
    train_set, test_set = random_split(cifar10dvs, [train_size, test_size])

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j,
        pin_memory=True
    )
    # train_lable = []
    # val_label = []
    #
    # for frame, label in tqdm(train_data_loader):
    #     train_lable.append(label)
    # for frame, label in tqdm(test_data_loader):
    #     val_label.append(label)
    # flat_data = [item.item() for sublist in train_lable for item in sublist]
    # flat_data1 = [item.item() for sublist in val_label for item in sublist]
    #
    # # 统计label出现的个数
    # label_counts = Counter(flat_data)
    # label_counts1 = Counter(flat_data1)
    #
    # # 输出结果
    # print("*****************************************")
    # sorted_data = {key: value for key, value in sorted(label_counts.items(), key=lambda x: x[0])}
    # sorted_data1 = {key: value for key, value in sorted(label_counts1.items(), key=lambda x: x[0])}
    #
    # print(sorted_data)
    # print(sorted_data1)

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = -1

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'T{args.T}_b{args.b}_{args.opt}_lr{args.lr}_c{args.channels}')

    if args.amp:
        out_dir += '_amp'

    # if args.cupy:
    #     out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')

    # writer = SummaryWriter(out_dir, purge_step=start_epoch)
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))
    train_acc_list = []
    test_acc_list = []
    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in tqdm(train_data_loader):
            optimizer.zero_grad()
            frame = frame.to(args.device)
            frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()

            if scaler is not None:
                with amp.autocast():
                    out_fr = net(frame).mean(0)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples

        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in tqdm(test_data_loader):
                frame = frame.to(args.device)
                frame = frame.transpose(0, 1)  # [N, T, C, H, W] -> [T, N, C, H, W]

                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = net(frame).mean(0)
                loss = F.mse_loss(out_fr, label_onehot)
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_time = time.time()
        test_speed = test_samples / (test_time - train_time)
        test_loss /= test_samples
        test_acc /= test_samples
        # writer.add_scalar('test_loss', test_loss, epoch)
        # writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(
            f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
    print("********************************************************************")
    print("train_acc_list:", train_acc_list)
    print("test_acc_list:", test_acc_list)
    print("********************************************************************")
    epoch_list = range(args.epochs)
    plt.plot(epoch_list, train_acc_list, 'b', label='Training accuracy')
    plt.plot(epoch_list, test_acc_list, 'r', label='validation accuracy')
    plt.title('Training and validation accuracy')
    plt.legend(loc='lower right')
    plt.figure()
    plt.show()


if __name__ == '__main__':
    main()
