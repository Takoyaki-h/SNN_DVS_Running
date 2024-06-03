import torch
import sys
import torch.nn.functional as F
from torch.cuda import amp
from spikingjelly.activation_based import functional
from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
from ResNet18_d import ResNet18
from VGG9 import VGG9
import time
import os
import argparse
import datetime
from tqdm import tqdm
import matplotlib.pyplot as plt
from LayerPolicyNet import LayerPolicyNet


def get_reward(preds, lable, max_threshold):
    # 选择的阈值越大，奖励越大
    sparse_reward = (2 - max_threshold) ** 2

    # 预测正确为1，错误为0
    match = (preds.argmax(1) == lable)
    reward = sparse_reward
    # 预测错误的
    reward[~match] = -1
    # reward = reward.unsqueeze(1)

    return reward


def main():
    parser = argparse.ArgumentParser(description='Classify DVS Gesture')
    parser.add_argument('-T', default=5, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=400, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data-dir', type=str, default="/home/hp/DataSet/DVS128Gesture/",
                        help='root dir of DVS Gesture dataset')
    parser.add_argument('-out-dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')
    parser.add_argument('-opt', type=str, default="sgd", help='use which optimizer. SGD or Adam')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-amp', action='store_true', default=True, help='automatic mixed precision training')
    parser.add_argument('-model', default='VGG9', type=str, help='model use')
    parser.add_argument('-neuron', default='PVBMLYLIFH(Hard)', type=str, help='neuron use')
    parser.add_argument('-manual_seed', default='100', type=str, help='manual_seed')
    parser.add_argument('-len_prob', default='9', type=str, help='policynet output')
    parser.add_argument('-vth_list', default='0.7-1.5', type=str, help='vth_list')
    parser.add_argument('-drop_rate', default='0.2', type=str, help='drop_rate')

    # 设置随机数种子
    torch.manual_seed(100)
    args = parser.parse_args()
    print(args)

    net = VGG9(11)
    # net = ResNet18(11)
    functional.set_step_mode(net, 'm')

    print(net)

    net.to(args.device)

    train_set = DVS128Gesture(root=args.data_dir, train=True, data_type='frame', frames_number=args.T,
                              split_by='number')
    test_set = DVS128Gesture(root=args.data_dir, train=False, data_type='frame', frames_number=args.T,
                             split_by='number')

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

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()
    start_epoch = 0
    max_test_acc = -1
    max_epoch = 0

    optimizer = None
    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    out_dir = os.path.join(args.out_dir,
                           f'T{args.T}_{args.model}_{args.neuron}_seed{args.manual_seed}_b{args.b}_{args.opt}_lr{args.lr}_policyout{args.len_prob}')

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        print(f'Mkdir {out_dir}.')
    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))
        args_txt.write('\n')
        args_txt.write(' '.join(sys.argv))
    train_acc_list = []
    test_acc_list = []
    start = time.time()
    thresholds_list = [0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.3, 1.4, 1.5]
    len_list = len(thresholds_list)
    p_net = LayerPolicyNet(state_dim=2 * 128 * 128, num_layers=8,
                           num_thresholds=len_list)
    p_net.eval().to(args.device)
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
            label_onehot = F.one_hot(label, 11).float()

            if scaler is not None:
                with amp.autocast():
                    # out_fr = net(frame).mean(0)
                    TimeStep = frame.shape[0]
                    bs = frame.shape[1]
                    # data.shape=[bs*T,2,128,128]
                    data = frame.reshape((TimeStep * bs,) + frame.shape[2:])
                    # p_net的输出应该是[bs*T,11]
                    # p_net = LayerPolicyNet(state_dim=data.shape[1] * data.shape[2] * data.shape[3], num_layers=8,
                    #                        num_thresholds=len_list)
                    # p_net.to(args.device)
                    # vth_pro的shape为[bs(t,8,11]
                    vth_pro = p_net(data)
                    # max_prob_index.shape=[bs]
                    max_prob_index = torch.argmax(vth_pro, dim=2)
                    # 每个样本一个vth
                    thresholds_tensor = torch.tensor(thresholds_list)
                    thresholds_tensor = thresholds_tensor.to(args.device)
                    # max_threshold.shape=[bs*T,8]
                    max_threshold = thresholds_tensor[max_prob_index]
                    net.set_max_threshold(max_threshold)
                    out_fr = net(data)
                    o = torch.zeros((bs,) + out_fr.shape[1:], device=out_fr.device)
                    for t in range(TimeStep):
                        o += out_fr[t * bs:(t + 1) * bs, ...]
                    o /= TimeStep
                    # 先对layer层面求平均
                    mean_values = torch.mean(max_threshold, dim=1, keepdim=True)
                    flattened_means = mean_values.squeeze(1)
                    th = torch.zeros((bs,) + flattened_means.shape[1:], device=max_threshold.device)
                    for t in range(TimeStep):
                        th += flattened_means[t * bs:(t + 1) * bs, ...]
                    th /= TimeStep
                    # 根据vth_pro添加loss
                    loss = -th * torch.log(th).sum()
                    # o.shape=[bs,11] lable.shape=[bs],th.shape=[bs], loss.shape=[bs],reward.shape=[bs,1]
                    reward = get_reward(o, label, th)
                    loss = loss.to(args.device)
                    reward = reward.to(args.device)
                    mse_loss = F.mse_loss(o, label_onehot)
                    loss = mse_loss - 0.1 * loss * reward
                    loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                train_samples += label.numel()
                train_loss += loss.item() * label.numel()
                train_acc += (o.argmax(1) == label).float().sum().item()
            else:
                print("------else-----")
                # y_frame = []
                # for t in range(frame.shape[0]):
                #     frame_t = frame[t]
                #     frame_t = net(frame_t)
                #     y_frame.append(frame_t)
                TimeStep = frame.shape[0]
                bs = frame.shape[1]
                data = frame.reshape((TimeStep * bs,) + frame.shape[2:])
                out_fr = net(data)
                o = torch.zeros((bs,) + out_fr.shape[1:], device=out_fr.device)
                for t in range(TimeStep):
                    o += out_fr[t * bs:(t + 1) * bs, ...]
                o /= TimeStep
                # out_fr = out_fr.mean(0)
                loss = F.mse_loss(o, label_onehot)
                loss.backward()
                optimizer.step()

                train_samples += label.numel()
                train_loss += loss.item() * label.numel()
                train_acc += (o.argmax(1) == label).float().sum().item()

            functional.reset_net(net)

        train_time = time.time()
        train_speed = train_samples / (train_time - start_time)
        train_loss /= train_samples
        train_acc /= train_samples
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
                label_onehot = F.one_hot(label, 11).float()
                # y_frame = []
                # for t in range(frame.shape[0]):
                #     frame_t = frame[t]
                #     frame_t = net(frame_t)
                #     y_frame.append(frame_t)
                TimeStep = frame.shape[0]
                bs = frame.shape[1]
                data = frame.reshape((TimeStep * bs,) + frame.shape[2:])
                # p_net = LayerPolicyNet(data.shape[1] * data.shape[2] * data.shape[3], 1000, TimeStep)
                # p_net.to(args.device)
                vth_pro = p_net(data)
                max_prob_index = torch.argmax(vth_pro, dim=2)
                # 每个样本一个vth
                thresholds_tensor = torch.tensor(thresholds_list)
                thresholds_tensor = thresholds_tensor.to(args.device)
                # max_threshold.shape=[bs*T,8]
                max_threshold = thresholds_tensor[max_prob_index]
                net.set_max_threshold(max_threshold)
                out_fr = net(data)
                o = torch.zeros((bs,) + out_fr.shape[1:], device=out_fr.device)
                for t in range(TimeStep):
                    o += out_fr[t * bs:(t + 1) * bs, ...]
                o /= TimeStep
                # 先对layer层面求平均
                mean_values = torch.mean(max_threshold, dim=1, keepdim=True)
                flattened_means = mean_values.squeeze(1)
                th = torch.zeros((bs,) + flattened_means.shape[1:], device=max_threshold.device)
                for t in range(TimeStep):
                    th += flattened_means[t * bs:(t + 1) * bs, ...]
                th /= TimeStep
                # 根据vth_pro添加loss
                loss = -th * torch.log(th).sum()
                # o.shape=[bs,11] lable.shape=[bs],th.shape=[bs], loss.shape=[bs],reward.shape=[bs,1]
                reward = get_reward(o, label, th)
                loss = loss.to(args.device)
                reward = reward.to(args.device)
                mse_loss = F.mse_loss(o, label_onehot)
                loss = mse_loss - 0.1 * loss * reward
                loss = loss.mean()
                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (o.argmax(1) == label).float().sum().item()
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
            max_epoch = epoch
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
            f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}, max_epoch ={max_epoch}')
        print(f'train speed ={train_speed: .4f} images/s, test speed ={test_speed: .4f} images/s')
        print(
            f'escape time = {(datetime.datetime.now() + datetime.timedelta(seconds=(time.time() - start_time) * (args.epochs - epoch))).strftime("%Y-%m-%d %H:%M:%S")}\n')
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("********************************************************************")
        print("thresholds_list:", thresholds_list)
        print("********************************************************************")
        print("train_acc_list:", train_acc_list)
        print("test_acc_list:", test_acc_list)
        print("********************************************************************")
    end = time.time()
    print("********************************************************************")
    print("totalTime:", end - start)
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
