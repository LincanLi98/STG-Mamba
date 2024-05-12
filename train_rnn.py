import torch
import numpy as np
from torch.autograd import Variable
import time
import math
from RNN import *


def TrainLSTM(train_dataloader, valid_dataloader, num_epochs=3):
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size

    lstm = LSTM(input_dim, hidden_dim, output_dim)

    lstm.cuda()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    learning_rate = 1e-4
    optimizer = torch.optim.RMSprop(lstm.parameters(), lr=learning_rate)

    use_gpu = torch.cuda.is_available()

    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []

    losses_epoch = []

    cur_time = time.time()
    pre_time = time.time()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        trained_number = 0

        valid_dataloader_iter = iter(valid_dataloader)

        for data in train_dataloader:
            inputs, labels = data

            if inputs.shape[0] != batch_size:
                continue

            if use_gpu:
                inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
            else:
                inputs, labels = Variable(inputs), Variable(labels)

            labels = torch.squeeze(labels)

            lstm.zero_grad()

            Hidden_State, Cell_State = lstm.loop(inputs)

            loss_train = loss_MSE(Hidden_State, labels)

            losses_train.append(loss_train.data)

            optimizer.zero_grad()

            loss_train.backward()

            optimizer.step()

            # validation
            try:
                inputs_val, labels_val = next(valid_dataloader_iter)
            except StopIteration:
                valid_dataloader_iter = iter(valid_dataloader)
                inputs_val, labels_val = next(valid_dataloader_iter)

            if use_gpu:
                inputs_val, labels_val = Variable(inputs_val.cuda()), Variable(labels_val.cuda())
            else:
                inputs_val, labels_val = Variable(inputs_val), Variable(labels_val)

            labels_val = torch.squeeze(labels_val)

            Hidden_State, Cell_State = lstm.loop(inputs_val)

            loss_valid = loss_MSE(Hidden_State, labels_val)
            losses_valid.append(loss_valid.data)

            # output
            trained_number += 1

            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy() / interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy() / interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format( \
                    trained_number * batch_size, \
                    loss_interval_train, \
                    loss_interval_valid, \
                    np.around([cur_time - pre_time], decimals=8)))
                pre_time = cur_time

        loss_epoch = loss_valid.cpu().data.numpy()
        losses_epoch.append(loss_epoch)

    return lstm, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]



def TestLSTM(lstm, test_dataloader, max_speed):
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()

    use_gpu = torch.cuda.is_available()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.MSELoss()

    tested_batch = 0

    losses_mse = []
    losses_l1 = []
    MAEs = []
    MAPEs = []
    MSEs = []
    MSPEs = []
    RMSEs = []
    R2s = []


    for data in test_dataloader:
        inputs, labels = data

        if inputs.shape[0] != batch_size:
            continue

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        Hidden_State, Cell_State = lstm.loop(inputs)

        labels = torch.squeeze(labels)

        loss_MSE = torch.nn.MSELoss()
        loss_L1 = torch.nn.L1Loss()
        loss_mse = loss_MSE(Hidden_State, labels)
        loss_l1 = loss_L1(Hidden_State, labels)
        MAE = torch.mean(torch.abs(Hidden_State - torch.squeeze(labels)))
        MAPE = torch.mean(torch.abs(Hidden_State - torch.squeeze(labels)) / torch.squeeze(labels))
        MSE = torch.mean((torch.squeeze(labels) - Hidden_State)**2)
        RMSE = math.sqrt(torch.mean((torch.squeeze(labels) - Hidden_State)**2))
        R2 = 1-((torch.squeeze(labels)-Hidden_State)**2).sum()/(((torch.squeeze(labels))-(torch.squeeze(labels)).mean())**2).sum()

        losses_mse.append(loss_mse.data)
        losses_l1.append(loss_l1.data)
        MAEs.append(MAE.data)
        MAPEs.append(MAPE.data)
        MSEs.append(MSE.item())
        RMSEs.append(RMSE)
        R2s.append(R2.item())

        tested_batch += 1

        if tested_batch % 1000 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format( \
                tested_batch * batch_size, \
                np.around([loss_l1.data[0]], decimals=8), \
                np.around([loss_mse.data[0]], decimals=8), \
                np.around([cur_time - pre_time], decimals=8)))
            pre_time = cur_time

    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    MAEs = np.array(MAEs)
    MAPEs = np.array(MAPEs)
    MSEs = np.array(MSEs)
    RMSEs = np.array(RMSEs)
    R2s = np.array(R2s)

    mean_l1 = np.mean(losses_l1) * max_speed
    std_l1 = np.std(losses_l1) * max_speed
    MAE_ = np.mean(MAEs) * max_speed
    std_MAE_ = np.std(MAEs) * max_speed
    MAPE_ = np.mean(MAPEs) * 100
    MSE_ = np.mean(MSEs) * (max_speed ** 2)
    RMSE_ = np.mean(RMSEs) * max_speed
    R2_ = np.mean(R2s)

    results = [MAE_, std_MAE_, MAPE_, MSE_, RMSE_, R2_]

    print('Tested: MAE: {}, std_MAE: {}, MAPE: {}, MSE: {}, RMSE: {}, R2: {}'.format(MAE_, std_MAE_, MAPE_, MSE_, RMSE_, R2_))
    return results
