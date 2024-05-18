import time
import numpy as np
import math
import pandas as pd
import torch
import torch.optim as optim
from STGMamba import *

from torch.autograd import Variable


def TrainSTG_Mamba(train_dataloader, valid_dataloader, A, K=3, num_epochs=1, mamba_features=307):
    # 'mamba_features=184' if we use Knowair dataset.
    # 'mamba_features=307' if we use PEMS04 Dataste,.
    # 'mamba_features=80' if we use HZ_Metro dataset. 
    inputs, labels = next(iter(train_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()
    input_dim = fea_size
    hidden_dim = fea_size
    output_dim = fea_size

    kfgn_mamba_args = ModelArgs(
        K=K,
        A=torch.Tensor(A),
        feature_size=A.shape[0],
        d_model=fea_size,  # hidden_dim is fea_size
        n_layer=4,
        features=mamba_features
    )

    kfgn_mamba = KFGN_Mamba(kfgn_mamba_args)
    kfgn_mamba.cuda()
    
    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    learning_rate = 1e-4
    optimizer = optim.AdamW(kfgn_mamba.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01, amsgrad=False)

    use_gpu = torch.cuda.is_available()

    interval = 100
    losses_train = []
    losses_interval_train = []
    losses_valid = []
    losses_interval_valid = []
    losses_epoch = []  # Initialize the list for epoch losses

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

            kfgn_mamba.zero_grad()

            labels = torch.squeeze(labels)
            pred = kfgn_mamba(inputs)  # Updated to use new model

            loss_train = loss_MSE(pred, labels)

            optimizer.zero_grad()
            loss_train.backward()
            optimizer.step()

            losses_train.append(loss_train.data)

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

            pred = kfgn_mamba(inputs_val)
            loss_valid = loss_MSE(pred, labels_val)
            losses_valid.append(loss_valid.data)

            trained_number += 1

            if trained_number % interval == 0:
                cur_time = time.time()
                loss_interval_train = np.around(sum(losses_train[-interval:]).cpu().numpy() / interval, decimals=8)
                losses_interval_train.append(loss_interval_train)
                loss_interval_valid = np.around(sum(losses_valid[-interval:]).cpu().numpy() / interval, decimals=8)
                losses_interval_valid.append(loss_interval_valid)
                print('Iteration #: {}, train_loss: {}, valid_loss: {}, time: {}'.format(
                    trained_number * batch_size,
                    loss_interval_train,
                    loss_interval_valid,
                    np.around([cur_time - pre_time], decimals=8)))
                pre_time = cur_time

        loss_epoch = loss_valid.cpu().data.numpy()
        losses_epoch.append(loss_epoch)

    return kfgn_mamba, [losses_train, losses_interval_train, losses_valid, losses_interval_valid]


def TestSTG_Mamba(kfgn_mamba, test_dataloader, max_speed):
    inputs, labels = next(iter(test_dataloader))
    [batch_size, step_size, fea_size] = inputs.size()

    cur_time = time.time()
    pre_time = time.time()

    use_gpu = torch.cuda.is_available()

    loss_MSE = torch.nn.MSELoss()
    loss_L1 = torch.nn.L1Loss()

    tested_batch = 0

    losses_mse = []
    losses_l1 = []
    MAEs = []
    MAPEs = []
    MSEs = []
    RMSEs = []
    VARs = []

    #predictions = []
    #ground_truths = []

    for data in test_dataloader:
        inputs, labels = data

        if inputs.shape[0] != batch_size:
            continue

        if use_gpu:
            inputs, labels = Variable(inputs.cuda()), Variable(labels.cuda())
        else:
            inputs, labels = Variable(inputs), Variable(labels)

        pred = kfgn_mamba(inputs)
        labels = torch.squeeze(labels)

        loss_mse = F.mse_loss(pred, labels)
        loss_l1 = F.l1_loss(pred, labels)
        MAE = torch.mean(torch.abs(pred - torch.squeeze(labels)))
        MAPE = torch.mean(torch.abs(pred - torch.squeeze(labels)) / torch.abs(torch.squeeze(labels)))
        # Calculate MAPE only for non-zero labels
        non_zero_labels = torch.abs(labels) > 0
        if torch.any(non_zero_labels):
            MAPE_values = torch.abs(pred - torch.squeeze(labels)) / torch.abs(torch.squeeze(labels))
            MAPE = torch.mean(MAPE_values[non_zero_labels])
            MAPEs.append(MAPE.item())

        MSE = torch.mean((torch.squeeze(labels) - pred)**2)
        RMSE = math.sqrt(torch.mean((torch.squeeze(labels) - pred)**2))
        VAR = 1-(torch.var(torch.squeeze(labels)-pred))/torch.var(torch.squeeze(labels))

        losses_mse.append(loss_mse.item())
        losses_l1.append(loss_l1.item())
        MAEs.append(MAE.item())
        MAPEs.append(MAPE.item())
        MSEs.append(MSE.item())
        RMSEs.append(RMSE)
        VARs.append(VAR.item())

        #predictions.append(pd.DataFrame(pred.cpu().data.numpy()))
        #ground_truths.append(pd.DataFrame(labels.cpu().data.numpy()))

        tested_batch += 1

        if tested_batch % 100 == 0:
            cur_time = time.time()
            print('Tested #: {}, loss_l1: {}, loss_mse: {}, time: {}'.format(
                tested_batch * batch_size,
                np.around([loss_l1.data[0]], decimals=8),
                np.around([loss_mse.data[0]], decimals=8),
                np.around([cur_time - pre_time], decimals=8)))
            pre_time = cur_time

    losses_l1 = np.array(losses_l1)
    losses_mse = np.array(losses_mse)
    MAEs = np.array(MAEs)
    MAPEs = np.array(MAPEs)
    MSEs = np.array(MSEs)
    RMSEs = np.array(RMSEs)
    VARs = np.array(VARs)

    mean_l1 = np.mean(losses_l1) * max_speed
    std_l1 = np.std(losses_l1) * max_speed
    mean_mse = np.mean(losses_mse) * max_speed
    MAE_ = np.mean(MAEs) * max_speed
    std_MAE_ = np.std(MAEs) * max_speed #std_MAE measures the consistency & stability of the model's performance across different test sets or iterations. Usually if (std_MAE/MSE)<=10%., means the trained model is good.
    MAPE_ = np.mean(MAPEs) * 100
    MSE_ = np.mean(MSEs) * (max_speed ** 2)
    RMSE_ = np.mean(RMSEs) * max_speed
    VAR_ = np.mean(VARs)
    results = [MAE_, std_MAE_, MAPE_, MSE_, RMSE_, VAR_]

    print('Tested: MAE: {}, std_MAE: {}, MAPE: {}, MSE: {}, RMSE: {}, VAR: {}'.format(MAE_, std_MAE_, MAPE_, MSE_, RMSE_, VAR_))
    return results
