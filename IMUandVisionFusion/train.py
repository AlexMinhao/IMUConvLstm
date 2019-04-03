import torch
from torch.autograd import Variable
import time
import os
import sys

from helper import *
from utils import *
from definitions import *
from sklearn.metrics import f1_score

def train_epoch(epoch, train_loader, model, loss_function, optimizer,
                train_logger, train_batch_logger, total, correct, f1_train_total):
    print('train at epoch {}'.format(epoch+1))

    model.train()
    f1_train = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time()
    adjust_learning_rate(optimizer, epoch)
    for i, (seqs, labels) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time() - end_time)

        seqs = get_variable(seqs.float())
        labels = get_variable(labels.long())

        outputs = model(seqs)
        # print(outputs[1:3, :])
        labels = labels.squeeze()
        loss = loss_function(outputs, labels)
        losses.update(loss.data, seqs.size(0))
        _, preds = torch.max(outputs.data, 1)

        acc = get_acc(outputs, labels)
        accuracies.update(acc, seqs.size(0))
        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


        total[0] += labels.size(0)
        labels_reshape = labels
        correct[0] += (preds == labels_reshape.data).sum().item()

        ifCorrect = np.array((preds == labels_reshape.data).cpu().numpy())
        failure_case_ind = np.where(ifCorrect == 0)
        label_for_failure_case = np.array(labels_reshape.cpu().numpy())
        label_for_pred_case = np.array(preds.cpu().numpy())
        failure_case_True_label = label_for_failure_case[failure_case_ind]
        failure_case_Pred_label = label_for_pred_case[failure_case_ind]
        # print('Failure_case_True  {0} % '.format(failure_case_True_label))
        # print('Failure_case_Pred  {0} % '.format(failure_case_Pred_label))

        f1_train.update(f1_score(labels_reshape.cpu().numpy(), preds.cpu().numpy(), average='micro'))
        f1_train_total.update(f1_score(labels_reshape.cpu().numpy(), preds.cpu().numpy(), average='micro'))
        batch_time.update(time() - end_time)
        end_time = time()

        train_batch_logger.log({
            'epoch': epoch,
            'batch': i + 1,
            'iter': (epoch - 1) * len(train_loader) + (i + 1),
            'loss': losses.val,
            'acc': accuracies.val,
            'lr': optimizer.param_groups[0]['lr'],
            'Failure_case_True': failure_case_True_label,
            'Failure_case_Pred': failure_case_Pred_label
        })


        if (i + 1) % 10 == 0:
            print(
                'Epoch [%d/%d], Train_Iter [%d/%d] Loss: %.6f, Acc: %.6f, Time: %.3f, F1-score: %.3f, F1-score.avg: %.3f, lr: %.7f '
                % (epoch + 1, EPOCH, i + 1, len(train_loader), loss.item(), accuracies.avg,
                   batch_time.val, f1_train.val, f1_train.avg, optimizer.param_groups[0]['lr']))

    train_logger.log({
        'epoch': epoch,
        'loss': losses.avg,
        'acc': accuracies.avg,
        'lr': optimizer.param_groups[0]['lr'],
        'f1_score.avg':f1_train.avg
    })

    if epoch % CHECK_POINTS == 0:
        checkpoint(epoch, model, optimizer)