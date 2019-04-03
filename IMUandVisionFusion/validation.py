import torch
from torch.autograd import Variable
import time
import sys
from helper import *
from utils import *
from definitions import *
from sklearn.metrics import f1_score

def val_epoch(epoch, valition_loader, model, loss_function, logger, total, correct,f1_val_total):
    print('validation at epoch {}'.format(epoch+1))

    model.eval()
    f1_val = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    accuracies = AverageMeter()

    end_time = time()
    for i, (seqs, labels) in enumerate(valition_loader):
        # measure data loading time
        data_time.update(time() - end_time)

        seqs = get_variable(seqs.float())
        labels = get_variable(labels.long())

        outputs = model(seqs)

        labels = labels.squeeze()
        loss = loss_function(outputs, labels)
        losses.update(loss.data, seqs.size(0))
        _, preds = torch.max(outputs.data, 1)

        acc = get_acc(outputs, labels)
        accuracies.update(acc, seqs.size(0))

        total[0] += labels.size(0)
        labels_reshape = labels
        correct[0] += (preds == labels_reshape.data).sum()

        ifCorrect = np.array((preds == labels_reshape.data).cpu().numpy())
        failure_case_ind = np.where(ifCorrect == 0)
        label_for_failure_case = np.array(labels_reshape.cpu().numpy())
        label_for_pred_case = np.array(preds.cpu().numpy())
        failure_case_True_label = label_for_failure_case[failure_case_ind]
        failure_case_Pred_label = label_for_pred_case[failure_case_ind]
        print('Failure_case_True  {0} % '.format(failure_case_True_label))
        print('Failure_case_Pred  {0} % '.format(failure_case_Pred_label))
        f1_val.update(f1_score(labels_reshape.cpu().numpy(), preds.cpu().numpy(), average='micro'))
        f1_val_total.update(f1_score(labels_reshape.cpu().numpy(), preds.cpu().numpy(), average='micro'))

        batch_time.update(time() - end_time)
        end_time = time()

        if (i + 1) % 1 == 0:
            print(
                'Epoch [%d/%d], Validation_Iter [%d/%d] Loss: %.6f, Acc: %.6f, Time: %.3f, F1-score: %.3f, F1-score.avg: %.3f '
                % (epoch + 1, EPOCH, i + 1, len(valition_loader), loss.item(), accuracies.avg,
                   batch_time.val, f1_val.val, f1_val.avg))

    logger.log({'epoch': epoch, 'loss': losses.avg, 'acc': accuracies.avg, 'f1_score.avg': f1_val.avg})

    return losses.avg
