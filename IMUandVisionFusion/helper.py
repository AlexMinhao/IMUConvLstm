import torch
import numpy as np
from time import time


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


t = None
def timeit(name = ''):
    global t
    if t is None:
        print('timer start')
        t = time()
        return
    print(name,int((time()-t)*1000))
    t = time()


def get_f1_score(pred_choice, target):
    # TP    predict and label both eq to 1
    tp = ((pred_choice == 1) & (target.data == 1)).cpu().sum()
    # TN    predict and label both eq to 0
    tn = ((pred_choice == 0) & (target.data == 0)).cpu().sum()
    # FN    predict 0 label 1
    fn = ((pred_choice == 0) & (target.data == 1)).cpu().sum()
    # FP    predict 1 label 0
    fp = ((pred_choice == 1) & (target.data == 0)).cpu().sum()
    return tp, tn, fn, fp

# def data_augmentation(x,y):
#



# def mean_error(output,target):
#     batch_size = target.size(0)
#     diff = torch.abs(output - target).view(batch_size, JOINT_LEN, 3)
#     sqr_sum = torch.sum(torch.pow(diff, 2), 2)
#     sqrt_row = torch.sqrt(sqr_sum)
#     sqrt_row = sqrt_row.mean(dim=0)
#     # print sqrt_row
#     # print sqrt_row[10],sqrt_row[14],sqrt_row[17],sqrt_row[20],sqrt_row[6]
#
#     # print (sqrt_row)
#     if batch_size != 0:
#         return torch.mean(sqrt_row),sqrt_row
#     else:
#         return 0
