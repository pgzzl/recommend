import numpy as np
import torch
import os
import random
import pickle
import argparse
from collections import deque, Counter
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
from torch.utils.tensorboard import SummaryWriter

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device:'+device.type)

# appRec_dataset={'app_size':app_size,'lib_size':lib_size,
#          'app_mapping':app_mapping,'lib_mapping':lib_mapping,
#          'train_app_list': train_app_list, 'test_app_list': test_app_list,
#         'train_pair': train_pair}

def main(args):
    with open(args.data,'rb') as f:
        dataset=pickle.load(f)
        app_size,lib_size=dataset['app_size'],dataset['lib_size']
        train_app_list,test_app_list=dataset['train_app_list'],dataset['test_app_list']
        train_pair=dataset['train_pair']

    print('app_size:',app_size)
    print('lib_size:',lib_size)
    print('Load complete')

    #Counter得到的是一个字典，key为值，value为出现的次数
    counts = Counter([num for sublist in train_app_list for num in sublist])
    #将lib按出现的次数降序排序
    item_ranked_list = sorted(counts, key=lambda x: counts[x], reverse=True)
    max_count = counts[item_ranked_list[0]]
    print("出现次数最高的项目:", item_ranked_list[0])
    print("出现次数:", max_count)
    k_list = [5,10]
    precisions, recalls, hits,ndcgs = [], [], [], []

    for k in k_list:
        precision,recall,hit=0,0,0
        ndcg_scores=[]
        length=0

        for app in range(app_size):
            test=set(test_app_list[app])
            if (len(test)>0):
                length +=1
                recommendation=[]
                for lib in item_ranked_list:
                    if len(recommendation)==k:
                        break
                    if lib not in train_app_list[app]:
                        recommendation.append(lib)
                # global txt
                pred = set(recommendation)
                val = len(test & pred)
                precision += val / k
                recall += val / len(test)

                if val>0:
                    hit +=1
                
                dcg = 0
                for i, item in enumerate(recommendation):
                    if item in test:
                        relevance = 1  # Assuming binary relevance, where relevant items have a relevance of 1
                        rank = i + 1  # Rank of the recommended item
                        dcg += (2 ** relevance - 1) / np.log2(rank + 1)

                ideal_dcg = 0
                ideal_ranked_list = sorted(test, key=lambda x: -len(train_app_list[app]) if x in train_app_list[
                    app] else 0)
                for i, item in enumerate(ideal_ranked_list):
                    relevance = 1  # Assuming binary relevance, where relevant items have a relevance of 1
                    rank = i + 1  # Rank of the item in the ideal ranked list
                    ideal_dcg += (2 ** relevance - 1) / np.log2(rank + 1)

                if ideal_dcg > 0:
                    ndcg = dcg / ideal_dcg
                else:
                    ndcg = 0
                ndcg_scores.append(ndcg)
        precisions.append(precision / length)
        recalls.append(recall / length)
        hits.append(hit / length)
        ndcgs.append(np.mean(ndcg_scores))           
    print('P@5: %.4f, P@10: %.4f, R@5: %.4f, R@10: %.4f, H@5: %.4f, H@10: %.4f, NDCG@5: %.4f, NDCG@10: %.4f' % (
        precisions[0], precisions[1], recalls[0], recalls[1], hits[0], hits[1], ndcgs[0], ndcgs[1]))


if __name__ == '__main__':
       # Parse argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--data',
                        type=str,
                        default=os.path.join('preprocessed', 'appRec.pickle'),
                        help="File path for data")
    # Seed
    parser.add_argument('--seed',
                        type=int,
                        default=0,
                        help="Seed (For reproducability)")
    # Model
    parser.add_argument('--dim',
                        type=int,
                        default=128,
                        help="Dimension for embedding")
    # Optimizer
    parser.add_argument('--lr',
                        type=float,
                        default=0.0002,
                        help="Learning rate")
    parser.add_argument('--weight_decay',
                        type=float,
                        default=0.025,
                        help="Weight decay factor")
    # Training
    parser.add_argument('--n_epochs',
                        type=int,
                        default=800,
                        help="Number of epoch during training")
    parser.add_argument('--batch_size',
                        type=int,
                        default=1,
                        help="Batch size in one iteration")
    parser.add_argument('--print_every',
                        type=int,
                        default=1000,
                        help="Period for printing smoothing loss during training")
    parser.add_argument('--eval_every',
                        type=int,
                        default=1000,
                        help="Period for evaluating precision and recall during training")
    parser.add_argument('--save_every',
                        type=int,
                        default=20000,
                        help="Period for saving model during training")
    parser.add_argument('--model',
                        type=str,
                        default=os.path.join('output', 'bpr.pt'),
                        help="File path for model")
    args = parser.parse_args()
    main(args)

 
 