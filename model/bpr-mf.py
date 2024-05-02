import os
import pickle
import numpy as np
from collections import deque
import random
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import IterableDataset,DataLoader,get_worker_info
from torch.utils.tensorboard import SummaryWriter

device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def sqdist(p1, p2):
    return (p1 - p2).pow(2).sum(dim=-1)

#生成三元组
class TripletUniformPair(IterableDataset):
    def __init__(self, num_item, user_list, pair, shuffle, num_epochs):
        self.num_item = num_item
        self.user_list = user_list
        self.pair = pair
        self.shuffle = shuffle
        self.num_epochs = num_epochs

    def __iter__(self):
        worker_info = get_worker_info()
        # Shuffle per epoch
        self.example_size = self.num_epochs * len(self.pair)
        self.example_index_queue = deque([])
        self.seed = 0
        if worker_info is not None:
            self.start_list_index = worker_info.id
            self.num_workers = worker_info.num_workers
            self.index = worker_info.id
        else:
            self.start_list_index = None
            self.num_workers = 1
            self.index = 0
        return self

    def __next__(self):
        if self.index >= self.example_size:
            raise StopIteration
        # If `example_index_queue` is used up, replenish this list.
        while len(self.example_index_queue) == 0:
            index_list = list(range(len(self.pair)))
            if self.shuffle:
                random.Random(self.seed).shuffle(index_list)
                self.seed += 1
            if self.start_list_index is not None:
                index_list = index_list[self.start_list_index::self.num_workers]
                # Calculate next start index
                self.start_list_index = (self.start_list_index + (
                        self.num_workers - (len(self.pair) % self.num_workers))) % self.num_workers
            self.example_index_queue.extend(index_list)
        result = self._example(self.example_index_queue.popleft())
        self.index += self.num_workers
        return result

    def _example(self, idx):
        u = self.pair[idx][0]
        i = self.pair[idx][1]
        j = np.random.randint(self.num_item)
        while j in self.user_list[u]:
            j = np.random.randint(self.num_item)
        return u, i, j

class BPR(nn.Module):
    def __init__(self, user_size,item_size,dim,weight_decay ) :
        super().__init__()
        self.W=nn.Parameter(torch.empty(user_size,dim))
        self.H=nn.Parameter(torch.empty(item_size,dim))
        nn.init.xavier_normal(self.W.data)
        nn.init.xavier_normal(self.H.data)
        self.weight_decay=weight_decay

    def forward(self,u,i,j):
        u=self.W[u,:]
        i=self.H[i,:]
        j=self.H[j,:]
        
        #计算内积偏好得分,u形状为(batch_size,dim)，i为positive item，j为negative item
        #按对应元素相乘后得到的形状为(batch_size,dim),沿着第二个维度求和，最后形状为(batchsize,1)
        x_ui=torch.mul(u,i).sum(dim=1)
        x_uj=torch.mul(u,j).sum(dim=1)

        x_uij=x_ui-x_uj
        log_prob=F.logsigmoid(x_uij).sum()
        regularization=self.weight_decay*(
            u.norm(dim=1).pow(2).sum()+i.norm(dim=1).pow(2).sum()+j.norm(dim=1).pow(2).sum()
        )

        #返回bpr_loss,和user item向量表示
        return -log_prob+regularization,self.W,self.H
    
    def recommend(self,u):
        u=self.W[u,:]
        #mm为矩阵乘法,u为(batchsize,dim),item转置为(dim,item_size)
        x_ui=torch.mm(u,self.H.t())
        pred=torch.argsort(x_ui,dim=1)
        return pred
    
    def precision_and_recall_k(self,user_emd,item_emd,train_user_list,test_user_list,klist,batch=512):
        #user_embedding 形状为(user_size,dim)，不参与计算反向转播更新参数
        #取前k个
        max_k=max(klist)

        result=None

        for i in range(0,user_emd.shape[0],batch):
            #(要处理的user，item_size),newones除了张量形状，类型和device与调用的tensor相同
            mask=user_emd.new_ones([min([batch,user_emd[0]-i]),item_emd.shape[0]])
            for j in range(batch):
                if i+j>=user_emd.shape[0]:
                    break
                #scatter_() 方法用于按照给定的索引，在指定的维度上将指定的值赋值给张量。
                #将已经训练过的用户的项目置零，以便在预测时不会推荐已经训练过的项目。
                mask[j].scatter_(dim=0,index=torch.tensor(list(train_user_list[i+j])).to(device),
                                 value=torch.tensor(0,0).to(device))
                cur_result=torch.mm(user_emd[i:i+min(batch,user_emd.shape[0]-i),:],item_emd.t())

                cur_result=torch.sigmoid(cur_result)
                assert not torch.any(torch.isnan(cur_result))

                # Make zero for already observed item
                #将已经训练过的用户的项目置零，以便在预测时不会推荐已经训练过的项目。
                cur_result = torch.mul(mask, cur_result)

                _,cur_result=torch.topk(cur_result,k=max_k,dim=1)
                result = cur_result if result is None else torch.cat((result, cur_result), dim=0)

        result=result.cpu()

        # Sort indice and get test_pred_topk
        precisions, recalls, hits = [], [], []
        for k in klist:
            precision, recall, hit = 0, 0, 0
            length = 0
            for i in range(user_emd.shape[0]):
                test = set(test_user_list[i])
                if (len(test) > 0):
                    length += 1
                    pred = set(result[i, :k].numpy().tolist())
                    val = len(test & pred)
                    # precision += val / max([min([k, len(test)]), 1])
                    precision += val / k
                    # recall += val / max([len(test), 1])
                    recall += val / len(test)
                    if val > 0:
                        hit += 1
            precisions.append(precision / length)
            recalls.append(recall / length)
            hits.append(hit / length)
        return precisions, recalls, hits


def main(args):
    # Initialize seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load preprocess data
    with open(args.data, 'rb') as f:
        dataset = pickle.load(f)
        user_size, item_size = dataset['app_size'], dataset['lib_size']
        train_user_list, test_user_list = dataset['train_app_list'], dataset['test_app_list']
        train_pair = dataset['train_pair']
    print('user size:', user_size)
    print('item_size:', item_size)
    print('Load complete')

    # Create dataset, model, optimizer
    dataset = TripletUniformPair(item_size, train_user_list, train_pair, True, args.n_epochs)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=16)
    model = BPR(user_size, item_size, args.dim, args.weight_decay).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter()

    # Training
    smooth_loss = 0
    idx = 0
    for u, i, j in loader:
        optimizer.zero_grad()
        loss, user_embedding, item_embedding = model(u, i, j)
        loss.backward()
        optimizer.step()
        writer.add_scalar('train/loss', loss, idx)
        smooth_loss = smooth_loss * 0.99 + loss * 0.01
        if idx % args.print_every == (args.print_every - 1):
            print('loss: %.4f' % smooth_loss)
        if idx % args.eval_every == (args.eval_every - 1):
            plist, rlist, hitlist = model.precision_and_recall_k(model.W.detach(),
                                                        model.H.detach(),
                                                        train_user_list,
                                                        test_user_list,
                                                        klist=[1, 5, 10])
            print('P@5: %.4f P@10: %.4f,  R@5: %.4f, R@10: %.4f, H@5: %.4f, H@10: %.4f' % (
                 plist[1], plist[2], rlist[1], rlist[2], hitlist[1], hitlist[2]))
            writer.add_scalars('eval', {'P@1': plist[0],
                                        'P@5': plist[1],
                                        'P@10': plist[2]}, idx)
            writer.add_scalars('eval', {'R@1': rlist[0],
                                        'R@5': rlist[1],
                                        'R@10': rlist[2]}, idx)
        if idx % args.save_every == (args.save_every - 1):
            dirname = os.path.dirname(os.path.abspath(args.model))
            os.makedirs(dirname, exist_ok=True)
            user_path = os.path.join('output', 'user.pth')
            item_path = os.path.join('output', 'item.pth')
            torch.save(user_embedding, user_path)
            torch.save(item_embedding, item_path)
        # if idx ==
        idx += 1
    a = torch.load(os.path.join('output', 'user.pth'))
    b = torch.load(os.path.join('output', 'item.pth'))



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
                        default=2000,
                        help="Period for printing smoothing loss during training")
    parser.add_argument('--eval_every',
                        type=int,
                        default=2000,
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