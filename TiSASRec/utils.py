import sys
import copy
import torch
import random
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from multiprocessing import Process, Queue

# sampler for batch generation
def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def computeRePos(time_seq, time_span):
    
    size = time_seq.shape[0]
    time_matrix = np.zeros([size, size], dtype=np.int32)
    for i in range(size):
        for j in range(size):
            span = abs(time_seq[i]-time_seq[j])
            if span > time_span:
                time_matrix[i][j] = time_span
            else:
                time_matrix[i][j] = span
    return time_matrix


def Relation(user_train, usernum, maxlen, time_span):
    data_train = dict()
    for user in tqdm(range(1, usernum+1), desc='Preparing relation matrix'):
        if user not in user_train.keys() or len(user_train[user]) <= 1: continue
        time_seq = np.zeros([maxlen], dtype=np.int32)
        idx = maxlen - 1
        for i in reversed(user_train[user][:-1]):
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break
        data_train[user] = computeRePos(time_seq, time_span)
    return data_train


def sample_function(user_train, usernum, itemnum, batch_size, maxlen, relation_matrix, result_queue, SEED):
    def sample():

        user = np.random.randint(1, usernum + 1)
        while user not in user_train.keys() or len(user_train[user]) <= 1: user = np.random.randint(1, usernum + 1)

        seq = np.zeros([maxlen], dtype=np.int32)
        time_seq = np.zeros([maxlen], dtype=np.int32)
        pos = np.zeros([maxlen], dtype=np.int32)
        neg = np.zeros([maxlen], dtype=np.int32)
        nxt = user_train[user][-1][0]
        idx = maxlen - 1

        ts = set(map(lambda x: x[0],user_train[user]))
        for i in reversed(user_train[user][:-1]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            pos[idx] = nxt
            if nxt != 0: neg[idx] = random_neq(1, itemnum + 1, ts)
            nxt = i[0]
            idx -= 1
            if idx == -1: break
        time_matrix = relation_matrix[user]
        return (user, seq, time_seq, time_matrix, pos, neg)

    np.random.seed(SEED)
    while True:
        one_batch = []
        for i in range(batch_size):
            one_batch.append(sample())

        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, User, usernum, itemnum, relation_matrix, batch_size=64, maxlen=10, n_workers=1):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for i in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(User,
                                                      usernum,
                                                      itemnum,
                                                      batch_size,
                                                      maxlen,
                                                      relation_matrix,
                                                      self.result_queue,
                                                      np.random.randint(2e9)
                                                      )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def timeSlice(time_set):
    time_min = min(time_set)
    time_map = dict()
    for time in time_set: # float as map key?
        time_map[time] = int(round(float(time-time_min)))
    return time_map


def cleanAndsort(User, time_map):
    User_filted = dict()
    user_set = set()
    item_set = set()
    for user, items in User.items():
        user_set.add(user)
        User_filted[user] = items
        for item in items:
            item_set.add(item[0])
    user_map = dict()
    item_map = dict()
    for u, user in enumerate(user_set):
        user_map[user] = u
    for i, item in enumerate(item_set):
        item_map[item] = i
    
    for user, items in User_filted.items():
        User_filted[user] = sorted(items, key=lambda x: x[1])

    User_res = dict()
    for user, items in User_filted.items():
        User_res[user_map[user]] = list(map(lambda x: [item_map[x[0]], time_map[x[1]]], items))

    time_max = set()
    for user, items in User_res.items():
        time_list = list(map(lambda x: x[1], items))
        time_diff = set()
        for i in range(len(time_list)-1):
            if time_list[i+1]-time_list[i] != 0:
                time_diff.add(time_list[i+1]-time_list[i])
        if len(time_diff)==0:
            time_scale = 1
        else:
            time_scale = min(time_diff)
        time_min = min(time_list)
        User_res[user] = list(map(lambda x: [x[0], int(round((x[1]-time_min)/time_scale)+1)], items))
        time_max.add(max(set(map(lambda x: x[1], User_res[user]))))

    return User_res, len(user_set), len(item_set), max(time_max)


# train/val/test data generation
def data_partition(fname, args):
    usernum = 0
    itemnum = 0
    User = defaultdict(list)
    user_train = {}
    user_valid = {}
    user_test = {}
    user_list = list()
    item_list = list()
    neglist = defaultdict(list)
    user_neg = {}

    time_set = set()
    # assume user/item index starting from 1
    # f = open('data/cross_data/%s_all.csv' % fname, 'r')
    f = open(args.dataset_path+'%s_all.csv' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = float(t)
        user_list.append(u)
        item_list.append(i)
        time_set.add(t)

        User[u].append([i, t])

    time_map = timeSlice(time_set)
    User, usernum, itemnum, timenum = cleanAndsort(User, time_map)
    # usernum = len(set(user_list))
    # itemnum = len(set(item_list))

    # f = open('data/cross_data/%s_negative.csv' % fname, 'r')
    f = open(args.dataset_path+'%s_negative.csv' % fname, 'r')
    for line in f:
        l = line.rstrip().split(',')
        u = int(l[0])
        for j in range(1, 101):
            i = int(l[j])
            neglist[u].append(i)

    for user in User:
        nfeedback = len(User[user])
        if nfeedback < 3:
            user_train[user] = User[user]
            user_valid[user] = []
            user_test[user] = []
        else:
            user_train[user] = User[user][:-2]
            user_valid[user] = []
            user_valid[user].append(User[user][-2])
            user_test[user] = []
            user_test[user].append(User[user][-1])
        user_neg[user] = neglist[user]
    return [user_train, user_valid, user_test, user_neg, usernum, itemnum, timenum]


def test_load(dataset):
    [train, valid, test, neg, usernum, itemnum, timenum] = copy.deepcopy(dataset)
    test_user = []
    test_candidates = []
    for u in range(1, usernum+1):
        if u not in train.keys() or len(train[u]) < 1 or len(test[u]) < 1: continue
        rated = set(map(lambda x: x[0], train[u]))
        rated.add(0)
        rated.add(valid[u][0][0])
        item_idx = [test[u][0][0]]
        for t in neg[u]:
            item_idx.append(t)
        test_user.append(u)
        test_candidates.append(item_idx)
    return test_user, test_candidates

# TODO: merge evaluate functions for test and val set
# evaluate on test set
def evaluate(model, dataset, args, test_user, test_candidates):
    [train, valid, test, neg, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    HT_5 = 0.0
    NDCG_5 = 0.0
    HT_10 = 0.0
    NDCG_10 = 0.0
    HT_20 = 0.0
    NDCG_20 = 0.0

    test_num = 0.0

    for k in range(len(test_user)):
        u = test_user[k]

        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        seq[idx] = valid[u][0][0]
        time_seq[idx] = valid[u][0][1]
        idx -= 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break

        item_idx = test_candidates[k]

        time_matrix = computeRePos(time_seq, args.time_span)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], item_idx]])
        predictions = predictions[0] # - for 1st argsort DESC

        rank = predictions.argsort().argsort()[0].item()

        test_num += 1

        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1
        if test_num % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return HT_5 / test_num, NDCG_5 / test_num, HT_10 / test_num, NDCG_10 / test_num, HT_20 / test_num, NDCG_20 / test_num

def valid_load(dataset):
    [train, valid, test, neg, usernum, itemnum, timenum] = copy.deepcopy(dataset)
    valid_user = []
    valid_candidates = []
    for u in range(1, usernum+1):
        if u not in train.keys() or len(train[u]) < 1 or len(valid[u]) < 1: continue
        rated = set(map(lambda x: x[0], train[u]))
        rated.add(0)
        item_idx = [valid[u][0][0]]
        for t in neg[u]:
            item_idx.append(t)
        valid_user.append(u)
        valid_candidates.append(item_idx)

    return valid_user, valid_candidates
# evaluate on val set
def evaluate_valid(model, dataset, args, valid_user, valid_candidates):
    [train, valid, test, neg, usernum, itemnum, timenum] = copy.deepcopy(dataset)

    HT_5 = 0.0
    NDCG_5 = 0.0
    HT_10 = 0.0
    NDCG_10 = 0.0
    HT_20 = 0.0
    NDCG_20 = 0.0
    valid_num = 0.0

    for k in range(len(valid_user)):
        u = valid_user[k]


        seq = np.zeros([args.maxlen], dtype=np.int32)
        time_seq = np.zeros([args.maxlen], dtype=np.int32)
        idx = args.maxlen - 1
        for i in reversed(train[u]):
            seq[idx] = i[0]
            time_seq[idx] = i[1]
            idx -= 1
            if idx == -1: break

        item_idx = valid_candidates[k]

        time_matrix = computeRePos(time_seq, args.time_span)

        predictions = -model.predict(*[np.array(l) for l in [[u], [seq], [time_matrix], item_idx]])
        predictions = predictions[0]

        rank = predictions.argsort().argsort()[0].item()

        valid_num += 1

        if rank < 5:
            NDCG_5 += 1 / np.log2(rank + 2)
            HT_5 += 1
        if rank < 10:
            NDCG_10 += 1 / np.log2(rank + 2)
            HT_10 += 1
        if rank < 20:
            NDCG_20 += 1 / np.log2(rank + 2)
            HT_20 += 1
        if valid_num % 100 == 0:
            print('.', end="")
            sys.stdout.flush()

    return HT_5 / valid_num, NDCG_5 / valid_num, HT_10 / valid_num, NDCG_10 / valid_num, HT_20 / valid_num, NDCG_20 / valid_num