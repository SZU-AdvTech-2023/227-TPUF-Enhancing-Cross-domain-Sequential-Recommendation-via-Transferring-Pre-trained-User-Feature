from __future__ import absolute_import, division, print_function

import pandas as pd
import numpy as np
from sklearn import preprocessing
import argparse
from collections import defaultdict
from multiprocessing import Process, Queue

parser = argparse.ArgumentParser()
parser.add_argument('-s', default='Movies_and_TV', help='category')
parser.add_argument('-t', default='Books', help='category')

args = parser.parse_args()
processed_file_prefix_s = "cross_data/" + args.s + "_"
processed_file_prefix_t = "cross_data/" + args.t + "_"
# ================================================================================
# obtain implicit feedbacks
def data_partition(fname, fname2):
    usernum = 0
    itemnum1 = 0
    User = defaultdict(list)
    User1 = list()
    User2 = list()
    user_neg1 = list()

    itemnum2 = 0
    user_neg2 = list()

    user_map = dict()
    item_map1 = dict()
    item_map2 = dict()

    user_ids = list()
    item_ids1 = list()
    item_ids2 = list()

    Time = defaultdict(list)

    f = open('processed_data_all/%s_train.csv' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = int(t)

        user_ids.append(u)
        item_ids1.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('processed_data_all/%s_valid.csv' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = int(t)

        user_ids.append(u)
        item_ids1.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('processed_data_all/%s_test.csv' % fname, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = int(t)

        user_ids.append(u)
        item_ids1.append(i)
        User[u].append(i)
        Time[u].append(t)

    for u in user_ids:
        if u not in user_map:
            user_map[u] = usernum + 1
            usernum += 1
    for i in item_ids1:
        if i not in item_map1:
            item_map1[i] = itemnum1 + 1
            itemnum1 += 1

    for user in User:
        u = user_map[user]
        tt = 0
        for item in User[user]:
            i = item_map1[item]
            t = Time[user][tt]
            User1.append([u,i,t])

            tt+=1

    User = defaultdict(list)
    Time = defaultdict(list)

    f = open('processed_data_all/%s_train.csv' % fname2, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = int(t)

        user_ids.append(u)
        item_ids2.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('processed_data_all/%s_valid.csv' % fname2, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = int(t)

        user_ids.append(u)
        item_ids2.append(i)
        User[u].append(i)
        Time[u].append(t)

    f = open('processed_data_all/%s_test.csv' % fname2, 'r')
    for line in f:
        u, i, t = line.rstrip().split(',')
        u = int(u)
        i = int(i)
        t = int(t)

        user_ids.append(u)
        item_ids2.append(i)
        User[u].append(i)
        Time[u].append(t)

    for i in item_ids2:
        if i not in item_map2:
            item_map2[i] = itemnum2 + 1
            itemnum2 += 1

    for user in User:
        u = user_map[user]
        tt = 0
        for item in User[user]:
            i = item_map2[item]
            t = Time[user][tt]
            User2.append([u,i,t])

            tt+=1

    f = open('processed_data_all/%s_negative.csv' % fname, 'r')
    for line in f:
        l = line.rstrip().split(',')
        u = user_map[int(l[0])]
        neglist = [u]
        for j in range(1, 101):
            i = item_map1[int(l[j])]
            neglist.append(i)
        user_neg1.append(neglist.copy())

    f = open('processed_data_all/%s_negative.csv' % fname2, 'r')
    for line in f:
        l = line.rstrip().split(',')
        u = user_map[int(l[0])]
        neglist = [u]
        for j in range(1, 101):
            i = item_map2[int(l[j])]
            neglist.append(i)
        user_neg2.append(neglist.copy())

    pd.DataFrame(User1).to_csv(processed_file_prefix_s+"all.csv", header=False, index=False)
    pd.DataFrame(User2).to_csv(processed_file_prefix_t+"all.csv", header=False, index=False)
    pd.DataFrame(user_neg1).to_csv(processed_file_prefix_s+"negative.csv", header=False, index=False)
    pd.DataFrame(user_neg2).to_csv(processed_file_prefix_t+"negative.csv", header=False, index=False)

data_partition(args.s, args.t)