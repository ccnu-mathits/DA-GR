'''
Created on Aug 8, 2016
Processing datasets.

@author: Xiangnan He (xiangnanhe@gmail.com)

Modified  on Nov 10, 2017, by Lianhai Miao
'''

import scipy.sparse as sp
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from collections import defaultdict
import math


class Data_loader:
    def __init__(self,dataset,batch_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.datasize = dataset.shape[0]

    def get_batch(self):
        index = torch.randint(self.datasize,[self.batch_size])
        return self.dataset[index]


class GDataset(object):

    def __init__(self, dataname, user_path, group_path, user_in_group_path, num_negatives, test_pos_neg_num):
        '''
        Constructor
        '''
        self.dataname = dataname
        self.user_in_group_path = user_in_group_path
        self.num_negatives = num_negatives
        self.test_pos_neg_num = test_pos_neg_num
        # user data
        self.user_trainMatrix, self.user_explicit_trainMatrix = self.load_rating_file_as_matrix(user_path + "Train.txt")
        self.num_users, self.num_items = self.user_trainMatrix.shape

        self.usertest_input, self.item_test_input = self.get_test_instances(user_path + "Negative.txt")
        
        # self.u_items_dict, self.i_users_dict = self.get_ui_dict(user_path + "Train.txt")
        # group data
        self.group_trainMatrix, self.gro_explicit_trainMatrix = self.load_rating_file_as_matrix(group_path + "Train.txt")
        
        self.grouptest_input, self.groupitem_test = self.get_test_instances(group_path + "Negative.txt")
        # self.gro_items_dict, self.i_groups_dict = self.get_ui_dict(group_path + "Train.txt")
        self.gro_members_dict, self.user_groups_dict = self.get_group_member_dict(user_in_group_path)
        self.num_groups = len(self.gro_members_dict)

        self.gro_user = self.get_group_member_input(user_in_group_path)
        self.userTrain = self.get_ug_train_instances(self.user_trainMatrix)
        self.groTrain = self.get_ug_train_instances(self.group_trainMatrix)
        


    def get_group_member_input(self, user_in_group_path):
        gro_input = []
        member_input = []
        with open(user_in_group_path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                a = line.split(' ')
                g = int(a[0])
                # g_m_d[g] = []
                for m in a[1].split(','):
                    gro_input.append(g)
                    member_input.append(int(m))
                line = f.readline().strip()
        return gro_input, member_input

    

    def get_ug_train_instances(self, train):
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                neg_item_input.append(j)
        u_pi_ni = [[u, pi, ni] for u, pi, ni in zip(user_input, pos_item_input, neg_item_input)]
        return np.array(u_pi_ni)

    def get_group_member_dict(self, user_in_group_path):
        g_m_d = defaultdict(list)
        u_g_d = defaultdict(list)
        with open(user_in_group_path, 'r') as f:
            line = f.readline().strip()
            while line != None and line != "":
                if self.dataname == 'MaFengWo' or self.dataname == 'CAMRa2011':
                    a = line.split(' ')
                    g = int(a[0])
                    # g_m_d[g] = []
                    for m in a[1].split(','):
                        g_m_d[g].append(int(m))
                        u_g_d[int(m)].append(g)
                elif self.dataname == 'ml-latest-small':
                    a = line.split(' ')
                    g = int(a[0])
                    # g_m_d[g] = []
                    for m in a[1:]:
                        g_m_d[g].append(int(m))
                        u_g_d[int(m)].append(g)
                line = f.readline().strip()
        return g_m_d, u_g_d

    # def get_user_groups_dict(self, user_in_group_path):
    #     u_g_d = defaultdict(list)
    #     with open(user_in_group_path, 'r') as f:
    #         line = f.readline().strip()
    #         while line != None and line != "":
    #             a = line.split(' ')
    #             g = int(a[0])
    #             for m in a[1].split(','):
    #                 u_g_d[int(m)].append(g)
    #             line = f.readline().strip()
    #     return u_g_d
    
    
    def load_rating_file_as_list(self, filename):
        ratingList = []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                user, item = int(arr[0]), int(arr[1])
                ratingList.append([user, item])
                line = f.readline()
        return ratingList

    def get_ui_dict(self, filename):
        # user_i -> items: 1: [10, 23]
        with open(filename, 'r') as reader:
            u_i_dict = defaultdict(list)
            i_u_dict = defaultdict(list)
            for line in reader:
                # 162540,32,4,...
                user_id, item_id = map(int, line.split(' ')[:2]) # ',' or '\t'
                u_i_dict[user_id].append(item_id)
                i_u_dict[item_id].append(user_id)

        return u_i_dict, i_u_dict

    def load_negative_file(self, filename):
        testRatingList, negativeList = [], []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                if line == "\n":
                    line = f.readline()
                    continue
                arr = line.split(" ")
                user, item = eval(arr[0])[0], eval(arr[0])[1]
                testRatingList.append([user, item])
                negatives = []
                for x in arr[1:]:
                    negatives.append(int(x))
                negativeList.append(negatives)
                line = f.readline()
        return testRatingList, negativeList

    def load_rating_file_as_matrix(self, filename):
        # Get number of users and items
        num_users, num_items = 0, 0
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                u, i = int(arr[0]), int(arr[1])
                num_users = max(num_users, u) 
                num_items = max(num_items, i) 
                line = f.readline()
        # Construct matrix 
        mat = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        mat_explicit = sp.dok_matrix((num_users + 1, num_items + 1), dtype=np.float32)
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                arr = line.split(" ")
                if len(arr) > 2:
                    user, item, rating = int(arr[0]), int(arr[1]), eval(arr[2])
                    if (rating > 0):
                        mat[user, item] = 1.0
                        mat_explicit[user, item] = rating
                else:
                    user, item = int(arr[0]), int(arr[1])
                    mat[user, item] = 1.0
                line = f.readline()
        return mat, mat_explicit

    def get_explicit_train_instances(self, train):
        user_input, item_input, rating = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        max_rating = max(train.values())
        for (u, i) in train.keys():
            # positive instance
            user_input.append(u)
            item_input.append(i)
            rating.append(train[u, i]/max_rating)
            for _ in range(self.num_negatives):
                # negative instances
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                item_input.append(j)
                rating.append(0)
        # pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, item_input, rating

    def get_train_instances(self, train):
        user_input, pos_item_input, neg_item_input = [], [], []
        num_users = train.shape[0]
        num_items = train.shape[1]
        for (u, i) in train.keys():
            # positive instance
            for _ in range(self.num_negatives):
                pos_item_input.append(i)
            # negative instances
            for _ in range(self.num_negatives):
                j = np.random.randint(num_items)
                while (u, j) in train:
                    j = np.random.randint(num_items)
                user_input.append(u)
                neg_item_input.append(j)
        pi_ni = [[pi, ni] for pi, ni in zip(pos_item_input, neg_item_input)]
        return user_input, pi_ni
    
    def get_test_instances(self, filename):
        user_input, item_input = [], []
        with open(filename, "r") as f:
            line = f.readline()
            while line != None and line != "":
                if line == "\n":
                    line = f.readline()
                    continue
                if self.dataname == 'MaFengWo' or self.dataname == 'CAMRa2011':
                    arr = line.split(" ")
                    user, pos_item = eval(arr[0])[0], eval(arr[0])[1]
                    user_input.append(user)
                    item_input.append(pos_item)
                    for x in arr[1:]:
                        user_input.append(user)
                        item_input.append(int(x))
                elif self.dataname == 'ml-latest-small':
                    arr = line.split(' ')
                    user = int(arr[0])
                    pos_item = int(arr[1])
                    user_input.append(user)
                    item_input.append(pos_item)
                    for x in arr[2:]:
                        user_input.append(user)
                        item_input.append(int(x))
                line = f.readline()
        return user_input, item_input

    def get_user_dataloader(self, batch_size):
        user, positem_negitem_at_u = self.get_train_instances(self.user_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(user), torch.LongTensor(positem_negitem_at_u))
        user_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        return user_train_loader

    def get_group_dataloader(self, batch_size):
        # group and positem_negitem_at_g are two lists
        group, positem_negitem_at_g = self.get_train_instances(self.group_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(group), torch.LongTensor(positem_negitem_at_g))
        group_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        
        return group_train_loader

    def get_ug_dataloader(self, batch_size):
        gro_input, member_input = self.get_group_member_input(self.user_in_group_path)
        train_data = TensorDataset(torch.LongTensor(gro_input), torch.LongTensor(member_input))
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        
        return train_loader

    def get_user_explicit_dataloader(self, batch_size):
        user_input, item_input, rating = self.get_explicit_train_instances(self.user_explicit_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(user_input), torch.LongTensor(item_input), torch.Tensor(rating))
        user_explicit_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        
        return user_explicit_train_loader

    def get_gro_explicit_dataloader(self, batch_size):
        gro_input, item_input, rating = self.get_explicit_train_instances(self.gro_explicit_trainMatrix)
        train_data = TensorDataset(torch.LongTensor(gro_input), torch.LongTensor(item_input), torch.Tensor(rating))
        gro_explicit_train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)
        
        return gro_explicit_train_loader


    def get_user_test_dataloader(self):
        test_tensor_data = TensorDataset(torch.LongTensor(self.usertest_input), torch.LongTensor(self.item_test_input))
        test_loader = DataLoader(test_tensor_data,batch_size=self.test_pos_neg_num + 1, shuffle=False)
        return test_loader
    
    def get_gro_test_dataloader(self):
        test_tensor_data = TensorDataset(torch.LongTensor(self.grouptest_input), torch.LongTensor(self.groupitem_test))
        test_loader = DataLoader(test_tensor_data, batch_size=self.test_pos_neg_num + 1, shuffle=False)
        return test_loader






