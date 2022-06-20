import faulthandler; faulthandler.enable()
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from time import time
import os
from model.Group_aggregator import Group_aggregator
from utils.util import Helper, AGREELoss
from dataset import GDataset
import argparse
from model.useremb import Shared_encoder_pred_domain
from utils.util import DiffLoss, ReverseLayerF

torch.autograd.set_detect_anomaly(True)

dataname = 'MaFengWo'
# dataname = 'CAMRa2011'
# dataname = 'ml-latest-small'
parser = argparse.ArgumentParser()
parser.add_argument('--train_scheme', type=str, default='all')
parser.add_argument('--rec_scheme', type=str, default='M1-b')
parser.add_argument('--path', type=str, default='/home/admin123/ruxia/HAN-CDGRccnu/Experiments/DA-GRnew/data/' + dataname)
parser.add_argument('--user_dataset', type=str, default= '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/DA-GRnew/data/' + dataname + '/userRating')
parser.add_argument('--group_dataset', type=str, default= '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/DA-GRnew/data/' + dataname + '/groupRating')
parser.add_argument('--user_in_group_path', type=str, default= '/home/admin123/ruxia/HAN-CDGRccnu/Experiments/DA-GRnew/data/' + dataname + '/groupMember.txt')
parser.add_argument('--embedding_size_t_list', type=list, default=[32])
parser.add_argument('--n_epoch', type=int, default=260)
parser.add_argument('--early_stop', type=int, default=100)
parser.add_argument('--num_negatives', type=list, default=4)
parser.add_argument('--test_num_ng', type=int, default=100)
parser.add_argument('--batch_size_list', type=list, default=[128])
parser.add_argument('--lr', type=float, default=0.002) # ml dataset
parser.add_argument('--p_list', type=list, default=[0.4])
# parser.add_argument('--lr', type=float, default=0.000005) # cam dataset
parser.add_argument('--cuda', type=int, default=1)
parser.add_argument('--drop_ratio_list', type=list, default=[0.2])
parser.add_argument('--topK_list', type=list, default=[5, 10])
parser.add_argument('--type_m_gro', type=str, default='group')
parser.add_argument('--type_m_user', type=str, default='user')
parser.add_argument('--user_weight_list', type=float, default=[0.5])    # [0, 1.5] user prediction loss
parser.add_argument('--domain_weight_list', type=float, default=[0.002]) #[0.002, 0.2] gamma user domain loss , 0.08, 0.1, 0.15
parser.add_argument('--orthogonality_weight_list', type=float, default=[0.04, 0.06, 0.08, 0.1, 0.14, 0.18, 0.2]) # [0.02, 2] beta weight user private and share encoder difference loss , 0.2, 1, 1.5, 2
args = parser.parse_args()

DEVICE = torch.device('cuda:{}'.format(args.cuda)) if torch.cuda.is_available() else torch.device('cpu')
# DEVICE = torch.device('cpu')


class DAGR(nn.Module):
    def __init__(self, enc_u, env_i, g2e, enc_gro, enc_us2e, embedding_dim, drop_ratio, rec_scheme, device):
        super(DAGR, self).__init__()
        self.enc_u = enc_u
        self.env_i = env_i
        self.g2e = g2e
        self.enc_gro = enc_gro
        self.enc_us2e = enc_us2e
        self.device = device
        self.embedding_dim_t = embedding_dim
        self.drop_ratio = drop_ratio
        self.rec_scheme = rec_scheme
        self.shared_encoder_pred_domain = Shared_encoder_pred_domain(embedding_dim, 1)
        
        self.loss_similarity = torch.nn.BCELoss()
        self.loss_diff = DiffLoss()
        
        # initial model
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight)
            if isinstance(m, nn.Embedding):
                nn.init.xavier_normal_(m.weight)

    def forward_gro(self, gro_inputs, g_item_inputs):
        item_embeds_full = self.env_i(g_item_inputs)   # [B, C]
        g_embeds_full = self.enc_gro(gro_inputs, g_item_inputs, self.rec_scheme) # [B, C]

        # g_embeds_full = self.g2e(gro_inputs) # [B, C]

        element_embeds = torch.mul(g_embeds_full, item_embeds_full)
        preds = torch.sigmoid(element_embeds.sum(1)) # agree loss
        # preds = element_embeds.sum(1) # bpr loss
        return preds

    def forward_user(self, user_inputs, u_item_inputs):    
        item_embeds_full = self.env_i(u_item_inputs)   # [B, C]
        
        u_embeds_private = self.enc_u(user_inputs) # [B, C]
        u_embeds_share = self.enc_us2e(user_inputs)

    
        u_embeds_full = (u_embeds_private + u_embeds_share)/2

        # predictions 
        element_embeds = torch.mul(u_embeds_full, item_embeds_full)
        preds = torch.sigmoid(element_embeds.sum(1)) # agree loss
        # preds = element_embeds.sum(1) # bpr loss

        # member_diff = self.loss_diff(u_embeds_share, u_embeds_private)

        return preds

    def forward_ug(self, group, member, p):
        # user_share_embeds = self.enc_us2e[group, member, :]
        user_share_embeds = self.enc_us2e(member)
        user_private_embeds = self.enc_u(member)
        gro_embed = self.g2e(group)

        user_dann = self.get_adversarial_result(user_share_embeds, p, source=True)
        gro_dann = self.get_adversarial_result(gro_embed, p, source=False)
        domain_loss = (user_dann + gro_dann)

        member_diff = self.loss_diff(user_share_embeds, user_private_embeds)

        return user_share_embeds,user_private_embeds,gro_embed, domain_loss, member_diff

    def get_adversarial_result(self, x, p, source=True):
    
        loss_fn = nn.BCELoss()
        
        if source:
            domain_label = torch.ones(len(x)).long().to(self.device)
        else:
            domain_label = torch.zeros(len(x)).long().to(self.device)
            
        # get the reversed feature
        x = ReverseLayerF.apply(x, p)

        domain_pred = self.shared_encoder_pred_domain(x)
        
        loss_adv = loss_fn(domain_pred, domain_label.float().unsqueeze(dim=1))
        
        return loss_adv



# train the model
def training(model, train_loader, epoch_id, type_m, lr, user_weight, orthogonality_weight):
    t = time()
    # optimizer = optim.RMSprop(model.parameters(), lr)
    optimizer = optim.Adam(model.parameters(), lr)
    loss_function = AGREELoss()
    # loss_function = BPRLoss()

    losses = []
    for batch_id, (u, pi_ni) in enumerate(train_loader): 
        # Data Load
        user_input = u
        pos_item_input = pi_ni[:, 0]
        neg_item_input = pi_ni[:, 1]
        user_input, pos_item_input, neg_item_input = user_input.to(DEVICE), pos_item_input.to(DEVICE), neg_item_input.to(DEVICE)

        # Forward
        if type_m == 'group':
            pos_preds = model.forward_gro(user_input, pos_item_input)
            neg_preds = model.forward_gro(user_input, neg_item_input)
            loss = loss_function(pos_preds, neg_preds)

        elif type_m == "user":
            pos_preds = model.forward_user(user_input, pos_item_input)
            neg_preds = model.forward_user(user_input, neg_item_input)
            loss = user_weight * loss_function(pos_preds, neg_preds)
            # memb_diff = pos_memb_diff + neg_memb_diff
            # loss = loss + 0.0 * orthogonality_weight * memb_diff

        # Zero_grad
        model.zero_grad()
        # loss = loss_function(pos_preds, neg_preds)
        # record loss history
        losses.append(loss)  
        # Backward
        loss.backward(torch.ones_like(loss))
        # Update parameters
        optimizer.step()

    print('Iteration %d, training time is: [%.3f s], loss is [%.4f ]' % (epoch_id, time() - t, torch.mean(torch.tensor(losses))))


def training_ug(model, train_loader, epoch_id, p, lr, domain_weight, orthogonality_weight):
    ug_train = time()
    # optimizer = optim.RMSprop(model.parameters(), lr)
    optimizer = optim.Adam(model.parameters(), lr)

    domain_losses = []
    user_share_embeds_epoch, user_private_embeds_epoch, gro_embed_epoch = [], [], []
    for _, (group_input, member_input) in enumerate(train_loader): 
        # Data Load
        group_input, member_input = group_input.to(DEVICE), member_input.to(DEVICE)
        
        # Forward
        user_share_embeds, user_private_embeds, gro_embed, domain_loss, memb_diff2 = model.forward_ug(group_input, member_input, p)
        user_share_embeds_epoch.append(user_share_embeds)
        user_private_embeds_epoch.append(user_private_embeds)
        gro_embed_epoch.append(gro_embed)


        # Zero_grad
        model.zero_grad()
        
        loss = - domain_weight * domain_loss + orthogonality_weight * memb_diff2
        # loss = - domain_weight * domain_loss

        # record loss history
        domain_losses.append(domain_weight * domain_loss)
 
        # Backward
        loss.backward(torch.ones_like(loss))
        # Update parameters
        optimizer.step()
        # torch.mean(torch.tensor(domain_losses)),
    print('Iteration %d, training time is: [%.3f s], domain_loss is [%.4f ]' % (epoch_id, time()- ug_train, \
       torch.mean(torch.tensor(domain_losses))))

    return torch.stack(user_share_embeds_epoch,0).view(-1, user_share_embeds.shape[-1]),\
            torch.stack(user_private_embeds_epoch,0).view(-1, user_private_embeds.shape[-1]),\
            torch.stack(gro_embed_epoch,0).view(-1, gro_embed.shape[-1])


def evaluation(model, helper, test_data_loader, K, type_m, DEVICE):
    model = model.to(DEVICE)
    # set the module in evaluation mode
    model.eval()
    HR, NDCG = helper.metrics(model, test_data_loader, K, type_m, DEVICE)
    return HR, NDCG


if __name__ == '__main__':
    torch.random.manual_seed(1314)
    # initial helper
    helper = Helper()
    # initial dataSet class
    dataset = GDataset(dataname, args.user_dataset, args.group_dataset, args.user_in_group_path, args.num_negatives, args.test_num_ng)
    # get group number
    g_m_d = dataset.gro_members_dict
    num_groups = dataset.num_groups
    num_users, num_items = dataset.num_users, dataset.num_items

    print('Data prepare is over!')

    dir_name = os.path.join(args.path, args.train_scheme + "-visualization-" +  args.rec_scheme)

    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

    for embedding_size_t in args.embedding_size_t_list:
        for p in args.p_list:
            for user_weight in args.user_weight_list:
                for orthogonality_weight in args.orthogonality_weight_list:
                    for domain_weight in args.domain_weight_list:
                        for batch_size in args.batch_size_list:
                            for drop_ratio in args.drop_ratio_list:
                                for i in range(1):
                                    up2e = nn.Embedding(num_users, embedding_size_t).to(DEVICE)
                                    v2e = nn.Embedding(num_items, embedding_size_t).to(DEVICE)
                                    g2e = nn.Embedding(num_groups, embedding_size_t).to(DEVICE)

                                    us2e = nn.Embedding(num_users, embedding_size_t).to(DEVICE) # user_combined
                                    # us2e = torch.nn.Parameter(torch.randn(num_groups, num_users, embedding_size_t)* 0.01)

                                    enc_u = up2e
                                    env_i = v2e
                                    # enc_gro = g2e
                                    enc_gro = Group_aggregator(us2e, v2e, g2e, embedding_size_t, g_m_d, drop_ratio, DEVICE).to(DEVICE)
                                    
                                    # build HANCDGR model
                                    model = DAGR(enc_u, env_i, g2e, enc_gro, us2e, embedding_size_t, drop_ratio, args.rec_scheme, DEVICE).to(DEVICE)

                                    # args information
                                    print("DAGR at embedding size_t %d, run Iteration:%d, drop_ratio at %1.2f" %(embedding_size_t, args.n_epoch, drop_ratio))

                                    # train the model
                                    HR_gro = []
                                    NDCG_gro = []
                                    HR_user = []
                                    NDCG_user = []
                                    user_train_time = []
                                    gro_train_time = []
                                    best_hr_gro = 0
                                    best_ndcg_gro = 0
                                    stop = 0
                                    for epoch in range(args.n_epoch):
                                        # set the module in training mode
                                        model.train()
                                        t1_user = time()
                                        if args.train_scheme == "all":
                                            # train the user
                                            training(model, dataset.get_user_dataloader(batch_size), epoch, args.type_m_user, args.lr, user_weight, orthogonality_weight)
                                            # train the group
                                            training(model, dataset.get_group_dataloader(batch_size), epoch, args.type_m_gro, args.lr, user_weight, orthogonality_weight)
                                            # train group-member relationship
                                            if epoch == 0 or epoch == 200:
                                                user_share_embeds, user_private_embeds, gro_embed = training_ug(model, dataset.get_ug_dataloader(batch_size), epoch, p, args.lr, domain_weight, orthogonality_weight)
                                                u_share_file = os.path.join(dir_name, 'u_share_' + str(epoch))
                                                u_private_file = os.path.join(dir_name, 'u_private_' + str(epoch))
                                                g_embeds_file = os.path.join(dir_name, 'g_embeds_' + str(epoch))
                                                np.savetxt(u_share_file, user_share_embeds.cpu().detach().numpy(), fmt='%1.4f', delimiter=' ')
                                                np.savetxt(u_private_file, user_private_embeds.cpu().detach().numpy(), fmt='%1.4f', delimiter=' ')
                                                np.savetxt(g_embeds_file, gro_embed.cpu().detach().numpy(), fmt='%1.4f', delimiter=' ')

                                        elif args.train_scheme == "user-gro":
                                            # train the user
                                            training(model, dataset.get_user_dataloader(batch_size), epoch, args.type_m_user, args.lr, user_weight, orthogonality_weight)
                                            # train the group
                                            training(model, dataset.get_group_dataloader(batch_size), epoch, args.type_m_gro, args.lr, user_weight, orthogonality_weight)
                                        
                                        elif args.train_scheme == "user":
                                            # train the user
                                            training(model, dataset.get_user_dataloader(batch_size), epoch, args.type_m_user, args.lr, user_weight, orthogonality_weight)
                                        elif args.train_scheme == "gro":
                                            # train the group
                                            training(model, dataset.get_group_dataloader(batch_size), epoch, args.type_m_gro, args.lr, user_weight, orthogonality_weight)

                                        # evaluation
                                        t2 = time()
                                        u_hr, u_ndcg = evaluation(model, helper, dataset.get_user_test_dataloader(), args.topK_list, args.type_m_user, DEVICE)
                                        HR_user.append(u_hr)
                                        NDCG_user.append(u_ndcg)

                                        t3 = time()
                                        hr, ndcg = evaluation(model, helper, dataset.get_gro_test_dataloader(), args.topK_list, args.type_m_gro, DEVICE)
                                        HR_gro.append(hr)
                                        NDCG_gro.append(ndcg)

                                        if hr[0] > best_hr_gro:
                                            best_hr_gro = hr[0]
                                            best_ndcg_gro = ndcg[0]
                                            stop = 0
                                        else:
                                            stop = stop + 1
                                        print('Test HR_user:', u_hr, '| Test NDCG_user:', u_ndcg)                                                
                                        print('Test HR_gro:', hr, '| Test NDCG_gro:', ndcg)
                                        if stop >= args.early_stop:
                                            print('*' * 20, 'stop training', '*' * 20)                                                                                            
                                            print('Group Iteration %d [%.1f s]: HR_group NDCG_group' % (epoch, time() - t1_user))
                                            print('HR_user:', HR_user[-1], '| NDCG_user:', NDCG_user[-1])
                                            print('Best HR_gro:', HR_gro[-1], '| Best NDCG_gro:', NDCG_gro[-1])
                                            break

                                    
                                    # EVA_user = np.column_stack((HR_user, NDCG_user, user_train_time))
                                    # EVA_gro = np.column_stack((HR_gro, NDCG_gro))

                                    EVA_data = np.column_stack((HR_user, NDCG_user, HR_gro, NDCG_gro))

                                    print("save to file...")

                                    filename = "EVA_%s_%s_%s_E%d_p%1.3f_user%1.3f_beta%1.3f_gamma%1.5f_batch%d_drop_ratio%1.2f_lr_%1.5f_%d" % (args.rec_scheme, args.type_m_gro, args.type_m_user, embedding_size_t, p, user_weight, orthogonality_weight, domain_weight, batch_size, drop_ratio, args.lr, i)

                                    filename = os.path.join(dir_name, filename)

                                    np.savetxt(filename, EVA_data, fmt='%1.4f', delimiter=' ')

                                    print("Done!")
