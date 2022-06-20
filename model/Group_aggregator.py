import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from model.attention import AttentionLayer

class Group_aggregator(nn.Module):
    def __init__(self, us2e, v2e, g2e, embedding_dim, group_member_dict, drop_ratio, device):
        super(Group_aggregator, self).__init__() 
        self.us2e = us2e
        self.v2e = v2e
        self.g2e = g2e
        self.embedding_dim = embedding_dim
        self.group_member_dict = group_member_dict
        self.drop_ratio = drop_ratio
        self.device = device
        self.attention = AttentionLayer(2 * embedding_dim, drop_ratio)
        
        
    def forward(self, gro_inputs, item_inputs, rec_scheme):
        group_embeds_full = self.g2e(gro_inputs)     # [B, C] M1-b
        ####### group-members-agg #################
        g_embeds_with_attention = torch.zeros([len(gro_inputs), self.embedding_dim]).to(self.device)
        start = time.time()
        user_ids = [self.group_member_dict[usr.item()] for usr in gro_inputs] # [B,1]
        MAX_MENBER_SIZE = max([len(menb) for menb in user_ids]) # the great group size = 44
        new_group_ids, menb_ids, item_ids, mask = [None]*len(user_ids), [None]*len(user_ids),  [None]*len(user_ids),  [None]*len(user_ids)
        for i in range(len(user_ids)):
            postfix = [0]*(MAX_MENBER_SIZE - len(user_ids[i]))
            new_group_ids[i] = [gro_inputs[i].item()]*len(user_ids[i]) + postfix
            menb_ids[i] = user_ids[i] + postfix
            item_ids[i] = [item_inputs[i].item()]*len(user_ids[i]) + postfix
            mask[i] = [1]*len(user_ids[i]) + postfix
        
        new_group_ids, menb_ids, item_ids, mask = torch.Tensor(new_group_ids).long().to(self.device),\
                                    torch.Tensor(menb_ids).long().to(self.device),\
                                    torch.Tensor(item_ids).long().to(self.device),\
                                    torch.Tensor(mask).float().to(self.device)
        
        # menb_share_emb =  self.us2e[new_group_ids, menb_ids, :] # [B, N, C]
        menb_share_emb =  self.us2e(menb_ids)    # [B, N, C]
        menb_share_emb *= mask.unsqueeze(dim=-1) # [B, N, S]

        #############################################
        ### Vanilla attention layer to get group emb
        #############################################
        item_emb = self.v2e(item_ids) # [B, N, C] 
        item_emb *= mask.unsqueeze(dim=-1) # [B, N, C]

        group_item_emb = torch.cat((menb_share_emb, item_emb), dim=-1) # [B, N, 2C], N=MAX_MENBER_SIZE
        # group_item_emb = group_item_emb.view(-1, group_item_emb.size(-1)) # [B * N, 2C]
        attn_weights = self.attention(group_item_emb)# [B, N, 1]
        # attn_weights = attn_weights.squeeze(dim=-1) # [B, N]
        attn_weights = torch.clip(attn_weights.squeeze(dim=-1), -50, 50)
        attn_weights_exp = attn_weights.exp() * mask # [B, N]
        attn_weights_sm = attn_weights_exp/torch.sum(attn_weights_exp, dim=-1, keepdim=True) # [B, N] 
        attn_weights_sm = attn_weights_sm.unsqueeze(dim=1) # [B, 1, N]
        g_embeds_with_attention = torch.bmm(attn_weights_sm, menb_share_emb) # [B, 1, C]
        g_embeds_with_attention = g_embeds_with_attention.squeeze(dim=1)
        
        if rec_scheme == 'M1-ab':
            gro_emb = g_embeds_with_attention + group_embeds_full 
        elif rec_scheme == 'M1-a':
            gro_emb = g_embeds_with_attention
        elif rec_scheme == 'M1-b':
            gro_emb = group_embeds_full

        gro_emb = gro_emb.to(self.device)

        return gro_emb





            





