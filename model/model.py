
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.prediction import PredictLayer, PredDomainLayer
from model.useremb import Userprivate, UGshare, Shared_encoder_pred_domain
from utils.util import DiffLoss, ReverseLayerF, CosineSimilarity

class DAGR(nn.Module):
    def __init__(self, enc_u, env_i, g2e, enc_gro, enc_us2e, u2e_explicit, g2e_explicit, embedding_dim, drop_ratio, rec_scheme, device):
        super(DAGR, self).__init__()
        self.enc_u = enc_u
        self.env_i = env_i
        self.g2e = g2e
        self.enc_gro = enc_gro
        self.enc_us2e = enc_us2e
        self.u2e_explicit = u2e_explicit
        self.g2e_explicit = g2e_explicit
        self.device = device
        self.embedding_dim_t = embedding_dim
        self.drop_ratio = drop_ratio
        self.rec_scheme = rec_scheme
        self.shared_encoder_pred_domain = Shared_encoder_pred_domain(embedding_dim, 1)
        
        self.loss_similarity = torch.nn.BCELoss()
        self.loss_diff = DiffLoss()
        self.cosine_similarity = CosineSimilarity()
    
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

    def forward_gro_explicit(self, gro_inputs, g_item_inputs, rating):
        item_embeds_full = self.env_i(g_item_inputs)   # [B, C]
        # g_embeds_full = self.enc_gro(gro_inputs, g_item_inputs, self.rec_scheme) # [B, C]
        g_embeds_full = self.g2e_explicit[gro_inputs] # [B, C]
        
        preds = self.cosine_similarity(g_embeds_full, item_embeds_full)

        loss_explicit = self.loss_similarity(preds, rating)

        return loss_explicit

    def forward_user(self, user_inputs, u_item_inputs):    
        item_embeds_full = self.env_i(u_item_inputs)   # [B, C]
        u_embeds_private = self.enc_u(user_inputs) # [B, C]
        # u_embeds_share = self.enc_us2e(user_inputs)

        u_embeds_full = u_embeds_private

        # predictions 
        element_embeds = torch.mul(u_embeds_full, item_embeds_full)
        preds = torch.sigmoid(element_embeds.sum(1)) # agree loss
        # preds = element_embeds.sum(1) # bpr loss
        return preds
    
    def forward_user_explicit(self, user_inputs, u_item_inputs, rating):
        item_embeds_full = self.env_i(u_item_inputs)   # [B, C]
        u_embeds_private = self.enc_u(user_inputs) # [B, C]
        # u_embeds_share = self.enc_us2e(user_inputs)
        u_embeds_full = u_embeds_private

        preds = self.cosine_similarity(u_embeds_full, item_embeds_full)

        loss_explicit = self.loss_similarity(preds, rating)

        return loss_explicit

       
    def forward_ug(self, group, member, p):
        user_share_embeds = self.enc_us2e[group, member, :]
        # user_share_embeds = self.enc_us2e(member)
        user_private_embeds = self.enc_u(member)
        # gro_embed = self.g2e(group)
        
        # user_dann = self.get_adversarial_result(user_share_embeds, p, source=True)
        # gro_dann = self.get_adversarial_result(gro_embed, p, source=False)
        # domain_loss = (user_dann + gro_dann)

        member_diff = self.loss_diff(user_share_embeds, user_private_embeds)

        return member_diff

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


