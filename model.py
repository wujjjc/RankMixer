import torch
import torch.nn as nn
import math
from config import Config
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size:list, output_size, activation=nn.GELU, dropout=0.1):
        """多层感知机

        Args:
            input_size (_type_): 输入
            hidden_size (list): 隐藏层
            output_size (_type_): 输出
            activation (_type_, optional): 激活函数. Defaults to nn.GELU.
            dropout (float, optional): Dropout概率. Defaults to 0.1.
        """
        super(MLP, self).__init__()
        dense = []
        for hidden in hidden_size:
            dense.append(nn.Linear(input_size, hidden))
            dense.append(activation())
            dense.append(nn.Dropout(dropout))
            input_size = hidden
        dense.append(nn.Linear(input_size, output_size))
        self.net = nn.Sequential(*dense)

    def forward(self, x):
        return self.net(x)

class RankMixerBlock(nn.Module):
    def __init__(self, head=32, emb_size=32, expert=4, preln=False):
        super(RankMixerBlock, self).__init__()
        self.head = head
        self.emb_size = emb_size
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.route = nn.Parameter(torch.empty(head, emb_size, expert))
        self.w1 = nn.Parameter(torch.empty(head, expert, emb_size, 4 * emb_size))
        self.b1 = nn.Parameter(torch.zeros(head, expert, 4 * emb_size))
        self.w2 = nn.Parameter(torch.empty(head, expert, 4 * emb_size, emb_size))
        self.b2 = nn.Parameter(torch.zeros(head, expert, emb_size))
        self.preln = preln
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.route, a=math.sqrt(5))
        self.relu = nn.ReLU()
        self.gelu = nn.GELU()
    
    def tokenmixing(self, token):
        """把token按注意力头重新组合

        Args:
            token (_type_): [batch_size, seq_len, embedding_dim]
        
        return:
            token: [batch_size, head, seq_len * embedding_dim // head]
        """
        batch_size, seq_len, embedding_dim = token.size()
        token = token.view(batch_size, seq_len, self.head, embedding_dim // self.head)
        token = token.permute(0, 2, 1, 3).contiguous()
        token = token.view(batch_size, self.head, seq_len * embedding_dim // self.head)
        return token

    def forward(self, token):
        # token: [batch_size, seq_len, embedding_dim]
        if self.preln:
            token = token + self.tokenmixing(self.norm1(token))
        else:
            token = self.norm1(token + self.tokenmixing(token))  # [batch_size, head, seq_len * embedding_dim // head]
        r_train = torch.softmax(torch.einsum('bhe, hex -> bhx', token, self.route), dim=-1) # [batch_size, head, expert]
        r_infer = torch.relu(torch.einsum('bhe, hex -> bhx', token, self.route)) # [batch_size, head, expert]
        r = r_train if self.training else r_infer
        if self.preln:
            token_ = self.gelu(torch.einsum('bhe, hxek -> bhxk', self.norm2(token), self.w1) + self.b1)  # [batch_size, head, expert, 4 * embedding_dim]
        else:
            token_ = self.gelu(torch.einsum('bhe, hxek -> bhxk', token, self.w1) + self.b1)  # [batch_size, head, expert, 4 * embedding_dim]
        token_ = torch.einsum('bhxk, hxke-> bhxe', token_, self.w2) + self.b2  # [batch_size, head, expert, embedding_dim]
        token_ = (token_ * r.unsqueeze(-1)).sum(dim=-2)  # [batch_size, head, embedding_dim]
        if self.preln:
            token = token + token_
        else:
            token = self.norm2(token + token_)  # [batch_size, head, embedding_dim]
        loss = r_infer.sum(dim=-1).mean()
        num_experts = r.shape[0] * r.shape[1] * r.shape[2]
        num_active_experts = (r > 0).sum().item()
        return token, loss, num_active_experts, num_experts

class sRankMixerBlock(nn.Module):
    def __init__(self, head=32, emb_size=32, preln=False):
        super(sRankMixerBlock, self).__init__()
        self.head = head
        self.emb_size = emb_size
        self.norm1 = nn.LayerNorm(emb_size)
        self.norm2 = nn.LayerNorm(emb_size)
        self.w1 = nn.Parameter(torch.empty(head, emb_size, 4 * emb_size))
        self.w2 = nn.Parameter(torch.empty(head, 4 * emb_size, emb_size))
        nn.init.kaiming_uniform_(self.w1, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.w2, a=math.sqrt(5))
        self.gelu = nn.GELU()
        self.preln = preln
    
    def tokenmixing(self, token):
        """把token按注意力头重新组合

        Args:
            token (_type_): [batch_size, seq_len, embedding_dim]
        
        return:
            token: [batch_size, head, seq_len * embedding_dim // head]
        """
        batch_size, seq_len, embedding_dim = token.size()
        token = token.view(batch_size, seq_len, self.head, embedding_dim // self.head)
        token = token.permute(0, 2, 1, 3).contiguous()
        token = token.view(batch_size, self.head, seq_len * embedding_dim // self.head)
        return token

    def forward(self, token):
        # token: [batch_size, head, embedding_dim]
        if self.preln:
            token = token + self.tokenmixing(self.norm1(token))
        else:
            token = self.norm1(token + self.tokenmixing(token))  # [batch_size, head, seq_len * embedding_dim // head]
        token_ = self.gelu(torch.einsum('bhe, het -> bht', token, self.w1))  # [batch_size, head, 4 * embedding_dim]
        token_ = torch.einsum('bht, hte-> bhe', token_, self.w2)  # [batch_size, head, embedding_dim]
        if self.preln:
            token = token_ + token  # [batch_size, head, embedding_dim]
        else:
            token = self.norm2(token_ + token)  # [batch_size, head, embedding_dim]
        loss = 0.0
        return token, loss, 1, 1
    


class RankMixer(nn.Module):
    def __init__(self, num_user_id, num_adgroup_id, num_cate_id, num_customer_id, num_brand, num_campaign_id,
                 num_cms_segid, num_cms_group_id, embedding_dim=64, head=32, expert=4, block_num=2, MoE=True, preln=False):
        super(RankMixer, self).__init__()
        self.head = head
        self.user_embedding = nn.Embedding(num_user_id, embedding_dim, padding_idx=0)
        self.adgroup_embedding = nn.Embedding(num_adgroup_id, embedding_dim, padding_idx=0)
        self.cate_embedding = nn.Embedding(num_cate_id, embedding_dim, padding_idx=0)
        self.customer_embedding = nn.Embedding(num_customer_id, embedding_dim, padding_idx=0)
        self.brand_embedding = nn.Embedding(num_brand, embedding_dim, padding_idx=0)
        self.campaign_embedding = nn.Embedding(num_campaign_id, embedding_dim, padding_idx=0)
        self.cms_segid_embedding = nn.Embedding(num_cms_segid, embedding_dim, padding_idx=0)
        self.cms_group_id_embedding = nn.Embedding(num_cms_group_id, embedding_dim, padding_idx=0)
        self.age_embedding = nn.Embedding(7, embedding_dim)
        self.gender_embedding = nn.Embedding(3, embedding_dim)
        self.pvalue_embedding = nn.Embedding(4, embedding_dim)
        self.shopping_embedding = nn.Embedding(4, embedding_dim)
        self.occupation_embedding = nn.Embedding(2, embedding_dim)
        self.new_user_class_embedding = nn.Embedding(5, embedding_dim)
        self.price_embedding = MLP(1, [embedding_dim], embedding_dim)
        if MoE:
            self.blocks = nn.ModuleList([RankMixerBlock(head=head, emb_size=Config.token_dim, expert=expert, preln=preln) for _ in range(block_num)])
        else:
            self.blocks = nn.ModuleList([sRankMixerBlock(head=head, emb_size=Config.token_dim, preln=preln) for _ in range(block_num)])
        self.mlp = MLP(16 * Config.token_dim, [256, 64, 16], 1)
        # 16 个语义 token 的单层投影（对齐论文的 tokenization）
        self.token_proj_user_id = nn.Linear(embedding_dim, Config.token_dim)           # user_id
        self.token_proj_age = nn.Linear(embedding_dim, Config.token_dim)               # age_level
        self.token_proj_gender = nn.Linear(embedding_dim, Config.token_dim)            # gender
        self.token_proj_occupation = nn.Linear(embedding_dim, Config.token_dim)        # occupation
        self.token_proj_adgroup_id = nn.Linear(embedding_dim, Config.token_dim)        # adgroup_id
        self.token_proj_cate_id = nn.Linear(embedding_dim, Config.token_dim)           # cate_id
        self.token_proj_brand = nn.Linear(embedding_dim, Config.token_dim)             # brand
        self.token_proj_campaign_id = nn.Linear(embedding_dim, Config.token_dim)       # campaign_id
        self.token_proj_customer_id = nn.Linear(embedding_dim, Config.token_dim)       # customer_id
        self.token_proj_price = nn.Linear(embedding_dim, Config.token_dim)             # price
        self.token_proj_cms_segid = nn.Linear(embedding_dim, Config.token_dim)         # cms_segid
        self.token_proj_cms_group_id = nn.Linear(embedding_dim, Config.token_dim)      # cms_group_id
        self.token_proj_pvalue = nn.Linear(embedding_dim, Config.token_dim)            # pvalue_level
        self.token_proj_shopping = nn.Linear(embedding_dim, Config.token_dim)          # shopping_level
        self.token_proj_new_user_class = nn.Linear(embedding_dim, Config.token_dim)    # new_user_class_level
        self.token_proj_history = nn.Linear(embedding_dim, Config.token_dim)           # his
        self.norm = nn.LayerNorm(Config.token_dim)
        self.preln = preln
        self.gate = nn.Parameter(torch.zeros(head))
        
    
    def forward(self, user_id, adgroup_id, cate_id, customer_id, brand, campaign_id, cms_segid, cms_group_id, age, gender, pvalue, shopping, occupation, new_user_class, his, mask, price):
        # 1. Embedding lookup
        user_emb = self.user_embedding(user_id)
        adgroup_emb = self.adgroup_embedding(adgroup_id)
        cate_emb = self.cate_embedding(cate_id)
        customer_emb = self.customer_embedding(customer_id)
        brand_emb = self.brand_embedding(brand)
        campaign_emb = self.campaign_embedding(campaign_id)
        cms_segid_emb = self.cms_segid_embedding(cms_segid)
        cms_group_id_emb = self.cms_group_id_embedding(cms_group_id)
        age_emb = self.age_embedding(age)
        gender_emb = self.gender_embedding(gender)
        pvalue_emb = self.pvalue_embedding(pvalue)
        shopping_emb = self.shopping_embedding(shopping)
        occupation_emb = self.occupation_embedding(occupation)
        new_user_class_emb = self.new_user_class_embedding(new_user_class)
        his_emb = self.adgroup_embedding(his) # [B, seq_len, D]
        his_emb = his_emb.sum(-2) / mask.sum(-1, keepdim=True).clamp(min=1e-8)  # [B, D]
        price_emb = self.price_embedding(price.unsqueeze(-1)) # [B, D]

        # 2. Semantic Tokenization: 16 个语义组 → 单层 projection → 16 个 token
        tok_user_id = self.token_proj_user_id(user_emb)                              # [B, 128]
        tok_age = self.token_proj_age(age_emb)                                       # [B, 128]
        tok_gender = self.token_proj_gender(gender_emb)                              # [B, 128]
        tok_occupation = self.token_proj_occupation(occupation_emb)                  # [B, 128]
        tok_adgroup_id = self.token_proj_adgroup_id(adgroup_emb)                     # [B, 128]
        tok_cate_id = self.token_proj_cate_id(cate_emb)                              # [B, 128]
        tok_brand = self.token_proj_brand(brand_emb)                                 # [B, 128]
        tok_campaign_id = self.token_proj_campaign_id(campaign_emb)                  # [B, 128]
        tok_customer_id = self.token_proj_customer_id(customer_emb)                  # [B, 128]
        tok_price = self.token_proj_price(price_emb)                                 # [B, 128]
        tok_cms_segid = self.token_proj_cms_segid(cms_segid_emb)                     # [B, 128]
        tok_cms_group_id = self.token_proj_cms_group_id(cms_group_id_emb)            # [B, 128]
        tok_pvalue = self.token_proj_pvalue(pvalue_emb)                              # [B, 128]
        tok_shopping = self.token_proj_shopping(shopping_emb)                        # [B, 128]
        tok_new_user_class = self.token_proj_new_user_class(new_user_class_emb)      # [B, 128]
        tok_history = self.token_proj_history(his_emb)                               # [B, 128]

        token = torch.stack([tok_user_id, tok_age, tok_gender, tok_occupation,
                             tok_adgroup_id, tok_cate_id, tok_brand, tok_campaign_id,
                             tok_customer_id, tok_price, tok_cms_segid, tok_cms_group_id,
                             tok_pvalue, tok_shopping, tok_new_user_class, tok_history], dim=1)  # [B, 16, 128]
        loss = 0.0
        num_active_experts_total = 0
        num_experts_total = 0
        for net in self.blocks:
            token, loss_, num_active_experts, num_experts = net(token)
            loss += loss_
            num_active_experts_total += num_active_experts
            num_experts_total += num_experts
        if self.preln:
            token = self.norm(token) # [B, 16, 128]
        token = token * torch.sigmoid(self.gate).unsqueeze(0).unsqueeze(-1) # [B, 16, 128]
        out = (self.mlp(torch.flatten(token, start_dim=1))).squeeze(-1) # [batch_size,]
        if self.training:
            return out, loss
        return torch.sigmoid(out), num_active_experts_total, num_experts_total
