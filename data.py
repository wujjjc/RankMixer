import pandas as pd
from torch.utils.data import Dataset, DataLoader
from config import Config
import torch
"""
原始样本骨架raw_sample
我们从淘宝网站中随机抽样了114万用户8天内的广告展示/点击日志（2600万条记录），构成原始的样本骨架。
字段说明如下：
(1) user_id：脱敏过的用户ID；
(2) adgroup_id：脱敏过的广告单元ID；
(3) time_stamp：时间戳；
(4) pid：资源位；
(5) noclk：为1代表没有点击；为0代表点击；
(6) clk：为0代表没有点击；为1代表点击；
我们用前面7天的做训练样本（20170506-20170512），用第8天的做测试样本（20170513）。 20170512: 1494518400

广告基本信息表ad_feature
本数据集涵盖了raw_sample中全部广告的基本信息。字段说明如下：
(1) adgroup_id：脱敏过的广告ID；
(2) cate_id：脱敏过的商品类目ID；
(3) campaign_id：脱敏过的广告计划ID；
(4) customer：脱敏过的广告主ID；
(5) brand：脱敏过的品牌ID；
(6) price: 宝贝的价格
其中一个广告ID对应一个商品（宝贝），一个宝贝属于一个类目，一个宝贝属于一个品牌。

用户基本信息表user_profile
本数据集涵盖了raw_sample中全部用户的基本信息。字段说明如下：
(1) user：脱敏过的用户ID；
(2) cms_segid：微群ID；
(3) cms_group_id：cms_group_id；
(4) final_gender_code：性别 1:男,2:女；
(5) age_level：年龄层次；0-6
(6) pvalue_level：消费档次，1:低档，2:中档，3:高档；
(7) shopping_level：购物深度，1:浅层用户,2:中度用户,3:深度用户
(8) occupation：是否大学生 ，1:是,0:否
(9) new_user_class_level：城市层级 1-4
"""
def load_data(user_file="dataset/user_profile.csv", item_file="dataset/ad_feature.csv", raw_file="dataset/raw_sample.csv"):
    """读取csv文件里面的数据,并将离散数据进行映射,离散数据映射都从1开始,连续数据保持不变

    Args:
        user_file (_type_): 用户csv路径
        item_file (_type_): 商品csv路径
        raw_file (_type_): 交互csv路径

    Returns:
        item_feature:{
            enc_item_id:{
            'enc_cate_id': int,
            'enc_customer_id': int,
            'enc_brand': int,
            'enc_campaign_id': int,
            'price': float
            }}
        
        user_profile:{
        enc_user_id:{
            'enc_cms_segid': int,
            'enc_cms_group_id': int,
            'final_gender_code': int,1 - 2
            'age_level': int,0-6
            'pvalue_level': int,1-3 缺失值填0
            'shopping_level': int,1-3
            'occupation': int,0-1
            'new_user_class_level': int,1-4 缺失值填0
            }}
        
        user_item_interaction:{
        user_id:[
        (item_id, time_stamp, enc_pid, clk) 以及按时间排好序
        ]
        }
    """
    print("Loading data...")
    user_df = pd.read_csv(user_file)
    item_df = pd.read_csv(item_file)
    raw_df = pd.read_csv(raw_file)
    user_df.columns = user_df.columns.str.strip() # 去除列名中的空格
    item_df.columns = item_df.columns.str.strip() # 去除列名中的空格
    raw_df.columns = raw_df.columns.str.strip()  # 去除列名中的空格
    user_df['age_level'] = user_df['age_level'].fillna(0)
    user_df['pvalue_level'] = user_df['pvalue_level'].fillna(0)
    user_df['new_user_class_level'] = user_df['new_user_class_level'].fillna(0)
    item_df['brand'] = item_df['brand'].fillna(0)
    user_df['enc_user_id'] = pd.factorize(user_df['userid'])[0] + 1
    num_user_id = user_df['enc_user_id'].max() + 1
    user_df['enc_cms_segid'] = pd.factorize(user_df['cms_segid'])[0] + 1
    num_cms_segid = user_df['enc_cms_segid'].max() + 1
    user_df['enc_cms_group_id'] = pd.factorize(user_df['cms_group_id'])[0] + 1
    num_cms_group_id = user_df['enc_cms_group_id'].max() + 1
    item_df['enc_adgroup_id'] = pd.factorize(item_df['adgroup_id'])[0] + 1
    num_adgroup_id = item_df['enc_adgroup_id'].max() + 1
    item_df['enc_cate_id'] = pd.factorize(item_df['cate_id'])[0] + 1
    num_cate_id = item_df['enc_cate_id'].max() + 1
    item_df['enc_customer_id'] = pd.factorize(item_df['customer'])[0] + 1
    num_customer_id = item_df['enc_customer_id'].max() + 1
    item_df['enc_brand'] = pd.factorize(item_df['brand'])[0] + 1
    num_brand = item_df['enc_brand'].max() + 1
    item_df['enc_campaign_id'] = pd.factorize(item_df['campaign_id'])[0] + 1
    num_campaign_id = item_df['enc_campaign_id'].max() + 1
    user_df['age_level'] = user_df['age_level'].astype(int)
    user_df['pvalue_level'] = user_df['pvalue_level'].astype(int)
    user_df['shopping_level'] = user_df['shopping_level'].astype(int)
    user_df['occupation'] = user_df['occupation'].astype(int)
    user_df['new_user_class_level'] = user_df['new_user_class_level'].astype(int)
    item_df['price'] = item_df['price'].astype(float)
    user_mapping = dict(zip(user_df['userid'], user_df['enc_user_id']))
    item_mapping = dict(zip(item_df['adgroup_id'], item_df['enc_adgroup_id']))
    user_profile = user_df.set_index('enc_user_id')[['enc_cms_segid', 'enc_cms_group_id', 'final_gender_code', 'age_level', 'pvalue_level', 'shopping_level', 'occupation', 'new_user_class_level']].to_dict(orient='index')
    item_feature = item_df.set_index('enc_adgroup_id')[['enc_cate_id', 'enc_customer_id', 'enc_brand', 'enc_campaign_id', 'price']].to_dict(orient='index')
    raw_df['user'] = raw_df['user'].map(user_mapping)
    raw_df['adgroup_id'] = raw_df['adgroup_id'].map(item_mapping)
    raw_df.dropna(subset=['user', 'adgroup_id'], inplace=True) # 删除映射后为NaN的行
    raw_df['user'] = raw_df['user'].astype(int)
    raw_df['adgroup_id'] = raw_df['adgroup_id'].astype(int)
    raw_df['enc_pid'] = pd.factorize(raw_df['pid'])[0] + 1
    raw_df['time_stamp'] = raw_df['time_stamp'].astype(int)
    raw_df['clk'] = raw_df['clk'].astype(int)
    user_item_interaction = {}
    for uid, group in raw_df.groupby('user'):
        interactions = list(zip(group['adgroup_id'], group['time_stamp'], group['enc_pid'], group['clk']))
        interactions = sorted(interactions, key=lambda x: x[1])  # 按照时间戳排序
        user_item_interaction[uid] = interactions
    print("Data loaded successfully!")
    return item_feature, user_profile, user_item_interaction, num_user_id, num_adgroup_id, num_cate_id, num_customer_id, num_brand, num_campaign_id, num_cms_segid, num_cms_group_id

class TaobaoDataset(Dataset):
    def __init__(self, user_item_interaction, max_seq_len=Config.max_seq_len, train=True):
        """数据集dataset

        Args:
            user_item_interaction (dict): 用户交互数据
            max_seq_len (int): 保存的用户交互的最大长度,超过这个长度的交互记录将被丢弃
            train (bool): 训练集还是测试集, 训练集只保留end_time之前的交互记录,测试集只保留end_time之后的交互记录
        
        Returns:
            user_interaction: [
                (item_id, enc_pid, clk, [历史交互的item_id列表])
            ]
        """
        self.max_seq_len = max_seq_len
        self.train = train
        self.user_interaction = []
        for uid, interactions in user_item_interaction.items():
            item_list = []
            for item_id, time_stamp, enc_pid, clk in interactions:
                if train and time_stamp >= Config.end_time: # 训练集只保留end_time之前的交互
                    break
                if not train and time_stamp < Config.end_time: # 测试集只保留end_time之后的交互
                    if clk:
                        item_list.append(item_id)
                    continue
                self.user_interaction.append((uid, item_id, enc_pid, clk, item_list[-max_seq_len:]))
                if clk:
                    item_list.append(item_id)

    def __len__(self):
        return len(self.user_interaction)
    
    def __getitem__(self, idx):
        return self.user_interaction[idx]

def get_collate_fn(user_profile, item_feature, max_seq_len=Config.max_seq_len):
    """自定义的collate_fn函数, 用于DataLoader中对batch进行处理

    Args:
        batch (list): batch数据, 每个元素是一个(user_id, item_id, enc_pid, clk, history_item_list)
        item_feature:{
            enc_item_id:{
            'enc_cate_id': int,
            'enc_customer_id': int,
            'enc_brand': int,
            'enc_campaign_id': int,
            'price': float
            }}
        
        user_profile:{
        enc_user_id:{
            'enc_cms_segid': int,
            'enc_cms_group_id': int,
            'final_gender_code': int,1 - 2
            'age_level': int,0-6
            'pvalue_level': int,1-3 缺失值填0
            'shopping_level': int,1-3
            'occupation': int,0-1
            'new_user_class_level': int,1-4 缺失值填0
            }}
        max_seq_len (int): 用户交互历史的最大长度, 超过这个长度的交互记录将被丢弃

    Returns:
        {
        'user_ids': 用户id 1 - m #[B, ]
        'item_ids': 物品id 1 - n #[B, ]
        'pids':  #[B, ]
        'clks': 是否点击 01 #[B, ]
        'cate_ids': 物品类目id 1 - k #[B, ]
        'customer_ids': 广告主id 1 - l #[B, ]
        'brand': 品牌id 1 - r #[B, ]
        'campaign_ids': 广告计划id 1 - s #[B, ]
        'price': 商品价格 #[B, ]
        'cms_segid': 微群id 1 - t #[B, ]
        'cms_group_id': cms_group_id 1 - u #[B, ]
        gender: 性别 1 - 2 #[B, ]
        age_level: 年龄层次 0-6 #[B, ]
        pvalue_level: 消费档次 1-3 #[B, ]
        shopping_level: 购物深度 1-3 #[B, ]
        occupation: 职业 0-1 #[B, ]
        new_user_class_level: 城市层级 1-4 #[B, ]
        history_item_lists: 用户交互历史的物品id列表 #[B, max_seq_len]
        masks: 用户交互历史的mask, 1代表有交互记录, 0代表没有交互记录 #[B, max_seq_len]
        }
    """
    def collate_fn(batch):
        user_ids = [x[0] for x in batch]
        item_ids = [x[1] for x in batch]
        pids = [x[2] for x in batch]
        clks = [x[3] for x in batch]
        cate_ids = [item_feature[item_id]['enc_cate_id'] for item_id in item_ids]
        customer_ids = [item_feature[item_id]['enc_customer_id'] for item_id in item_ids]
        brand = [item_feature[item_id]['enc_brand'] for item_id in item_ids]
        campaign_ids = [item_feature[item_id]['enc_campaign_id'] for item_id in item_ids]
        price = [item_feature[item_id]['price'] for item_id in item_ids]
        cms_segid = [user_profile[uid]['enc_cms_segid'] for uid in user_ids]
        cms_group_id = [user_profile[uid]['enc_cms_group_id'] for uid in user_ids]
        gender = [user_profile[uid]['final_gender_code'] for uid in user_ids]
        age_level = [user_profile[uid]['age_level'] for uid in user_ids]
        pvalue_level = [user_profile[uid]['pvalue_level'] for uid in user_ids]
        shopping_level = [user_profile[uid]['shopping_level'] for uid in user_ids]
        occupation = [user_profile[uid]['occupation'] for uid in user_ids]
        new_user_class_level = [user_profile[uid]['new_user_class_level'] for uid in user_ids]
        history_item_lists = []
        masks = []
        for x in batch:
            his = x[4]
            pad = max_seq_len - len(his)
            his = [0] * pad + his[-max_seq_len:]
            history_item_lists.append(his)
            masks.append([0] * pad + [1] * (max_seq_len - pad))
        return {
            'user_ids': torch.tensor(user_ids, dtype=torch.long), #[B, ]
            'item_ids': torch.tensor(item_ids, dtype=torch.long), #[B, ]
            'pids': torch.tensor(pids, dtype=torch.long), #[B, ]
            'clks': torch.tensor(clks, dtype=torch.float), #[B, ]
            'cate_ids': torch.tensor(cate_ids, dtype=torch.long), #[B, ]
            'customer_ids': torch.tensor(customer_ids, dtype=torch.long), #[B, ]
            'brand': torch.tensor(brand, dtype=torch.long), #[B, ]
            'campaign_ids': torch.tensor(campaign_ids, dtype=torch.long), #[B, ]
            'price': torch.tensor(price, dtype=torch.float), #[B, ]
            'cms_segid': torch.tensor(cms_segid, dtype=torch.long), #[B, ]
            'cms_group_id': torch.tensor(cms_group_id, dtype=torch.long), #[B, ]
            'gender': torch.tensor(gender, dtype=torch.long), #[B, ]
            'age_level': torch.tensor(age_level, dtype=torch.long), #[B, ]
            'pvalue_level': torch.tensor(pvalue_level, dtype=torch.long), #[B, ]
            'shopping_level': torch.tensor(shopping_level, dtype=torch.long), #[B, ]
            'occupation': torch.tensor(occupation, dtype=torch.long), #[B, ]
            'new_user_class_level': torch.tensor(new_user_class_level, dtype=torch.long), #[B, ]
            'history_item_lists': torch.tensor(history_item_lists, dtype=torch.long), #[B, max_seq_len]
            'masks': torch.tensor(masks, dtype=torch.float) #[B, max_seq_len]
        }
    return collate_fn