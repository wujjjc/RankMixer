from data import load_data, TaobaoDataset, get_collate_fn
from torch.utils.data import DataLoader
from config import Config
from model import RankMixer
from train import train
import torch
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='RankMixer Training')
    parser.add_argument('--train_log', type=str, default='train_log.txt',
                        help='Path to training log file (default: train_log.txt)')
    parser.add_argument('--metric_log', type=str, default='metric_log.txt',
                        help='Path to metric log file (default: metric_log.txt)')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    item_feature, user_profile, user_item_interaction, num_user_id, num_adgroup_id,\
    num_cate_id, num_customer_id, num_brand, num_campaign_id, num_cms_segid, num_cms_group_id = load_data()
    print("开始创建数据")
    train_dataset = TaobaoDataset(user_item_interaction, max_seq_len=Config.max_seq_len, train=True)
    test_dataset = TaobaoDataset(user_item_interaction, max_seq_len=Config.max_seq_len, train=False)
    my_collate_fn = get_collate_fn(user_profile, item_feature)
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True,
                              collate_fn=my_collate_fn, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=Config.batch_size, shuffle=False,
                             collate_fn=my_collate_fn, num_workers=4, pin_memory=True)
    print("数据创建完成")
    device = Config.device
    model = RankMixer(num_user_id, num_adgroup_id, num_cate_id, num_customer_id, num_brand, num_campaign_id, 
                      num_cms_segid, num_cms_group_id, embedding_dim=Config.embedding_dim, head=8, expert=8, block_num=8, preln=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.learning_rate, weight_decay=Config.weight_decay)
    criterion = torch.nn.BCEWithLogitsLoss()
    train(model, train_loader, test_loader, optimizer, criterion, device,
          train_log_file=args.train_log, metric_log_file=args.metric_log)
    