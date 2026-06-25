import torch
from config import Config
import numpy as np
from sklearn.metrics import roc_auc_score
from torch.optim.lr_scheduler import LambdaLR


def _unpack_batch(batch, device):
    """将batch中的tensor移到指定设备"""
    return {k: v.to(device) for k, v in batch.items()}


def test(model, test_loader, device):
    """测试,获得auc

    Args:
        model: 模型
        test_loader: 测试集
        device: 设备

    Returns:
        auc: auc值
    """
    model.eval()
    all_labels = []
    all_preds = []
    with torch.no_grad():
        num_active_experts_total = 0
        num_experts_total = 0
        for batch in test_loader:
            b = _unpack_batch(batch, device)
            output, num_active_experts, num_experts = model(
                b['user_ids'], b['item_ids'], b['cate_ids'], b['customer_ids'],
                b['brand'], b['campaign_ids'], b['cms_segid'], b['cms_group_id'],
                b['age_level'], b['gender'], b['pvalue_level'], b['shopping_level'],
                b['occupation'], b['new_user_class_level'],
                b['history_item_lists'], b['masks'], b['price']
            )
            num_active_experts_total += num_active_experts
            num_experts_total += num_experts
            all_labels.extend(b['clks'].cpu().numpy())
            all_preds.extend(output.cpu().numpy())
    return roc_auc_score(np.array(all_labels), np.array(all_preds)), num_active_experts_total, num_experts_total


def _get_warmup_lr(step, warmup_steps, base_lr):
    """线性 warmup：step 从 0 到 warmup_steps 时，lr 从 0 线性增长到 base_lr"""
    if step >= warmup_steps:
        return 1.0
    return step / max(warmup_steps, 1)


def train(model, train_loader, test_loader, optimizer, criterion, device,
          train_log_file='train_log.txt', metric_log_file='metric_log.txt'):
    """训练

    Args:
        model: 模型
        train_loader: 训练集
        test_loader: 测试集
        optimizer: 优化器
        criterion: 损失函数
        device: 设备
        train_log_file: 训练日志文件路径
        metric_log_file: 指标日志文件路径
    """
    warmup_steps = Config.warmup_steps
    grad_clip = Config.grad_clip_norm
    scheduler = LambdaLR(optimizer, lambda step: _get_warmup_lr(step, warmup_steps, Config.learning_rate))
    global_step = 0

    print(f"开始训练 (warmup_steps={warmup_steps}, grad_clip={grad_clip})")
    best_auc = 0
    for epoch in range(Config.epochs):
        model.train()
        total_loss = 0
        for batch in train_loader:
            b = _unpack_batch(batch, device)
            optimizer.zero_grad()
            output, loss = model(
                b['user_ids'], b['item_ids'], b['cate_ids'], b['customer_ids'],
                b['brand'], b['campaign_ids'], b['cms_segid'], b['cms_group_id'],
                b['age_level'], b['gender'], b['pvalue_level'], b['shopping_level'],
                b['occupation'], b['new_user_class_level'],
                b['history_item_lists'], b['masks'], b['price']
            )
            loss = Config.lamb * loss + criterion(output, b['clks'])
            loss.backward()
            if grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1
            total_loss += loss.item()
        with open(train_log_file, 'a') as f:
            f.write(f'Epoch {epoch+1}/{Config.epochs}, Loss: {total_loss/len(train_loader)}\n')
        auc, num_active_experts_total, num_experts_total = test(model, test_loader, device)
        with open(metric_log_file, 'a') as f:
            f.write(f'Epoch {epoch+1}/{Config.epochs}, AUC: {auc}, '
                    f'Active Experts: {num_active_experts_total}, '
                    f'Total Experts: {num_experts_total}, '
                    f'Active Experts Ratio: {num_active_experts_total/num_experts_total:.4f}\n')
        if auc > best_auc:
            best_auc = auc
            torch.save(model.state_dict(), 'best_model.pth')
    print("训练完成")
