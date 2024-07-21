import math
import torch.optim as optim # 最適化モジュール

# 学習率スケジュールの設定（線形ウォームアップとコサイン減衰）
def get_scheduler(optimizer, num_train_steps):
    num_steps = num_train_steps // 2
    num_warmup_steps = num_steps 
    num_decay_steps = num_steps

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        if current_step < num_warmup_steps + num_decay_steps:
            progress = (current_step - num_warmup_steps) / float(max(1, num_decay_steps))
            return 0.5 * (1 + math.cos(math.pi * progress))
        return 0.0

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
