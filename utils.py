import torch
import os
from datetime import datetime
class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(preds, targets, k):
    batch_size = targets.size(0)
    _, pred = preds.topk(k, 1, True, True)
    correct = pred.eq(targets.view(-1, 1).expand_as(pred))
    correct_total = correct.view(-1).float().sum()
    return correct_total.item() * (100.0 / batch_size)

def save_model(bleu_4, best_bleu4, epochs_no_improve, patience,
                              encoder, decoder, optimizer, epoch, model_file, args):
    if bleu_4 > best_bleu4:
        best_bleu4 = bleu_4
        epochs_no_improve = 0
        if encoder.get_finetune_state():
                 torch.save({
                'encoder': encoder.state_dict(),
                'decoder': decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'args': vars(args)
            }, model_file)
        else:
            torch.save(decoder.state_dict(), model_file)
       
        print(f'第 {epoch} 轮，BLEU-4 提升至 {bleu_4:.4f}，已保存最佳模型。')
        
    else:
        epochs_no_improve += 1
        print(f'连续 {epochs_no_improve} 轮 BLEU-4 无提升。')

    is_stop = epochs_no_improve >= patience
    if is_stop:
        print(f'触发早停机制：连续 {patience} 轮 BLEU-4 无提升，训练终止。')

    return best_bleu4, epochs_no_improve, is_stop


def log_train(args, bleu4, model_path, log_path='logs/log_train.txt'):

    log_dir = os.path.dirname(log_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    log_lines = [
        f"Time: {now}",
        f"Best BLEU-4: {bleu4 * 100:.2f}",
        f"model: {model_path}"
    ]

    for k, v in vars(args).items():
        log_lines.append(f"{k}: {v}")
    
    log_lines.append("-" * 40)

    with open(log_path, 'a', encoding='utf-8') as f:
        f.write("\n".join(log_lines) + "\n")