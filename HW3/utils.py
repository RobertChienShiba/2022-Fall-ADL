import json
from matplotlib import pyplot as plt
import torch
from torch.nn import CrossEntropyLoss

from tw_rouge import get_rouge

def rl_loss(inputs, targets, greedy_rewards, sample_rewards):
    loss_fct = CrossEntropyLoss(reduction="none")
    sample_probs = torch.sum(-loss_fct(inputs, targets).reshape(len(sample_rewards), -1), 1)
    diff_rewards = (torch.tensor(greedy_rewards) - torch.tensor(sample_rewards)).cuda()
    # print(diff_rewards)
    # baseline = torch.mean(diff_rewards)
    # advantage_func = diff_rewards - baseline
    # print(advantage_func)
    rl_loss = torch.mean(diff_rewards * sample_probs)
    
    return rl_loss

def postprocess_text(preds, labels, avg=True):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(pred) for pred in preds]
    labels = ["\n".join(label) for label in labels]

    return get_rouge(preds, labels, avg)

def save_curve_plot(log_file, args):
    with open(log_file, 'r', encoding='utf-8') as f:
        log = json.load(f)

    fig = plt.figure(figsize=(25, 10))
    model_name = (args.model_name_or_path).split('/')[-1]
    fig.suptitle(f"{model_name} {args.num_epochs}epoch")

    ax = fig.add_subplot(1, 2, 1)
    ax.set_title('Loss')
    if log.get('ml_loss', None):
        # assert len(log['ml_loss']) == args.num_epochs
        ml_limit = min(args.num_epochs, len(log['ml_loss']))
        ax.plot([*range(1, ml_limit + 1)], log['ml_loss'][:ml_limit], label='Supervised Learning')
    if log.get('rl_loss', None):
        # assert len(log['rl_loss']) == args.num_epochs
        rl_limit = min(args.num_epochs, len(log['rl_loss']))
        ax.plot([*range(1, rl_limit + 1)], log['rl_loss'][:rl_limit], label='Reinforcement Learning')
    ax.set_xlim(0.9, min(ml_limit, rl_limit) + 0.1)
    ax.set_xlabel('Epoch')
    ax.legend()

    ax = fig.add_subplot(1, 2, 2)
    ax.set_title('Rouge')
    if 'rl_r1' in log.keys():
        assert len(log['rl_r1']) == len(log['rl_r2']) == len(log['rl_rL'])
        rl_limit = min(len(log['rl_r1']), args.num_epochs)
        ax.plot([*range(1, rl_limit + 1)], log['rl_r1'][:rl_limit], label='RL_rouge-1')
        ax.plot([*range(1, rl_limit + 1)], log['rl_r2'][:rl_limit], label='RL_rouge-2')
        ax.plot([*range(1, rl_limit + 1)], log['rl_rL'][:rl_limit], label='RL_rouge-L')
    if 'val_r1' in log.keys():
        assert len(log['val_r1']) == len(log['val_r2']) == len(log['val_rL'])
        ml_limit = min(len(log['val_r1']), args.num_epochs)
        ax.plot([*range(1, ml_limit + 1)], log['val_r1'][:ml_limit], label='SL_rouge-1')
        ax.plot([*range(1, ml_limit + 1)], log['val_r2'][:ml_limit], label='SL_rouge-2')
        ax.plot([*range(1, ml_limit + 1)], log['val_rL'][:ml_limit], label='SL_rouge-L')
    ax.set_xlim(0.9, min(ml_limit, rl_limit) + 0.1)
    ax.set_xlabel('Epoch')
    ax.legend()

    plt.tight_layout()
    fig.savefig('Learning_Curves_RL.png' if args.use_rl else 'Learning_Curve.png',  bbox_inches="tight")
    plt.close(fig)
