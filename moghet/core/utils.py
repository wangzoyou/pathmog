#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数模块

包含:
1. EarlyStopping: 早停机制，用于防止过拟合
2. setup_seed: 设置随机种子，确保结果可复现

作者: AI Assistant
日期: 2024-07-15
"""

import numpy as np
import torch
import random
import os
import copy
from sklearn.model_selection import StratifiedKFold


class EarlyStopping:
    """
    早停机制
    
    当验证指标连续多轮没有改善时，停止训练
    """
    def __init__(self, patience=10, verbose=False, delta=0, mode='min', path='checkpoint.pt'):
        """
        初始化早停对象
        
        参数:
            patience: 容忍验证指标不改善的轮数
            verbose: 是否打印早停信息
            delta: 最小改善阈值
            mode: 'min' 或 'max'，表示指标是越小越好还是越大越好
            path: 保存最佳模型的路径
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.delta = delta
        self.path = path
        self.best_model = None
        self.mode = mode
        self.improved = False
    
    def __call__(self, score, model, epoch=None):
        """
        在每个epoch结束时调用
        
        参数:
            score: 当前epoch的验证指标
            model: 当前模型
            epoch: 当前epoch数
        """
        if epoch is not None:
            current_epoch = epoch
        else:
            current_epoch = self.counter
        
        if self.mode == 'min':
            score = -score
        
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = current_epoch
            self.save_checkpoint(score, model)
            if self.verbose:
                print(f"Validation {self.mode} improved (-inf --> {score:.6f})")
            self.improved = True
        elif score > self.best_score + self.delta:
            self.best_score = score
            self.best_epoch = current_epoch
            self.save_checkpoint(score, model)
            self.counter = 0
            if self.verbose:
                if self.mode == 'min':
                    print(f"Validation {self.mode} improved ({-self.best_score + self.delta:.6f} --> {-score:.6f})")
                else:
                    print(f"Validation {self.mode} improved ({self.best_score - self.delta:.6f} --> {score:.6f})")
            self.improved = True
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            self.improved = False
            if self.counter >= self.patience:
                self.early_stop = True
    
    def save_checkpoint(self, score, model):
        """
        保存最佳模型
        
        参数:
            score: 当前验证指标
            model: 当前模型
        """
        if self.verbose:
            if self.mode == 'min':
                print(f"Validation loss decreased ({-self.best_score:.6f} --> {-score:.6f}). Saving model...")
            else:
                print(f"Validation metric increased ({self.best_score:.6f} --> {score:.6f}). Saving model...")
        
        self.best_model = copy.deepcopy(model.state_dict())


def setup_seed(seed):
    """
    设置随机种子，确保结果可复现
    
    参数:
        seed: 随机种子
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"随机种子已设置为: {seed}")


def generate_cv_masks(events, n_splits=5, random_state=42):
    """
    生成交叉验证掩码，使用分层抽样确保每折中事件比例一致
    
    参数:
        events: 事件状态数组
        n_splits: 折数
        random_state: 随机种子
        
    返回:
        folds: 包含每个折的训练集和测试集掩码的列表
    """
    # 创建K折交叉验证（使用StratifiedKFold根据事件状态进行分层抽样）
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 准备索引数组
    indices = np.arange(len(events))
    
    # 生成每个折的训练集和测试集掩码
    folds = []
    for train_idx, test_idx in skf.split(indices, events):
        # 创建训练集和测试集掩码
        train_mask = torch.zeros(len(events), dtype=torch.bool)
        test_mask = torch.zeros(len(events), dtype=torch.bool)
        
        train_mask[train_idx] = True
        test_mask[test_idx] = True
        
        folds.append((train_mask, test_mask))
    
    return folds

def colored_text(text, color):
    """
    返回带颜色的终端文本
    """
    colors = {
        'red': '\033[91m',
        'green': '\033[92m',
        'yellow': '\033[93m',
        'blue': '\033[94m',
        'cyan': '\033[96m',
        'end': '\033[0m'
    }
    color_code = colors.get(color, colors['end'])
    return f"{color_code}{text}{colors['end']}"

def print_title(title, length=60, char='='):
    """
    打印一个居中的标题
    """
    print("\n" + f" {title} ".center(length, char) + "\n")
