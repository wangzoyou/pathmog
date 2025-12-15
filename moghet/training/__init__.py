"""
模型训练模块

包含训练脚本、训练引擎和损失函数：
- train.py: 统一训练脚本
- engine.py: 训练和验证引擎
- loss.py: Cox损失等损失函数定义
"""

from .engine import train_one_epoch, evaluate
from .loss import cox_loss

__all__ = [
    'train_one_epoch',
    'evaluate',
    'cox_loss',
]

