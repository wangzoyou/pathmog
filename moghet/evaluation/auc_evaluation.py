"""
生存分析评估模块

实现针对癌症患者生存结果的多种评估指标，主要包括：
1. Harrell's C-index (一致性指数)
2. 时间依赖性AUC (Time-dependent AUC)

Harrell's C-index (代码中实现为 'c_index') 的计算公式大致为：
C-index = Σ_{i,j} I(T_i < T_j and E_i=1) * I(S_i > S_j) / Σ_{i,j} I(T_i < T_j and E_i=1)

其中：
- (i, j) 是所有可比较的患者对
- T: 生存时间, E: 事件状态 (1=事件, 0=删失)
- S: 模型预测的风险评分
- I(): 指示函数
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import roc_auc_score
import torch
from torch.utils.data import DataLoader


class SurvivalAUCEvaluator:
    """
    生存分析AUC评估器
    
    实现时间依赖性AUC计算，用于评估生存预测模型的性能
    """
    
    def __init__(self):
        """初始化AUC评估器"""
        pass
    
    def calculate_auc(self, 
                     risk_scores: np.ndarray, 
                     times: np.ndarray, 
                     events: np.ndarray,
                     method: str = 'time_dependent') -> float:
        """
        计算生存分析的AUC
        
        Args:
            risk_scores: 模型预测的风险评分 [n_samples]
            times: 生存时间 [n_samples]
            events: 事件发生指示 (1=event, 0=censored) [n_samples]
            method: 计算方法 ('time_dependent' 或 'standard')
            
        Returns:
            AUC值
        """
        if method == 'time_dependent':
            return self._calculate_c_index(risk_scores, times, events)
        elif method == 'standard':
            return self._calculate_standard_auc(risk_scores, times, events)
        else:
            raise ValueError(f"不支持的方法: {method}")
    
    def _calculate_c_index(self, 
                           risk_scores: np.ndarray, 
                           times: np.ndarray, 
                           events: np.ndarray) -> float:
        """
        计算 Harrell's C-index (一致性指数)
        
        对于每一对患者(i,j)，如果Ti < Tj且δi=1，那么我们期望risk_score_i > risk_score_j
        
        Args:
            risk_scores: 模型预测的风险评分
            times: 生存时间
            events: 事件发生指示
            
        Returns:
            时间依赖性AUC值
        """
        # 清理NaN值
        valid_mask = ~(np.isnan(risk_scores) | np.isnan(times) | np.isnan(events))
        if not np.any(valid_mask):
            print("警告: 所有数据都包含NaN，无法计算时间依赖性AUC")
            return 0.5
        
        clean_risk_scores = risk_scores[valid_mask]
        clean_times = times[valid_mask]
        clean_events = events[valid_mask]
        
        if len(clean_risk_scores) < 2:
            print("警告: 清理后数据不足，无法计算C-index")
            return 0.5
        
        n = len(clean_risk_scores)
            
        concordant_pairs = 0
        comparable_pairs = 0
        
        for i in range(n):
            for j in range(i + 1, n):
                # 只考虑可比较的患者对
                if self._is_comparable_pair(clean_times[i], clean_events[i], clean_times[j], clean_events[j]):
                    comparable_pairs += 1
                    
                    # 判断哪个患者应该有更高的风险评分
                    if self._should_have_higher_risk(clean_times[i], clean_events[i], clean_times[j], clean_events[j]):
                        # 患者i应该有更高的风险评分
                        if clean_risk_scores[i] > clean_risk_scores[j]:
                            concordant_pairs += 1
                        elif clean_risk_scores[i] == clean_risk_scores[j]:
                            concordant_pairs += 0.5  # 相等时算0.5
                    else:
                        # 患者j应该有更高的风险评分
                        if clean_risk_scores[j] > clean_risk_scores[i]:
                            concordant_pairs += 1
                        elif clean_risk_scores[i] == clean_risk_scores[j]:
                            concordant_pairs += 0.5
        
        if comparable_pairs == 0:
            return 0.5
            
        return concordant_pairs / comparable_pairs
    
    def _calculate_time_dependent_auc(self,
                                      risk_scores: np.ndarray,
                                      times: np.ndarray,
                                      events: np.ndarray) -> float:
        """
        计算时间依赖性AUC，通过在所有唯一事件时间点计算AUC并取平均值。
        这实现了类似于公式(15)的思想。

        Args:
            risk_scores: 模型预测的风险评分
            times: 生存时间
            events: 事件发生指示

        Returns:
            时间依赖性AUC值
        """
        # 获取所有唯一的事件时间点
        event_times = np.unique(times[events == 1])
        
        if len(event_times) == 0:
            return 0.5

        auc_values = []
        for t in event_times:
            # 使用现有帮助函数计算在时间点t的AUC
            auc_at_t, weight_at_t = self._calculate_auc_at_time(risk_scores, times, events, t)
            if weight_at_t > 0:  # 仅当该时间点有可比较的样本时才计入
                auc_values.append(auc_at_t)

        if not auc_values:
            return 0.5

        return np.mean(auc_values)

    def _is_comparable_pair(self, time_i: float, event_i: int, 
                           time_j: float, event_j: int) -> bool:
        """
        判断两个患者是否可比较
        
        可比较的条件：
        1. 一个患者观察到事件，另一个在事件发生时仍在风险集中
        2. 或者两个患者都观察到事件
        """
        # 如果其中一个患者观察到事件
        if event_i == 1 and time_i < time_j:
            return True
        if event_j == 1 and time_j < time_i:
            return True
        # 如果两个患者都观察到事件且时间不同
        if event_i == 1 and event_j == 1 and time_i != time_j:
            return True
            
        return False
    
    def _should_have_higher_risk(self, time_i: float, event_i: int,
                                time_j: float, event_j: int) -> bool:
        """
        判断患者i是否应该有更高的风险评分
        
        如果患者i更早发生事件，则应该有更高的风险评分
        """
        if event_i == 1 and event_j == 1:
            return time_i < time_j
        elif event_i == 1 and time_i < time_j:
            return True
        elif event_j == 1 and time_j < time_i:
            return False
        else:
            return False
    
    def _calculate_standard_auc(self, 
                               risk_scores: np.ndarray, 
                               times: np.ndarray, 
                               events: np.ndarray) -> float:
        """
        计算标准AUC (使用ROC曲线)
        
        Args:
            risk_scores: 模型预测的风险评分
            times: 生存时间
            events: 事件发生指示
            
        Returns:
            标准AUC值
        """
        try:
            # 清理NaN值
            valid_mask = ~(np.isnan(risk_scores) | np.isnan(times) | np.isnan(events))
            if not np.any(valid_mask):
                print("警告: 所有数据都包含NaN，无法计算标准AUC")
                return 0.5
            
            clean_risk_scores = risk_scores[valid_mask]
            clean_times = times[valid_mask]
            clean_events = events[valid_mask]
            
            if len(clean_risk_scores) == 0:
                print("警告: 清理后没有有效数据，无法计算标准AUC")
                return 0.5
            
            # 使用事件作为标签，风险评分作为预测概率
            auc = roc_auc_score(clean_events, clean_risk_scores)
            return auc
        except ValueError as e:
            print(f"计算标准AUC时出错: {e}")
            return 0.5
    
    def calculate_time_specific_auc(self, 
                                  risk_scores: np.ndarray, 
                                  times: np.ndarray, 
                                  events: np.ndarray,
                                  time_points: List[float]) -> Dict[float, float]:
        """
        计算特定时间点的AUC
        
        Args:
            risk_scores: 模型预测的风险评分
            times: 生存时间
            events: 事件发生指示
            time_points: 要计算AUC的时间点列表
            
        Returns:
            每个时间点的AUC字典
        """
        results = {}
        
        for t in time_points:
            # 创建时间t的二元标签
            # 在时间t之前发生事件的患者标记为1，否则为0
            binary_labels = (times <= t) & (events == 1)
            
            try:
                auc_at_t = roc_auc_score(binary_labels, risk_scores)
                results[t] = auc_at_t
            except ValueError:
                # 如果所有标签都相同，返回0.5
                results[t] = 0.5
        
        return results
    
    def evaluate_model_performance(self, 
                                 data_dict: Dict) -> Dict:
        """
        评估模型性能
        
        Args:
            data_dict: 包含预测结果和真实标签的字典
                必须包含: 'risk_scores', 'times', 'events'
                
        Returns:
            包含各种AUC指标的字典
        """
        risk_scores = data_dict['risk_scores']
        times = data_dict['times']
        events = data_dict['events']
        
        # 计算Harrell's C-index
        c_index = self._calculate_c_index(risk_scores, times, events)
        
        # 计算时间依赖性AUC (基于用户公式)
        time_dependent_auc = self._calculate_time_dependent_auc(risk_scores, times, events)
        
        # 计算标准AUC
        standard_auc = self._calculate_standard_auc(risk_scores, times, events)
        
        # 计算集成AUC
        integrated_auc = self.calculate_integrated_auc(risk_scores, times, events)
        
        # 计算特定时间点的AUC (1年, 3年, 5年)
        time_points = [1.0, 3.0, 5.0]  # 以年为单位
        time_specific_auc = self.calculate_time_specific_auc(risk_scores, times, events, time_points)
        
        results = {
            'c_index': c_index,
            'time_dependent_auc': time_dependent_auc,
            'standard_auc': standard_auc,
            'integrated_auc': integrated_auc,
            'time_specific_auc': time_specific_auc,
            'num_samples': len(risk_scores),
            'num_events': np.sum(events),
            'censoring_rate': 1 - np.mean(events)
        }
        
        return results

    def calculate_integrated_auc(self, 
                               risk_scores: np.ndarray,
                               times: np.ndarray, 
                               events: np.ndarray,
                               time_points: List[float] = None) -> float:
        """
        计算集成AUC (Integrated AUC)
        
        这是另一种常用的时间依赖性AUC计算方法，在多个时间点计算AUC并加权平均
        
        Args:
            risk_scores: 模型预测的风险评分
            times: 生存时间
            events: 事件发生指示
            time_points: 要计算AUC的时间点列表，如果为None则使用事件时间的四分位数
            
        Returns:
            集成AUC值
        """
        if time_points is None:
            # 使用所有事件时间的四分位数
            event_times = times[events == 1]
            if len(event_times) == 0:
                return 0.5
            time_points = np.percentile(event_times, [25, 50, 75])
        
        aucs = []
        weights = []
        
        for t in time_points:
            auc_t, weight_t = self._calculate_auc_at_time(risk_scores, times, events, t)
            if weight_t > 0:
                aucs.append(auc_t)
                weights.append(weight_t)
        
        if not aucs:
            return 0.5
            
        # 加权平均
        return np.average(aucs, weights=weights)
    
    def _calculate_auc_at_time(self, risk_scores: np.ndarray,
                              times: np.ndarray, events: np.ndarray,
                              time_point: float) -> Tuple[float, float]:
        """
        计算特定时间点的AUC
        
        Args:
            risk_scores: 模型预测的风险评分
            times: 生存时间
            events: 事件发生指示
            time_point: 要计算AUC的时间点
            
        Returns:
            (auc_value, weight) - AUC值和权重
        """
        # 创建时间点t的二元标签
        # 事件组：在时间t之前发生事件
        event_mask = (times <= time_point) & (events == 1)
        
        # 对照组：在时间t时仍在观察中（时间>t或时间=t且未发生事件）
        control_mask = times > time_point
        
        event_indices = np.where(event_mask)[0]
        control_indices = np.where(control_mask)[0]
        
        if len(event_indices) == 0 or len(control_indices) == 0:
            return 0.5, 0.0
        
        # 计算AUC
        y_true = np.concatenate([np.ones(len(event_indices)), np.zeros(len(control_indices))])
        y_scores = np.concatenate([risk_scores[event_indices], risk_scores[control_indices]])
        
        try:
            auc = roc_auc_score(y_true, y_scores)
        except ValueError:
            auc = 0.5
        
        # 权重为事件数量
        weight = len(event_indices)
        
        return auc, weight


def calculate_auc_from_predictions(model, 
                                 data_loader: DataLoader, 
                                 device: str = 'auto') -> Dict:
    """
    从模型预测计算AUC
    
    Args:
        model: 训练好的模型
        data_loader: 数据加载器
        device: 计算设备
        
    Returns:
        包含AUC评估结果的字典
    """
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    model.eval()
    all_risk_scores = []
    all_times = []
    all_events = []
    
    with torch.no_grad():
        for batch_data in data_loader:
            if batch_data is None:
                continue
                
            # 获取数据
            graphs_batch = batch_data['graphs_batch'].to(device)
            clinical_features = batch_data['clinical_features'].to(device)
            
            # 生存数据保留在CPU上
            time = batch_data['time']
            event = batch_data['event']
            
            # 模型预测
            risk_scores = model(graphs_batch, clinical_features)
            
            all_risk_scores.append(risk_scores.cpu().numpy())
            all_times.append(time.cpu().numpy())
            all_events.append(event.cpu().numpy())
    
    if not all_risk_scores:
        print("警告: 没有生成任何预测，无法计算AUC")
        return {'auc': 0.5, 'error': 'No predictions generated'}
    
    # 合并所有批次的结果
    risk_scores = np.concatenate(all_risk_scores).squeeze()
    times = np.concatenate(all_times).squeeze()
    events = np.concatenate(all_events).squeeze()
    
    # 计算AUC
    evaluator = SurvivalAUCEvaluator()
    results = evaluator.evaluate_model_performance({
        'risk_scores': risk_scores,
        'times': times,
        'events': events
    })
    
    # 添加主要AUC指标
    results['auc'] = results['time_dependent_auc']  # 主要指标，使用时间依赖性AUC
    
    return results


def calculate_cross_validation_auc(model_class, 
                                 train_data, 
                                 val_data, 
                                 test_data,
                                 n_folds: int = 5,
                                 **model_kwargs) -> Dict:
    """
    计算交叉验证的AUC
    
    Args:
        model_class: 模型类
        train_data: 训练数据
        val_data: 验证数据
        test_data: 测试数据
        n_folds: 交叉验证折数
        **model_kwargs: 模型参数
        
    Returns:
        交叉验证AUC结果
    """
    from sklearn.model_selection import KFold
    
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    fold_results = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_data)):
        print(f"训练第 {fold + 1}/{n_folds} 折...")
        
        # 训练模型
        model = model_class(**model_kwargs)
        # ... 训练过程 ...
        
        # 评估AUC
        evaluator = SurvivalAUCEvaluator()
        fold_auc = evaluator.calculate_auc(
            risk_scores=model.predict(test_data),
            times=test_data['times'],
            events=test_data['events']
        )
        
        fold_results.append({
            'fold': fold + 1,
            'auc': fold_auc
        })
    
    # 计算统计信息
    aucs = [r['auc'] for r in fold_results]
    results = {
        'fold_results': fold_results,
        'mean_auc': np.mean(aucs),
        'std_auc': np.std(aucs),
        'min_auc': np.min(aucs),
        'max_auc': np.max(aucs)
    }
    
    return results 