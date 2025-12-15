import os
import os.path as osp
import sys
import argparse
import pandas as pd
import numpy as np

# -- 项目路径设置 --
current_dir = osp.dirname(osp.abspath(__file__))
project_root = osp.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)
# -- 完成路径设置 --

from evaluation.auc_evaluation import SurvivalAUCEvaluator

def main():
    parser = argparse.ArgumentParser(description='从预测CSV文件和生存数据计算AUC指标')
    parser.add_argument('--dataset', type=str, required=True, help='数据集名称 (例如 KIRC)')
    parser.add_argument('--results_dir', type=str, default=osp.join(project_root, "moghet", "results"),
                        help='包含模型结果的根目录')
    parser.add_argument('--data_root_dir', type=str, default=osp.join(project_root, "moghet", "data", "processed"),
                        help='处理后数据的根目录')
    parser.add_argument('--model_type', type=str, default='fast', help='模型类型，用于定位结果文件夹 (例如 fast 或 smoke_test)')
    
    args = parser.parse_args()
    
    print(f"--- 开始从CSV文件为数据集 {args.dataset} 计算AUC ---")

    # --- 1. 构建路径 ---
    predictions_path = osp.join(args.results_dir, f"hierarchical_model_{args.dataset.lower()}_{args.model_type}", "all_patients_predictions.csv")
    survival_path = osp.join(args.data_root_dir, args.dataset, "patient_survival.csv")
    
    # --- 2. 检查文件是否存在 ---
    if not osp.exists(predictions_path):
        print(f"错误: 预测文件不存在: {predictions_path}")
        sys.exit(1)
    if not osp.exists(survival_path):
        print(f"错误: 生存数据文件不存在: {survival_path}")
        sys.exit(1)

    # --- 3. 加载数据 ---
    print("正在加载预测数据和生存数据...")
    try:
        predictions_df = pd.read_csv(predictions_path)
        survival_df = pd.read_csv(survival_path)
        # 重命名生存数据的第一列为 'patient_id' 以便合并
        survival_df.rename(columns={survival_df.columns[0]: 'patient_id'}, inplace=True)
    except Exception as e:
        print(f"加载文件时出错: {e}")
        sys.exit(1)
        
    print(f"成功加载 {len(predictions_df)} 条预测记录和 {len(survival_df)} 条生存记录。")

    # --- 4. 合并数据 ---
    print("正在根据 patient_id 合并数据...")
    merged_df = pd.merge(predictions_df, survival_df, on='patient_id', how='inner')
    
    print(f"成功合并 {len(merged_df)} 条记录。")
    if len(merged_df) == 0:
        print("错误: 预测数据和生存数据没有共同的患者ID。")
        sys.exit(1)
    
    # --- 4.5. 筛选测试集数据 ---
    if 'source' in merged_df.columns:
        test_df = merged_df[merged_df['source'] == 'holdout_test_set']
        print(f"筛选出测试集数据: {len(test_df)} 条记录")
        if len(test_df) == 0:
            print("错误: 没有找到测试集数据 (source='holdout_test_set')")
            sys.exit(1)
        merged_df = test_df
    else:
        print("警告: 预测文件中没有 'source' 列，将使用所有数据")

    # --- 5. 准备评估数据 ---
    # 将OS.time从天转换为年，以匹配模型训练时的处理方式
    risk_scores = merged_df['risk_score'].to_numpy()
    times = merged_df['OS.time'].to_numpy() / 365.0
    events = merged_df['OS'].to_numpy()
    
    # 检查数据中是否存在NaN
    if np.isnan(risk_scores).any() or np.isnan(times).any() or np.isnan(events).any():
        print("警告: 合并后的数据中检测到NaN值，将进行清理。")
        valid_mask = ~(np.isnan(risk_scores) | np.isnan(times) | np.isnan(events))
        risk_scores = risk_scores[valid_mask]
        times = times[valid_mask]
        events = events[valid_mask]
        print(f"清理后剩余 {len(risk_scores)} 条有效记录。")

    # --- 6. 执行评估 ---
    print("\n--- 开始评估 ---")
    evaluator = SurvivalAUCEvaluator()
    results = evaluator.evaluate_model_performance({
        'risk_scores': risk_scores,
        'times': times,
        'events': events
    })
    
    print("\n--- 评估结果 ---")
    print(f"C-Index (一致性指数): {results.get('c_index', 'N/A'):.4f}")
    print(f"时间依赖性 AUC (基于公式15): {results.get('time_dependent_auc', 'N/A'):.4f}")
    print(f"标准 AUC (ROC): {results.get('standard_auc', 'N/A'):.4f}")
    print(f"集成 AUC: {results.get('integrated_auc', 'N/A'):.4f}")
    print("\n各时间点AUC:")
    for t, auc in results.get('time_specific_auc', {}).items():
        print(f"  - {t} 年: {auc:.4f}")
    print("\n------------------")
    print(f"在 {results.get('num_samples', 0)} 个样本上评估, 其中包含 {results.get('num_events', 0)} 个事件。")
    print(f"删失率 (Censoring Rate): {results.get('censoring_rate', 'N/A'):.2%}")

if __name__ == "__main__":
    main()
