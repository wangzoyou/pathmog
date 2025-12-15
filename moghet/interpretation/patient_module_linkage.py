#!/usr/bin/env python3
"""
患者-模块联动分析：结合患者级模块重要性与临床分组，输出“哪些患者在哪个模块最敏感”的可视化。

数据依赖：
- 患者级模块重要性CSV（由 brca_patient_module_ablation.py 生成）：
  moghet/results/explanation_statistics/module_importance_<dataset>_fold<k>.csv
- 临床特征（简化版）：
  moghet/data/processed/<dataset>/patient_clinical_features.csv

输出：
- linkage 目录下热图、每模块Top-N患者条形图、子组分布箱线图
"""

import os
import os.path as osp
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_module_importance(project_root: str, dataset: str, fold: int) -> pd.DataFrame:
    # project_root 指向 moghet 目录
    imp_path = osp.join(project_root, 'results', 'explanation_statistics',
                        f'module_importance_{dataset.lower()}_fold{fold}.csv')
    if not osp.exists(imp_path):
        raise FileNotFoundError(f'找不到模块重要性文件: {imp_path}')
    df = pd.read_csv(imp_path)
    # 规范列名
    if 'patient_id' not in df.columns:
        # 尝试兼容索引型
        df.rename(columns={df.columns[0]: 'patient_id'}, inplace=True)
    return df


def load_clinical(project_root: str, dataset: str) -> pd.DataFrame:
    clin_path = osp.join(project_root, 'data', 'processed', dataset, 'patient_clinical_features.csv')
    if not osp.exists(clin_path):
        raise FileNotFoundError(f'找不到临床特征文件: {clin_path}')
    cdf = pd.read_csv(clin_path, index_col=0)
    return cdf


def ensure_output_dir(project_root: str, dataset: str) -> str:
    out_dir = osp.join(project_root, 'results', 'explanation_statistics', f'linkage_{dataset.lower()}')
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def plot_heatmap(df_imp: pd.DataFrame, out_dir: str, top_patients: int = 100):
    # 选择模块列
    module_cols = [c for c in df_imp.columns if c not in ['patient_id', 'baseline_risk']]
    # 选Top患者（按总重要性和排序）
    df_imp['_sum'] = df_imp[module_cols].sum(axis=1)
    df_top = df_imp.sort_values('_sum', ascending=False).head(top_patients)
    mat = df_top.set_index('patient_id')[module_cols]
    plt.figure(figsize=(12, max(6, top_patients * 0.08)))
    sns.heatmap(mat, cmap='viridis')
    plt.title(f'Top-{top_patients} Patients x Module Importance Heatmap')
    plt.xlabel('Module')
    plt.ylabel('Patient')
    plt.tight_layout()
    path = osp.join(out_dir, f'heatmap_top{top_patients}.png')
    plt.savefig(path, dpi=200, bbox_inches='tight')
    plt.close()
    return path


def plot_module_top_patients(df_imp: pd.DataFrame, out_dir: str, top_k: int = 20):
    module_cols = [c for c in df_imp.columns if c not in ['patient_id', 'baseline_risk']]
    saved = []
    for m in module_cols:
        sub = df_imp[['patient_id', m]].sort_values(m, ascending=False).head(top_k)
        plt.figure(figsize=(10, max(4, top_k * 0.3)))
        sns.barplot(data=sub, x=m, y='patient_id', orient='h')
        plt.title(f'Top-{top_k} Patients by Module: {m}')
        plt.xlabel('Importance (|Δrisk|)')
        plt.ylabel('Patient')
        plt.tight_layout()
        p = osp.join(out_dir, f'top_patients_{m}_top{top_k}.png')
        plt.savefig(p, dpi=200, bbox_inches='tight')
        plt.close()
        saved.append(p)
    return saved


def plot_subgroup_distribution(df_imp: pd.DataFrame, df_clin: pd.DataFrame, out_dir: str, subgroup_col: str = 'PAM50'):
    # 合并
    df = df_imp.merge(df_clin, left_on='patient_id', right_index=True, how='left')
    module_cols = [c for c in df_imp.columns if c not in ['patient_id', 'baseline_risk']]
    # PAM50 数值编码 -> 名称映射（按数据准备逻辑）
    pam50_map = {1: 'LumA', 2: 'LumB', 3: 'Her2', 4: 'Basal', 5: 'Normal'}
    if subgroup_col in df.columns:
        # 仅当列是数值编码时使用映射
        if pd.api.types.is_numeric_dtype(df[subgroup_col]):
            df[subgroup_col] = df[subgroup_col].map(pam50_map).fillna(df[subgroup_col])
        # 为每个模块绘制小提琴+箱线
        saved = []
        for m in module_cols:
            plt.figure(figsize=(8, 5))
            sns.violinplot(data=df, x=subgroup_col, y=m, inner='box', cut=0)
            plt.title(f'Module {m} Importance by {subgroup_col}')
            plt.xlabel(subgroup_col)
            plt.ylabel('Importance (|Δrisk|)')
            plt.tight_layout()
            p = osp.join(out_dir, f'{m}_by_{subgroup_col}.png')
            plt.savefig(p, dpi=200, bbox_inches='tight')
            plt.close()
            saved.append(p)
        return saved
    return []


def main():
    parser = argparse.ArgumentParser(description='患者-模块联动分析与可视化')
    parser.add_argument('--dataset', type=str, default='BRCA')
    parser.add_argument('--fold', type=int, default=1)
    parser.add_argument('--top_patients', type=int, default=100)
    parser.add_argument('--top_k', type=int, default=20)
    args = parser.parse_args()

    # 项目根（上一层）
    current = osp.dirname(osp.abspath(__file__))
    project_root = osp.dirname(current)

    # 加载数据
    df_imp = load_module_importance(project_root, args.dataset, args.fold)
    df_clin = load_clinical(project_root, args.dataset)

    out_dir = ensure_output_dir(project_root, args.dataset)

    # 1) 热图
    heatmap_path = plot_heatmap(df_imp, out_dir, top_patients=args.top_patients)

    # 2) 各模块Top-K患者榜单
    tops = plot_module_top_patients(df_imp, out_dir, top_k=args.top_k)

    # 3) 子组分布（PAM50，如存在）
    subgroup_plots = plot_subgroup_distribution(df_imp, df_clin, out_dir, subgroup_col='PAM50')

    # 保存一个清单
    manifest = {
        'heatmap': heatmap_path,
        'top_patients_plots': tops,
        'subgroup_plots': subgroup_plots
    }
    import json
    with open(osp.join(out_dir, 'linkage_manifest.json'), 'w') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    print('生成完成，输出目录:', out_dir)


if __name__ == '__main__':
    main()

