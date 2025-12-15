#!/usr/bin/env python3
"""
汇总患者级模块消融结果，生成：
1) 模块总体重要性排名（均值/中位数/95%置信区间）
2) 每个模块在不同患者子组（如PAM50/Stage）上的差异
3) 输出Markdown报告和CSV统计表
"""

import os
import os.path as osp
import argparse
import pandas as pd
import numpy as np
from scipy import stats


def mean_ci(a, alpha=0.05):
    a = np.array(a, dtype=float)
    a = a[~np.isnan(a)]
    if a.size == 0:
        return np.nan, (np.nan, np.nan)
    m = a.mean()
    se = stats.sem(a) if a.size > 1 else 0.0
    h = se * stats.t.ppf(1 - alpha/2, a.size - 1) if a.size > 1 else 0.0
    return m, (m - h, m + h)


def subgroup_compare(df, module_cols, clinical_df):
    results = {}
    merged = df.merge(clinical_df, left_on='patient_id', right_index=True, how='left')
    # 典型BRCA分组：PAM50, pathologic_stage
    subgroups = []
    if 'PAM50' in merged.columns:
        subgroups.append(('PAM50', merged['PAM50']))
    if 'pathologic_stage' in merged.columns:
        subgroups.append(('pathologic_stage', merged['pathologic_stage']))
    for name, series in subgroups:
        res = []
        for col in module_cols:
            grp = merged.groupby(series)[col].mean()
            res.append({'module': col, 'group_var': name, 'group_means': grp.to_dict()})
        results[name] = res
    return results


def main():
    parser = argparse.ArgumentParser(description='汇总模块消融结果并生成报告')
    parser.add_argument('--dataset', type=str, default='BRCA')
    parser.add_argument('--fold', type=int, default=1)
    args = parser.parse_args()

    scripts_dir = osp.dirname(osp.abspath(__file__))
    moghet_root = osp.dirname(scripts_dir)
    processed_dir = osp.join(moghet_root, 'data', 'processed', args.dataset)
    res_dir = osp.join(moghet_root, 'results', 'explanation_statistics')
    csv_path = osp.join(res_dir, f'module_importance_{args.dataset.lower()}_fold{args.fold}.csv')
    assert osp.exists(csv_path), f'找不到结果CSV: {csv_path}（请先运行患者级模块消融）'

    df = pd.read_csv(csv_path)
    module_cols = [c for c in df.columns if c not in ['patient_id', 'baseline_risk']]

    # 1) 模块总体重要性统计
    rows = []
    for col in module_cols:
        m, (lo, hi) = mean_ci(df[col].values)
        rows.append({
            'module': col,
            'mean_importance': m,
            'median_importance': float(np.nanmedian(df[col].values)),
            'ci95_low': lo,
            'ci95_high': hi,
            'non_nan_n': int(df[col].notna().sum())
        })
    overall_df = pd.DataFrame(rows).sort_values('mean_importance', ascending=False)

    # 2) 子组差异（若有可用临床列）
    clinical_csv = osp.join(processed_dir, 'patient_clinical_features.csv')
    subgroup_results = {}
    if osp.exists(clinical_csv):
        clinical_df = pd.read_csv(clinical_csv, index_col=0)
        subgroup_results = subgroup_compare(df[['patient_id'] + module_cols], module_cols, clinical_df)

    # 3) 保存统计与报告
    os.makedirs(res_dir, exist_ok=True)
    overall_csv = osp.join(res_dir, f'module_importance_overall_{args.dataset.lower()}_fold{args.fold}.csv')
    overall_df.to_csv(overall_csv, index=False)

    md_path = osp.join(res_dir, f'module_importance_report_{args.dataset.lower()}_fold{args.fold}.md')
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(f"# 模块消融总体报告（{args.dataset}，fold {args.fold}）\n\n")
        f.write("## 总体排名（均值±95%CI）\n\n")
        for _, r in overall_df.iterrows():
            f.write(f"- {r['module']}: {r['mean_importance']:.4f} (95%CI {r['ci95_low']:.4f}-{r['ci95_high']:.4f}), 中位数 {r['median_importance']:.4f}, n={int(r['non_nan_n'])}\n")
        if subgroup_results:
            f.write("\n## 子组差异（均值）\n")
            for name, res in subgroup_results.items():
                f.write(f"\n### 分组变量：{name}\n")
                for item in res:
                    means = ", ".join([f"{k}:{v:.4f}" for k,v in item['group_means'].items()])
                    f.write(f"- {item['module']}: {means}\n")
    print(f"总体统计保存到: {overall_csv}")
    print(f"报告保存到: {md_path}")

if __name__ == '__main__':
    main()


