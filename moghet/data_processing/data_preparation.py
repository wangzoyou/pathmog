#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
数据准备脚本 - 整合并优化了原有的kegg_parser.py和process_clinical_data.py

此脚本的目标是读取所有原始数据，并生成一系列干净、ID对齐的CSV表格文件，
为后续构建异构图做准备。脚本直接处理数据并生成表格，避免了不必要的中间图构建过程。

作者：AI Assistant
日期：2024-07-15
"""

import os
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
import logging
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import json
import argparse  # (新增) 导入argparse用于命令行参数处理

# (新增) 数据集配置 - 扩展支持更多癌症类型
DATASET_CONFIGS = {
    "BRCA": {
        "expression_file": "HiSeqV2",
        "mutation_file": "BRCA_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.BRCA.sampleMap_BRCA_clinicalMatrix"
    },
    "LUAD": {
        "expression_file": "HiSeqV2",
        "mutation_file": "LUAD_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.LUAD.sampleMap_LUAD_clinicalMatrix"
    },
    "COAD": {  # Colon Adenocarcinoma
        "expression_file": "HiSeqV2",
        "mutation_file": "COAD_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_data_by_genes",
        "clinical_file": "TCGA.COAD.sampleMap_COAD_clinicalMatrix"
    },
    "GBM": {  # Glioblastoma Multiforme
        "expression_file": "HiSeqV2",
        "mutation_file": "GBM_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.GBM.sampleMap_GBM_clinicalMatrix"
    },
    "KIRC": {  # Kidney Renal Clear Cell Carcinoma
        "expression_file": "HiSeqV2",
        "mutation_file": "KIRC_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.KIRC.sampleMap_KIRC_clinicalMatrix"
    },
    "LUNG": {  # Lung Cancer (combined)
        "expression_file": "HiSeqV2",
        "mutation_file": "LUNG_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.LUNG.sampleMap_LUNG_clinicalMatrix"
    },
    "OV": {  # Ovarian Serous Cystadenocarcinoma
        "expression_file": "HiSeqV2",
        "mutation_file": "OV_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.OV.sampleMap_OV_clinicalMatrix"
    },
    "SKCM": {  # Skin Cutaneous Melanoma
        "expression_file": "HiSeqV2",
        "mutation_file": "SKCM_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.SKCM.sampleMap_SKCM_clinicalMatrix"
    },
    "LIHC": { # Liver Hepatocellular Carcinoma
        "expression_file": "HiSeqV2",
        "mutation_file": "LIHC_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.LIHC.sampleMap_LIHC_clinicalMatrix"
    },
    # 新增支持的癌症类型
    "LUSC": { # Lung Squamous Cell Carcinoma
        "expression_file": "HiSeqV2",
        "mutation_file": "LUSC_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.LUSC.sampleMap_LUSC_clinicalMatrix"
    },
    "STAD": { # Stomach Adenocarcinoma
        "expression_file": "HiSeqV2",
        "mutation_file": "STAD_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.STAD.sampleMap_STAD_clinicalMatrix"
    },
    "UCEC": { # Uterine Corpus Endometrial Carcinoma
        "expression_file": "HiSeqV2",
        "mutation_file": "UCEC_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.UCEC.sampleMap_UCEC_clinicalMatrix"
    },
    "HNSC": { # Head and Neck Squamous Cell Carcinoma
        "expression_file": "HiSeqV2",
        "mutation_file": "HNSC_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.HNSC.sampleMap_HNSC_clinicalMatrix"
    },
    "PAAD": { # Pancreatic Adenocarcinoma
        "expression_file": "HiSeqV2",
        "mutation_file": "PAAD_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.PAAD.sampleMap_PAAD_clinicalMatrix"
    },
    "LGG": { # Brain Lower Grade Glioma
        "expression_file": "HiSeqV2",
        "mutation_file": "LGG_mc3_gene_level.txt",
        "cnv_file": "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes",
        "clinical_file": "TCGA.LGG.sampleMap_LGG_clinicalMatrix"
    }
}

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DataPreparation:
    """数据准备类，用于处理和生成异构图所需的所有表格数据"""
    
    def __init__(self, dataset_name, raw_data_base_dir=None, processed_data_base_dir=None):
        """
        初始化数据路径和存储容器
        (修改) 增加了 dataset_name 参数，用于指定要处理的数据集。
        """
        if dataset_name not in DATASET_CONFIGS:
            raise ValueError(f"错误: 数据集 '{dataset_name}' 的配置未找到。请在 DATASET_CONFIGS 中添加它。")
        
        self.dataset_name = dataset_name
        self.config = DATASET_CONFIGS[dataset_name]
        logger.info(f"正在为数据集 '{self.dataset_name}' 初始化数据准备流程...")

        # 设置数据路径
        self.root_dir = Path(__file__).parent.parent.absolute()
        
        # (修改) 基础路径
        raw_base_dir = raw_data_base_dir or (self.root_dir / "data" / "raw")
        processed_base_dir = processed_data_base_dir or (self.root_dir / "data" / "processed")
        
        # (修改) 根据数据集名称构建特定的数据路径
        # 处理文件夹名称映射 - 扩展支持新癌症类型的目录映射
        folder_mapping = {
            "COAD": "Colon",
            "KIRC": "Kidney", 
            "OV": "Ovarian",
            "SKCM": "SKCM",
            "LIHC": "liver",  # 注意新数据集中是小写
            "LUSC": "LUSC",
            "STAD": "STAD", 
            "UCEC": "UCEC",
            "HNSC": "HNSC",
            "PAAD": "PAAD",
            "LGG": "LGG"
        }
        actual_folder_name = folder_mapping.get(self.dataset_name, self.dataset_name)
        self.raw_data_dir = raw_base_dir / actual_folder_name
        self.processed_data_dir = processed_base_dir / self.dataset_name
        
        # 确保处理后的数据目录存在
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        # (修改) KEGG通路路径保持共享
        self.kegg_data_dir = raw_base_dir / "kegg_data"
        
        # 存储数据的容器
        self.pathway_info = {}  # 通路信息
        self.gene_info = {}     # 基因信息
        self.gene_gene_relations = []  # 基因-基因关系
        self.gene_pathway_relations = []  # 基因-通路关系
        
        # 存储患者数据
        self.expression_data = None
        self.mutation_data = None
        self.cnv_data = None
        self.clinical_data = None
        self.survival_data = None
        
        # 统一后的ID列表
        self.common_genes = None
        self.common_patients = None
        
        # 添加基因ID到符号的映射
        self.gene_id_to_symbol = {}  # Entrez ID到基因符号的映射
        self.gene_symbol_to_id = {}  # 基因符号到Entrez ID的映射

    def run(self):
        """执行完整的数据准备流程"""
        logger.info("开始执行数据准备流程...")
        
        # 1. 解析KEGG通路数据
        self.parse_kegg_pathways()
        
        # 2. 加载患者组学数据
        self.load_patient_data()
        
        # 3. 处理临床数据
        self.process_clinical_data()
        
        # 4. 保存所有处理后的数据
        self.save_processed_data()
        
        logger.info("数据准备流程完成。所有处理后的文件已保存到: %s", self.processed_data_dir)
        
        # 返回处理后数据的路径，便于后续操作
        return self.processed_data_dir
    
    def parse_kegg_pathways(self):
        """解析KEGG通路XML文件，提取基因和通路信息以及它们之间的关系"""
        logger.info("开始解析KEGG通路数据...")
        
        # 读取人类通路列表
        pathway_list_file = self.kegg_data_dir / "human_pathways.txt"
        if pathway_list_file.exists():
            with open(pathway_list_file, 'r') as f:
                pathway_lines = f.readlines()
                
            for line in pathway_lines:
                if line.strip():
                    parts = line.strip().split('\t')
                    pathway_id = parts[0]
                    pathway_name = parts[1] if len(parts) > 1 else ""
                    self.pathway_info[pathway_id] = {
                        "id": pathway_id,
                        "name": pathway_name
                    }
        
        # 加载基因ID和符号的映射关系
        self._load_gene_id_symbol_mapping()
        
        # 直接从基因文件中读取基因-通路关系
        self._load_gene_pathway_relations()
        
        # 遍历KEGG文件夹，解析XML文件获取基因-基因关系
        for xml_file in self.kegg_data_dir.glob("*.xml"):
            pathway_id = xml_file.stem  # 获取不带扩展名的文件名
            
            if pathway_id not in self.pathway_info:
                self.pathway_info[pathway_id] = {"id": pathway_id, "name": pathway_id}
            
            # 解析XML文件
            try:
                self._parse_kgml_file(xml_file, pathway_id)
            except Exception as e:
                logger.error(f"解析通路 {pathway_id} 失败: {e}")
        
        logger.info(f"成功解析了 {len(self.pathway_info)} 个KEGG通路")
        logger.info(f"找到 {len(self.gene_info)} 个基因")
        logger.info(f"找到 {len(self.gene_gene_relations)} 条基因-基因关系")
        logger.info(f"找到 {len(self.gene_pathway_relations)} 条基因-通路关系")

    def _load_gene_pathway_relations(self):
        """从hsa*_genes.txt文件加载基因-通路关系，只保留基因符号为gene_id"""
        logger.info("从基因文件加载基因-通路关系...")
        for gene_file in self.kegg_data_dir.glob("hsa*_genes.txt"):
            try:
                pathway_id = gene_file.stem.replace('_genes', '')
                with open(gene_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        entrez_id = parts[0]
                        gene_symbol = parts[1]
                        # 只保留符号为gene_id
                        self.gene_pathway_relations.append({
                            "gene_id": gene_symbol,
                            "pathway_id": pathway_id,
                            "relation_type": "is_member_of"
                        })
                        # 保存基因信息（只用符号）
                        if gene_symbol not in self.gene_info:
                            self.gene_info[gene_symbol] = {
                                "id": gene_symbol,
                                "name": gene_symbol,
                                "symbol": gene_symbol
                            }
            except Exception as e:
                logger.error(f"处理基因-通路关系文件 {gene_file} 时出错: {e}")

    def _parse_kgml_file(self, xml_file, pathway_id):
        """解析单个KGML文件，提取基因和关系信息"""
        tree = ET.parse(xml_file)
        root = tree.getroot()
        
        # 从XML中提取基因-基因关系
        gene_entries = {}
        for entry in root.findall(".//entry"):
            entry_id = entry.get("id")
            entry_type = entry.get("type")
            
            # 只处理基因类型的条目
            if entry_type == "gene":
                # 提取并清洗基因符号
                clean_gene_symbols = set()
                for graphics in entry.findall(".//graphics"):
                    name = graphics.get("name")
                    if name:
                        # 清洗逻辑：取第一个逗号/分号/空格前的部分，并去除"..."
                        symbol = name.split(',')[0].split(';')[0].split(' ')[0].replace('...', '')
                        if symbol:
                            clean_gene_symbols.add(symbol)
                
                if clean_gene_symbols:
                    gene_entries[entry_id] = list(clean_gene_symbols)
        
        # 从XML中提取基因间关系
        for relation in root.findall(".//relation"):
            entry1 = relation.get("entry1")
            entry2 = relation.get("entry2")
            
            if entry1 in gene_entries and entry2 in gene_entries:
                rel_type = "interacts_with"  # 默认关系类型
                
                # 尝试获取更具体的关系类型
                for subtype in relation.findall(".//subtype"):
                    rel_name = subtype.get("name")
                    if rel_name:
                        rel_type = rel_name
                        break
                
                # 为每对清洗后的基因符号添加关系
                for gene1_symbol in gene_entries[entry1]:
                    for gene2_symbol in gene_entries[entry2]:
                        # 确保两个基因不相同
                        if gene1_symbol != gene2_symbol:
                            self.gene_gene_relations.append({
                                "source_gene": gene1_symbol,
                                "target_gene": gene2_symbol,
                                "relation_type": rel_type
                            })
    
    def load_patient_data(self):
        """加载患者组学数据（表达谱、突变、CNV）"""
        logger.info("开始加载患者组学数据...")
        
        # (修改) 使用配置文件中的文件名加载基因表达数据
        expression_file = self.raw_data_dir / self.config["expression_file"]
        if expression_file.exists():
            logger.info("加载基因表达数据...")
            # 行为基因，列为样本，需要转置为行为样本，列为基因
            self.expression_data = pd.read_csv(expression_file, sep='\t', index_col=0)
            logger.info(f"原始表达数据形状: {self.expression_data.shape}，行为基因，列为样本")
            
            # 转置数据，使得行为样本，列为基因
            self.expression_data = self.expression_data.transpose()
            logger.info(f"转置后表达数据形状: {self.expression_data.shape}，行为样本，列为基因")
            
            # 标准化TCGA样本ID (格式为TCGA-XX-XXXX-XX)
            original_index = self.expression_data.index.tolist()
            self.expression_data.index = [self._standardize_patient_id(pid) for pid in self.expression_data.index]
            logger.info(f"样本ID示例 - 原始: {original_index[:3]} → 标准化后: {self.expression_data.index[:3]}")
            
            logger.info(f"表达数据形状: {self.expression_data.shape}")
        else:
            logger.warning(f"未找到表达数据文件: {expression_file}")
        
        # (修改) 使用配置文件中的文件名加载基因突变数据
        mutation_file = self.raw_data_dir / self.config["mutation_file"]
        if mutation_file.exists():
            logger.info("加载基因突变数据...")
            try:
                # 直接加载数据，行为基因，列为样本
                self.mutation_data = pd.read_csv(mutation_file, sep='\t', index_col=0)
                logger.info(f"原始突变数据形状: {self.mutation_data.shape}，行为基因，列为样本")
                
                # 转置数据，使得行为样本，列为基因
                self.mutation_data = self.mutation_data.transpose()
                logger.info(f"转置后突变数据形状: {self.mutation_data.shape}，行为样本，列为基因")
                
                # 确保数据是二进制格式 (0/1)
                self.mutation_data = self.mutation_data.astype(int)
                
                # 标准化TCGA样本ID
                original_index = self.mutation_data.index.tolist()
                self.mutation_data.index = [self._standardize_patient_id(pid) for pid in self.mutation_data.index]
                logger.info(f"样本ID示例 - 原始: {original_index[:3]} → 标准化后: {self.mutation_data.index[:3]}")
                
                logger.info(f"处理后的突变数据形状: {self.mutation_data.shape}")
            except Exception as e:
                logger.error(f"处理突变数据时出错: {e}")
                self.mutation_data = None
        else:
            logger.warning(f"未找到突变数据文件: {mutation_file}")
        
        # (修改) 使用配置文件中的文件名加载CNV数据
        cnv_file = self.raw_data_dir / self.config["cnv_file"]
        if cnv_file.exists():
            logger.info("加载CNV数据...")
            # 行为基因，列为样本
            self.cnv_data = pd.read_csv(cnv_file, sep='\t', index_col=0)
            logger.info(f"原始CNV数据形状: {self.cnv_data.shape}，行为基因，列为样本")
            
            # 转置数据，使得行为样本，列为基因
            self.cnv_data = self.cnv_data.transpose()
            logger.info(f"转置后CNV数据形状: {self.cnv_data.shape}，行为样本，列为基因")
            
            # 标准化TCGA样本ID
            original_index = self.cnv_data.index.tolist()
            self.cnv_data.index = [self._standardize_patient_id(pid) for pid in self.cnv_data.index]
            logger.info(f"样本ID示例 - 原始: {original_index[:3]} → 标准化后: {self.cnv_data.index[:3]}")
            
            logger.info(f"CNV数据形状: {self.cnv_data.shape}")
        else:
            logger.warning(f"未找到CNV数据文件: {cnv_file}")
        
        # 找出共有的基因和患者ID
        self._align_data_ids()
    
    def _standardize_patient_id(self, patient_id):
        """标准化TCGA患者ID格式，确保为TCGA-XX-XXXX格式"""
        if not isinstance(patient_id, str):
            return patient_id
            
        # 移除可能的版本号后缀
        if '.' in patient_id:
            patient_id = patient_id.split('.')[0]
        
        # 确保ID以TCGA-开头
        if not patient_id.startswith('TCGA-'):
            if patient_id.startswith('TCGA_'):
                patient_id = 'TCGA-' + patient_id[5:]
            elif not patient_id.startswith('TCGA'):
                patient_id = 'TCGA-' + patient_id
        
        # 提取TCGA-XX-XXXX部分 (前三段)
        if patient_id.startswith('TCGA-'):
            parts = patient_id.split('-')
            if len(parts) >= 3:
                return '-'.join(parts[:3])
        
        return patient_id
    
    def _load_gene_id_symbol_mapping(self):
        """只保留符号相关映射，后续所有用到gene_id的地方都只用符号"""
        logger.info("加载基因ID和符号的映射关系...")
        gene_files = list(self.kegg_data_dir.glob("hsa*_genes.txt"))
        for gene_file in gene_files:
            try:
                with open(gene_file, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                for line in lines[1:]:
                    parts = line.strip().split('\t')
                    if len(parts) >= 2:
                        entrez_id = parts[0]
                        gene_symbol = parts[1]
                        # 只保留符号相关映射
                        self.gene_symbol_to_id[gene_symbol] = entrez_id
            except Exception as e:
                logger.error(f"读取基因映射文件 {gene_file} 时出错: {e}")
        logger.info(f"加载了 {len(self.gene_symbol_to_id)} 个基因符号到ID的映射")
        
        # 打印一些示例映射
        sample_mappings = list(self.gene_id_to_symbol.items())[:5]
        logger.info(f"基因ID到符号映射示例: {dict(sample_mappings)}")
        
        # 检查是否有常见基因符号
        common_symbols = ["TP53", "BRCA1", "BRCA2", "EGFR", "KRAS", "NLK", "ELK1", "CD14", "CUL2", "CDC25B", "GSK3B"]
        for symbol in common_symbols:
            if symbol in self.gene_symbol_to_id:
                logger.info(f"找到基因符号 {symbol} 对应的ID: {self.gene_symbol_to_id[symbol]}")
            elif symbol.upper() in self.gene_symbol_to_id:
                logger.info(f"找到基因符号 {symbol.upper()} 对应的ID: {self.gene_symbol_to_id[symbol.upper()]}")
            else:
                logger.info(f"未找到基因符号 {symbol} 的映射")
    
    # 恢复原始的_align_data_ids方法，确保保留"取并取交"逻辑
    def _align_data_ids(self):
        """对齐不同数据源的基因ID和患者ID，只用gene_symbol与组学数据对齐"""
        logger.info("对齐不同数据源的ID...")
        
        # 收集KEGG的基因符号
        kegg_gene_symbols = set()
        for gene_symbol in self.gene_info.keys():
            kegg_gene_symbols.add(str(gene_symbol))
        for relation in self.gene_pathway_relations:
            gene_symbol = str(relation["gene_id"])
            kegg_gene_symbols.add(gene_symbol)
        logger.info(f"KEGG数据中共有 {len(kegg_gene_symbols)} 个基因符号")
        
        # 收集组学数据中的基因符号
        gene_symbols_from_data = set()
        if self.expression_data is not None:
            gene_symbols_from_data.update([str(col) for col in self.expression_data.columns])
        if self.mutation_data is not None:
            gene_symbols_from_data.update([str(col) for col in self.mutation_data.columns])
        if self.cnv_data is not None:
            gene_symbols_from_data.update([str(col) for col in self.cnv_data.columns])
        logger.info(f"组学数据中共有 {len(gene_symbols_from_data)} 个基因符号")
        
        # 只用gene_symbol做交集
        kegg_genes_in_data = gene_symbols_from_data & kegg_gene_symbols
        logger.info(f"组学数据中的基因符号与KEGG基因匹配: {len(kegg_genes_in_data)} 个 ({len(kegg_genes_in_data)/len(gene_symbols_from_data)*100 if gene_symbols_from_data else 0:.2f}%)")
        
        # 单独检查每种组学数据与KEGG的重叠
        if self.expression_data is not None:
            expr_genes = set([str(col) for col in self.expression_data.columns])
            expr_kegg_overlap = expr_genes & kegg_gene_symbols
            logger.info(f"表达数据与KEGG基因重叠: {len(expr_kegg_overlap)} 个 ({len(expr_kegg_overlap)/len(expr_genes)*100:.2f}%)")
            
        if self.mutation_data is not None:
            mut_genes = set([str(col) for col in self.mutation_data.columns])
            mut_kegg_overlap = mut_genes & kegg_gene_symbols
            logger.info(f"突变数据与KEGG基因重叠: {len(mut_kegg_overlap)} 个 ({len(mut_kegg_overlap)/len(mut_genes)*100:.2f}%)")
            
        if self.cnv_data is not None:
            cnv_genes = set([str(col) for col in self.cnv_data.columns])
            cnv_kegg_overlap = cnv_genes & kegg_gene_symbols
            logger.info(f"CNV数据与KEGG基因重叠: {len(cnv_kegg_overlap)} 个 ({len(cnv_kegg_overlap)/len(cnv_genes)*100:.2f}%)")
        
        # 从临床数据获取患者ID（在save_processed_data中使用）
        clinical_patients = []
        if self.clinical_data is not None:
            clinical_patients = list(self.clinical_data.index)
            logger.info(f"临床数据中有 {len(clinical_patients)} 个患者")
        
        # 这里我们保留原始的共有患者计算，但实际上不使用它
        # 在save_processed_data中，我们直接使用临床数据中的患者
        common_patients = set()
        all_sources_available = (self.expression_data is not None and 
                                self.mutation_data is not None and 
                                self.cnv_data is not None)
        
        if all_sources_available:
            # 如果所有数据源都可用，取三者的交集
            common_patients = set(self.expression_data.index) & set(self.mutation_data.index) & set(self.cnv_data.index)
            logger.info(f"所有三种组学数据共有的患者: {len(common_patients)}个")
        elif self.expression_data is not None and self.mutation_data is not None:
            # 如果只有表达和突变数据可用
            common_patients = set(self.expression_data.index) & set(self.mutation_data.index)
            logger.info(f"表达数据和突变数据共有的患者: {len(common_patients)}个")
        elif self.expression_data is not None and self.cnv_data is not None:
            # 如果只有表达和CNV数据可用
            common_patients = set(self.expression_data.index) & set(self.cnv_data.index)
            logger.info(f"表达数据和CNV数据共有的患者: {len(common_patients)}个")
        elif self.mutation_data is not None and self.cnv_data is not None:
            # 如果只有突变和CNV数据可用
            common_patients = set(self.mutation_data.index) & set(self.cnv_data.index)
            logger.info(f"突变数据和CNV数据共有的患者: {len(common_patients)}个")
        elif self.expression_data is not None:
            # 如果只有表达数据可用
            common_patients = set(self.expression_data.index)
            logger.info(f"仅使用表达数据中的患者: {len(common_patients)}个")
        elif self.mutation_data is not None:
            # 如果只有突变数据可用
            common_patients = set(self.mutation_data.index)
            logger.info(f"仅使用突变数据中的患者: {len(common_patients)}个")
        elif self.cnv_data is not None:
            # 如果只有CNV数据可用
            common_patients = set(self.cnv_data.index)
            logger.info(f"仅使用CNV数据中的患者: {len(common_patients)}个")
        
        # 记录每个数据源与临床数据共有的患者比例
        if self.clinical_data is not None:
            if self.expression_data is not None:
                clinical_expr_common = set(self.clinical_data.index) & set(self.expression_data.index)
                logger.info(f"临床数据与表达数据共有患者: {len(clinical_expr_common)}个 ({len(clinical_expr_common)/len(self.clinical_data.index)*100:.2f}%)")
            
            if self.mutation_data is not None:
                clinical_mut_common = set(self.clinical_data.index) & set(self.mutation_data.index)
                logger.info(f"临床数据与突变数据共有患者: {len(clinical_mut_common)}个 ({len(clinical_mut_common)/len(self.clinical_data.index)*100:.2f}%)")
            
            if self.cnv_data is not None:
                clinical_cnv_common = set(self.clinical_data.index) & set(self.cnv_data.index)
                logger.info(f"临床数据与CNV数据共有患者: {len(clinical_cnv_common)}个 ({len(clinical_cnv_common)/len(self.clinical_data.index)*100:.2f}%)")
        
        # 存储KEGG基因和临床患者（注意：在save_processed_data中，这些值会被再次设置）
        self.common_genes = list(kegg_gene_symbols)
        self.common_patients = clinical_patients
        
        logger.info(f"保存了 {len(self.common_genes)} 个KEGG基因和 {len(self.common_patients)} 个临床患者ID供后续处理")
    
    def process_clinical_data(self):
        """处理临床数据和生存数据"""
        logger.info("开始处理临床数据...")
        
        # (修改) 使用配置文件中的文件名加载临床数据
        clinical_file = self.raw_data_dir / self.config["clinical_file"]
        if clinical_file.exists():
            logger.info("加载临床数据...")
            # 加载数据，行为样本ID
            self.clinical_data = pd.read_csv(clinical_file, sep='\t', index_col=0)
            logger.info(f"原始临床数据形状: {self.clinical_data.shape}")
            
            # 显示前几个列名
            logger.info(f"临床数据前几个列: {list(self.clinical_data.columns[:5])}")
            
            # 标准化患者ID
            original_index = self.clinical_data.index.tolist()
            self.clinical_data.index = [self._standardize_patient_id(pid) for pid in self.clinical_data.index]
            logger.info(f"样本ID示例 - 原始: {original_index[:3]} → 标准化后: {self.clinical_data.index[:3]}")
            
            # 移除重复的患者ID
            if self.clinical_data.index.duplicated().any():
                dup_count = self.clinical_data.index.duplicated().sum()
                logger.warning(f"检测到 {dup_count} 个重复的患者ID，保留第一个出现的记录")
                self.clinical_data = self.clinical_data[~self.clinical_data.index.duplicated(keep='first')]
            
            # 不再过滤患者，保留所有临床数据中的患者
            logger.info(f"临床数据中有 {len(self.clinical_data)} 个患者")

            # 数据集特定的生存数据列名映射
            survival_column_mapping = {
                "BRCA": {
                    "time_col": "OS_Time_nature2012",
                    "event_col": "OS_event_nature2012"
                },
                "LUAD": {
                    "time_col": "days_to_death",  # LUAD可能使用不同的列名
                    "event_col": "vital_status"    # 需要根据实际列名调整
                }
            }
            
            # 获取当前数据集的生存数据列名
            current_mapping = survival_column_mapping.get(self.dataset_name, survival_column_mapping["BRCA"])
            time_col = current_mapping["time_col"]
            event_col = current_mapping["event_col"]
            
            # 检查生存数据列是否存在
            available_columns = list(self.clinical_data.columns)
            logger.info(f"可用的临床数据列: {available_columns}")
            
            # 尝试找到合适的生存时间列
            time_col_candidates = [time_col, "days_to_death", "OS.time", "overall_survival_time"]
            found_time_col = None
            for col in time_col_candidates:
                if col in available_columns:
                    found_time_col = col
                    break
            
            # 尝试找到合适的生存事件列
            event_col_candidates = [event_col, "vital_status", "OS", "overall_survival_event"]
            found_event_col = None
            for col in event_col_candidates:
                if col in available_columns:
                    found_event_col = col
                    break
            
            # 对于所有数据集都使用通用的生存数据处理逻辑
            # 不再使用简单的单列提取方法，因为TCGA数据需要根据vital_status来选择时间列
            if False:  # 禁用简单方法
                pass
            else:
                # 尝试更灵活的生存数据处理
                logger.info("尝试更灵活的生存数据处理...")
                
                # 通用的生存数据处理逻辑，适用于所有数据集
                if "vital_status" in available_columns:
                    logger.info(f"处理{self.dataset_name}数据集生存数据，保留删失信息...")
                    
                    # 获取临床数据
                    df = self.clinical_data.copy()
                    
                    # 调试信息
                    logger.info(f"=== {self.dataset_name}生存数据调试信息 ===")
                    logger.info(f"总患者数: {len(df)}")
                    logger.info(f"vital_status分布: {df['vital_status'].value_counts()}")
                    logger.info(f"days_to_death非空数量: {df['days_to_death'].notna().sum()}")
                    logger.info(f"days_to_last_followup非空数量: {df['days_to_last_followup'].notna().sum()}")
                    
                    # 处理不同的vital_status值格式
                    # 标准化vital_status值
                    df['vital_status_clean'] = df['vital_status'].str.upper().str.strip()
                    
                    # 识别存活和死亡患者
                    living_values = ['LIVING', 'ALIVE']
                    deceased_values = ['DECEASED', 'DEAD']
                    
                    living = df['vital_status_clean'].isin(living_values) & df['days_to_last_followup'].notna()
                    deceased = df['vital_status_clean'].isin(deceased_values) & df['days_to_death'].notna()
                    
                    logger.info(f"LIVING且有随访数据的患者数: {living.sum()}")
                    logger.info(f"DECEASED且有死亡数据的患者数: {deceased.sum()}")
                    logger.info(f"符合条件的患者总数: {living.sum() + deceased.sum()}")

                    survival = pd.DataFrame(index=df.index)
                    survival['OS'] = None
                    survival['OS.time'] = None

                    # 存活患者：OS=0（删失），时间为最后随访时间
                    survival.loc[living, 'OS'] = 0
                    survival.loc[living, 'OS.time'] = df.loc[living, 'days_to_last_followup']

                    # 死亡患者：OS=1（事件），时间为死亡时间
                    survival.loc[deceased, 'OS'] = 1
                    survival.loc[deceased, 'OS.time'] = df.loc[deceased, 'days_to_death']

                    logger.info(f"赋值后OS非空数量: {survival['OS'].notna().sum()}")
                    logger.info(f"赋值后OS.time非空数量: {survival['OS.time'].notna().sum()}")
                    logger.info(f"OS=0的数量: {(survival['OS'] == 0).sum()}")
                    logger.info(f"OS=1的数量: {(survival['OS'] == 1).sum()}")

                    # 只保留有生存时间和事件的，并且时间大于0
                    valid_survival = survival.dropna(subset=['OS', 'OS.time'])
                    
                    # 确保OS.time是数值类型
                    valid_survival['OS.time'] = pd.to_numeric(valid_survival['OS.time'], errors='coerce')
                    
                    # 再次删除转换后的NaN值，并过滤掉时间为0或负数的记录
                    valid_survival = valid_survival.dropna(subset=['OS', 'OS.time'])
                    valid_survival = valid_survival[valid_survival['OS.time'] > 0]
                    
                    logger.info(f"最终有效生存记录数: {len(valid_survival)}")
                    logger.info("=== 调试信息结束 ===")

                    if len(valid_survival) > 0:
                        self.survival_data = valid_survival
                        logger.info(f"成功提取 {len(self.survival_data)} 条生存记录（包含删失数据）")
                        logger.info(f"生存数据预览:\n{self.survival_data.head()}")
                        censored_count = (self.survival_data['OS'] == 0).sum()
                        event_count = (self.survival_data['OS'] == 1).sum()
                        logger.info(f"删失患者数: {censored_count}, 事件患者数: {event_count}")
                    else:
                        logger.warning("处理后没有有效的生存数据")
                        self.survival_data = None
                else:
                    logger.warning(f"{self.dataset_name}临床数据中未找到vital_status列")
                    self.survival_data = None
                
                # 如果仍然没有找到合适的生存数据
                if self.survival_data is None or len(self.survival_data) == 0:
                    logger.warning(f"在临床数据中未找到合适的生存数据列。尝试的列名: {time_col_candidates}, {event_col_candidates}")
                    logger.warning("可用的列名: " + str(available_columns))
                    self.survival_data = None
        
        else:
            logger.warning(f"未找到临床数据文件: {clinical_file}")
            self.clinical_data = None
            self.survival_data = None

        # 调用特征处理方法
        if self.clinical_data is not None:
            self._process_clinical_features()


    def _process_clinical_features(self):
        """处理临床特征，包括缺失值处理、标准化和独热编码"""
        logger.info("处理临床特征...")
        
        # 选择要使用的临床特征
        clinical_features = self.clinical_data.copy()
        
        # 根据数据集类型选择不同的临床特征
        if self.dataset_name == "BRCA":
            # BRCA数据集的特征选择
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 诊断时的年龄
                'gender',                               # 性别
                'pathologic_stage',                     # 病理分期
                'pathologic_T',                         # T分期
                'pathologic_N',                         # N分期
                'pathologic_M',                         # M分期
                'histological_type',                    # 组织学类型
                'ER_Status_nature2012',                 # ER状态
                'PR_Status_nature2012',                 # PR状态
                'HER2_Final_Status_nature2012',         # HER2状态
                'PAM50Call_RNAseq'                      # PAM50分子分型
            ]
        elif self.dataset_name == "LUAD":
            # LUAD数据集的特征选择
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 诊断时的年龄
                'gender',                               # 性别
                'pathologic_stage',                     # 病理分期
                'pathologic_T',                         # T分期
                'pathologic_N',                         # N分期
                'pathologic_M',                         # M分期
                'histological_type',                    # 组织学类型
                'EGFR',                                 # EGFR突变状态
                'KRAS',                                 # KRAS突变状态
                'ALK_translocation',                    # ALK重排状态
                'BRAF',                                 # BRAF突变状态
                'PIK3CA',                               # PIK3CA突变状态
                'STK11',                                # STK11突变状态
                'tobacco_smoking_history',              # 吸烟史
                'tobacco_smoking_history_indicator',    # 吸烟史指标
                'number_pack_years_smoked',             # 吸烟包年数
                'stopped_smoking_year',                 # 戒烟年份
                'year_of_tobacco_smoking_onset',        # 开始吸烟年份
                'anatomic_neoplasm_subdivision',        # 解剖学肿瘤分区
                'location_in_lung_parenchyma',          # 肺实质位置
                'longest_dimension',                    # 最长维度
                'shortest_dimension',                   # 最短维度
                'intermediate_dimension',               # 中间维度
                'karnofsky_performance_score',          # 卡氏评分
                'eastern_cancer_oncology_group',        # 东部肿瘤协作组
                'radiation_therapy',                    # 放疗
                'targeted_molecular_therapy',           # 靶向分子治疗
                'additional_pharmaceutical_therapy',    # 额外药物治疗
                'additional_radiation_therapy',         # 额外放疗
                'additional_surgery_locoregional_procedure',  # 额外局部手术
                'additional_surgery_metastatic_procedure',    # 额外转移手术
                'residual_tumor',                       # 残留肿瘤
                'new_tumor_event_after_initial_treatment',    # 初始治疗后新肿瘤事件
                'person_neoplasm_cancer_status',        # 患者肿瘤状态
                'primary_therapy_outcome_success',      # 主要治疗结果
                'progression_determined_by',            # 进展确定方式
                'followup_treatment_success',           # 随访治疗成功
                'lost_follow_up',                       # 失访
                'new_neoplasm_event_type'               # 新肿瘤事件类型
                # (修复) 移除vital_status，避免数据泄露
                # 'vital_status'                          # 生存状态
            ]
        elif self.dataset_name == "COAD":
            # 结肠腺癌数据集的特征选择 - 基于实际可用特征
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 诊断时的年龄
                'gender',                               # 性别
                'pathologic_stage',                     # 病理分期
                'pathologic_T',                         # T分期
                'pathologic_N',                         # N分期
                'pathologic_M',                         # M分期
                'histological_type',                    # 组织学类型
                'anatomic_neoplasm_subdivision',        # 解剖学肿瘤分区
                'lymphatic_invasion',                   # 淋巴管侵犯
                'venous_invasion',                      # 静脉侵犯
                'perineural_invasion_present',          # 神经周围侵犯
                'microsatellite_instability',           # 微卫星不稳定性
                'kras_gene_analysis_performed',         # KRAS基因分析
                'kras_mutation_found',                  # KRAS突变发现
                'braf_gene_analysis_performed',         # BRAF基因分析
                'colon_polyps_present',                 # 结肠息肉存在
                'history_of_colon_polyps',              # 结肠息肉病史
                'circumferential_resection_margin',     # 环周切缘
                'lymph_node_examined_count',            # 检查淋巴结数量
                'number_of_lymphnodes_positive_by_he'   # HE阳性淋巴结数量
            ]
        elif self.dataset_name == "GBM":
            # 胶质母细胞瘤数据集的特征选择 - 基于实际可用特征
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 诊断时的年龄
                'gender',                               # 性别
                'histological_type',                    # 组织学类型
                'karnofsky_performance_score',          # 卡氏评分
                'eastern_cancer_oncology_group',        # ECOG评分
                'radiation_therapy',                    # 放疗
                'chemo_therapy',                        # 化疗
                'additional_chemo_therapy',             # 额外化疗
                'additional_drug_therapy',              # 额外药物治疗
                'additional_immuno_therapy',            # 额外免疫治疗
                'additional_pharmaceutical_therapy',    # 额外药物治疗
                'additional_radiation_therapy',         # 额外放疗
                'targeted_molecular_therapy',           # 靶向分子治疗
                'hormonal_therapy',                     # 激素治疗
                'immuno_therapy',                       # 免疫治疗
                'prior_glioma',                         # 既往胶质瘤
                'other_dx'                              # 其他诊断
            ]
        elif self.dataset_name == "KIRC":
            # 肾透明细胞癌数据集的特征选择 - 基于实际可用特征
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 诊断时的年龄
                'gender',                               # 性别
                'pathologic_stage',                     # 病理分期
                'pathologic_T',                         # T分期
                'pathologic_N',                         # N分期
                'pathologic_M',                         # M分期
                'clinical_M',                           # 临床M分期
                'histological_type',                    # 组织学类型
                'neoplasm_histologic_grade',            # 肿瘤组织学分级
                'laterality',                           # 侧别
                'karnofsky_performance_score',          # 卡氏评分
                'eastern_cancer_oncology_group',        # ECOG评分
                'hemoglobin_result',                    # 血红蛋白结果
                'serum_calcium_result',                 # 血清钙结果
                'lactate_dehydrogenase_result',         # 乳酸脱氢酶结果
                'platelet_qualitative_result',          # 血小板定性结果
                'white_cell_count_result',              # 白细胞计数结果
                'erythrocyte_sedimentation_rate_result', # 红细胞沉降率结果
                'tobacco_smoking_history',              # 吸烟史
                'number_pack_years_smoked'              # 吸烟包年数
            ]
        elif self.dataset_name == "LUNG":
            # 肺癌数据集的特征选择 - 基于实际可用特征
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 诊断时的年龄
                'gender',                               # 性别
                'pathologic_stage',                     # 病理分期
                'pathologic_T',                         # T分期
                'pathologic_N',                         # N分期
                'pathologic_M',                         # M分期
                'histological_type',                    # 组织学类型
                'anatomic_neoplasm_subdivision',        # 解剖学肿瘤分区
                'location_in_lung_parenchyma',          # 肺实质位置
                'karnofsky_performance_score',          # 卡氏评分
                'eastern_cancer_oncology_group',        # ECOG评分
                'tobacco_smoking_history',              # 吸烟史
                'tobacco_smoking_history_indicator',    # 吸烟史指标
                'number_pack_years_smoked',             # 吸烟包年数
                'year_of_tobacco_smoking_onset',        # 开始吸烟年份
                'stopped_smoking_year',                 # 戒烟年份
                'radiation_therapy',                    # 放疗
                'targeted_molecular_therapy',           # 靶向分子治疗
                'EGFR',                                 # EGFR突变
                'KRAS',                                 # KRAS突变
                'ALK_translocation',                    # ALK重排
                'BRAF',                                 # BRAF突变
                'RET_translocation',                    # RET重排
                'ROS1_translocation',                   # ROS1重排
                'STK11',                                # STK11突变
                'egfr_mutation_performed',              # EGFR突变检测
                'kras_gene_analysis_performed',         # KRAS基因分析
                'kras_mutation_found'                   # KRAS突变发现
            ]
        elif self.dataset_name == "OV":
            # 卵巢癌数据集的特征选择 - 基于实际可用特征
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 诊断时的年龄
                'clinical_stage',                       # 临床分期
                'histological_type',                    # 组织学类型
                'neoplasm_histologic_grade',            # 肿瘤组织学分级
                'karnofsky_performance_score',          # 卡氏评分
                'eastern_cancer_oncology_group',        # ECOG评分
                'lymphatic_invasion',                   # 淋巴管侵犯
                'venous_invasion',                      # 静脉侵犯
                'tumor_residual_disease',               # 残留病灶
                'residual_tumor',                       # 残留肿瘤
                'radiation_therapy',                    # 放疗
                'additional_pharmaceutical_therapy',    # 额外药物治疗
                'additional_radiation_therapy'          # 额外放疗
            ]
        elif self.dataset_name == "SKCM":
            # 皮肤黑色素瘤数据集的特征选择 - 基于实际可用特征
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 诊断时的年龄
                'gender',                               # 性别
                'pathologic_stage',                     # 病理分期
                'pathologic_T',                         # T分期
                'pathologic_N',                         # N分期
                'pathologic_M',                         # M分期
                'breslow_depth_value',                  # Breslow深度值
                'melanoma_clark_level_value',           # Clark分级值
                'melanoma_ulceration_indicator',        # 黑色素瘤溃疡指标
                'melanoma_origin_skin_anatomic_site',   # 黑色素瘤皮肤解剖部位
                'malignant_neoplasm_mitotic_count_rate', # 恶性肿瘤有丝分裂计数率
                'lactate_dehydrogenase_result',         # 乳酸脱氢酶结果
                'primary_anatomic_site_count',          # 原发解剖部位计数
                'primary_melanoma_at_diagnosis_count',  # 诊断时原发黑色素瘤计数
                'primary_tumor_multiple_present_ind',   # 原发肿瘤多发存在指标
                'radiation_therapy',                    # 放疗
                'additional_pharmaceutical_therapy',    # 额外药物治疗
                'additional_radiation_therapy',         # 额外放疗
                'prior_systemic_therapy_type'           # 既往系统治疗类型
            ]
        elif self.dataset_name == "LIHC":
            # 肝细胞癌数据集的特征选择 - 基于实际可用特征
            selected_features = [
                'age_at_initial_pathologic_diagnosis',
                'gender',
                'pathologic_stage',
                'pathologic_T',
                'pathologic_N',
                'pathologic_M',
                'histological_type',
                'child_pugh_classification_grade',
                'fibrosis_ishak_score',
                'viral_hepatitis_serology',
                'albumin_result_specified_value',
                'bilirubin_upper_limit',
                'creatinine_value_in_mg_dl',
                'prothrombin_time_result_value',
                'platelet_result_count',
                'fetoprotein_outcome_value',
                'eastern_cancer_oncology_group'
            ]
        elif self.dataset_name == "LUSC":
            # 肺鳞癌数据集的特征选择 - 基于实际文件内容
            # (修复) 移除生存相关字段以避免数据泄露: vital_status, days_to_death, days_to_last_followup
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 年龄
                'gender',  # 性别
                'pathologic_stage',  # 病理分期
                'pathologic_T',  # T分期
                'pathologic_N',  # N分期
                'pathologic_M',  # M分期
                'neoplasm_histologic_grade',  # 组织学分级
                'histological_type',  # 组织学类型
                'tobacco_smoking_history',  # 吸烟史
                'karnofsky_performance_score',  # 卡氏评分
                'eastern_cancer_oncology_group'  # ECOG评分
            ]
        elif self.dataset_name == "STAD":
            # 胃腺癌数据集的特征选择 - 基于实际文件内容
            # (修复) 移除生存相关字段以避免数据泄露: vital_status, days_to_death, days_to_last_followup
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 年龄
                'gender',  # 性别
                'pathologic_stage',  # 病理分期
                'pathologic_T',  # T分期
                'pathologic_N',  # N分期
                'pathologic_M',  # M分期
                'neoplasm_histologic_grade',  # 组织学分级
                'h_pylori_infection',  # 幽门螺杆菌感染
                'histological_type'  # 组织学类型
            ]
        elif self.dataset_name == "UCEC":
            # 子宫内膜癌数据集的特征选择 - 基于实际文件内容
            # (修复) 移除生存相关字段以避免数据泄露: vital_status, days_to_death, days_to_last_followup
            # 注意: UCEC只有clinical_stage，没有pathologic_T/N/M分期
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 年龄
                'gender',  # 性别
                'clinical_stage',  # 临床分期
                'neoplasm_histologic_grade',  # 组织学分级
                'histological_type',  # 组织学类型
                'menopause_status',  # 绝经状态
                'birth_control_pill_history_usage_category'  # 避孕药使用史
            ]
        elif self.dataset_name == "HNSC":
            # 头颈鳞癌数据集的特征选择 - 基于实际文件内容
            # (修复) 移除生存相关字段以避免数据泄露: vital_status, days_to_death, days_to_last_followup
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 年龄
                'gender',  # 性别
                'pathologic_stage',  # 病理分期
                'pathologic_T',  # T分期
                'pathologic_N',  # N分期
                'pathologic_M',  # M分期
                'neoplasm_histologic_grade',  # 组织学分级
                'histological_type',  # 组织学类型
                'tobacco_smoking_history',  # 吸烟史
                'alcohol_history_documented',  # 饮酒史
                'hpv_status_by_ish_testing',  # HPV状态 (ISH检测)
                'hpv_status_by_p16_testing'  # HPV状态 (P16检测)
            ]
        elif self.dataset_name == "PAAD":
            # 胰腺腺癌数据集的特征选择 - 基于实际文件内容
            # (修复) 移除生存相关字段以避免数据泄露: vital_status, days_to_death, days_to_last_followup
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 年龄
                'gender',  # 性别
                'pathologic_stage',  # 病理分期
                'pathologic_T',  # T分期
                'pathologic_N',  # N分期
                'pathologic_M',  # M分期
                'neoplasm_histologic_grade',  # 组织学分级
                'histological_type',  # 组织学类型
                'history_of_diabetes',  # 糖尿病史
                'history_of_chronic_pancreatitis',  # 慢性胰腺炎史
                'tobacco_smoking_history',  # 吸烟史（关键预后特征）
                'alcohol_history_documented',  # 饮酒史（关键预后特征）
                'radiation_therapy'  # 放疗史（关键预后特征）
            ]
        elif self.dataset_name == "LGG":
            # 低级别胶质瘤数据集的特征选择 - 基于实际文件内容
            # (修复) 移除生存相关字段以避免数据泄露: vital_status, days_to_death, days_to_last_followup
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 年龄
                'gender',  # 性别
                'neoplasm_histologic_grade',  # 组织学分级
                'histological_type',  # 组织学类型
                'seizure_history',  # 癫痫史
                'motor_movement_changes',  # 运动功能变化
                'karnofsky_performance_score'  # 卡氏评分
            ]
        else:
            # 默认特征选择（通用）
            selected_features = [
                'age_at_initial_pathologic_diagnosis',  # 诊断时的年龄
                'gender',                               # 性别
                'pathologic_stage',                     # 病理分期
                'pathologic_T',                         # T分期
                'pathologic_N',                         # N分期
                'pathologic_M',                         # M分期
                'histological_type',                    # 组织学类型
                'vital_status'                          # 生存状态
            ]
        
        # 只保留存在的列
        available_columns = list(clinical_features.columns)
        logger.info(f"可用的临床特征列: {available_columns}")
        
        selected_features = [col for col in selected_features if col in clinical_features.columns]
        logger.info(f"为数据集 {self.dataset_name} 选择的临床特征: {selected_features}")
        
        clinical_features = clinical_features[selected_features]
        
        # 处理缺失值
        numeric_features = clinical_features.select_dtypes(include=[np.number]).columns
        categorical_features = clinical_features.select_dtypes(include=['object']).columns
        
        # 定义分类型数值变量（评分、等级等）- 这些应该用0或特定值填充，而非均值
        categorical_numeric_features = [
            'karnofsky_performance_score',      # 卡氏评分 (0-100的等级)
            'eastern_cancer_oncology_group',    # ECOG评分 (0-5的等级)
            'neoplasm_histologic_grade',        # 组织学分级 (G1-G4)
            'pathologic_stage',                 # 病理分期（虽然可能已是文本）
            'pathologic_T', 'pathologic_N', 'pathologic_M',  # TNM分期
            'melanoma_clark_level_value',       # Clark分级
            'child_pugh_classification_grade',  # Child-Pugh分级
            'fibrosis_ishak_score',             # 纤维化评分
            'primary_anatomic_site_count',      # 计数（整数）
            'primary_melanoma_at_diagnosis_count'  # 计数（整数）
        ]
        
        # 用均值填充真正的连续数值型特征的缺失值
        for col in numeric_features:
            if clinical_features[col].isnull().sum() > 0:
                # 如果是分类型数值变量，用0填充（表示未知/缺失）
                if col in categorical_numeric_features:
                    clinical_features[col] = clinical_features[col].fillna(0)
                    logger.info(f"  {col}: 用0填充缺失值（分类型数值）")
                else:
                    # 真正的连续变量用均值填充
                    mean_val = clinical_features[col].mean()
                    clinical_features[col] = clinical_features[col].fillna(mean_val)
                    logger.info(f"  {col}: 用均值{mean_val:.2f}填充缺失值（连续型）")
        
        # 用众数填充分类特征的缺失值 - 修复inplace警告
        for col in categorical_features:
            if clinical_features[col].isnull().sum() > 0:
                mode_val = clinical_features[col].mode()[0]
                # 修复pandas警告，不使用inplace=True
                clinical_features[col] = clinical_features[col].fillna(mode_val)
                logger.info(f"  {col}: 用众数'{mode_val}'填充缺失值（分类型）")
        
        # --- 数据泄露修复 ---
        # 移除全局的标准化和独热编码。
        # 这些步骤将被移至训练脚本的交叉验证循环中，以防止数据泄露。
        # 我们在这里只保存经过缺失值填充的、原始的临床特征。

        # 更新处理后的临床特征
        self.processed_clinical_features = clinical_features
        
        logger.info(f"处理后的临床特征形状: {self.processed_clinical_features.shape}")
        logger.info(f"临床特征列名: {list(self.processed_clinical_features.columns)}")
        logger.warning("请注意：临床特征的标准化和独热编码步骤已移除，需在模型训练时处理。")
    
    def save_processed_data(self):
        """保存所有处理后的数据到CSV文件"""
        logger.info("保存处理后的数据...")
        
        # 1. 保存KEGG相关数据
        # 保存通路信息
        pathway_df = pd.DataFrame.from_dict(self.pathway_info, orient='index')
        pathway_df.to_csv(self.processed_data_dir / "kegg_pathway_info.csv", index=False)
        
        # 保存基因信息（包含符号）
        gene_df = pd.DataFrame.from_dict(self.gene_info, orient='index')
        gene_df.to_csv(self.processed_data_dir / "kegg_gene_info.csv", index=False)
        
        # 保存基因ID到符号的映射
        id_symbol_df = pd.DataFrame({
            'gene_id': list(self.gene_id_to_symbol.keys()),
            'gene_symbol': list(self.gene_id_to_symbol.values())
        })
        id_symbol_df.to_csv(self.processed_data_dir / "gene_id_symbol_mapping.csv", index=False)
        
        # 获取最终要保留的、在组学数据中也存在的KEGG基因列表
        kegg_genes_in_data = set()
        if self.expression_data is not None:
            kegg_genes_in_data.update([str(col) for col in self.expression_data.columns])
        # (可以为 mutation_data 和 cnv_data 添加同样逻辑以求并集)
        kegg_gene_symbols = set(self.gene_info.keys())
        final_common_genes = kegg_genes_in_data & kegg_gene_symbols
        logger.info(f"将在图中保留 {len(final_common_genes)} 个共有的基因。")

        # 保存基因-基因关系 (进行过滤)
        if self.gene_gene_relations:
            gene_gene_df = pd.DataFrame(self.gene_gene_relations)
            logger.info(f"过滤前，有 {len(gene_gene_df)} 条基因-基因关系。")

            # 关键过滤步骤：确保源和目标基因都存在于我们的共有基因列表中
            filtered_gg_df = gene_gene_df[
                gene_gene_df['source_gene'].isin(final_common_genes) &
                gene_gene_df['target_gene'].isin(final_common_genes)
            ]
            logger.info(f"过滤后，保留了 {len(filtered_gg_df)} 条基因-基因关系。")

            if not filtered_gg_df.empty:
                filtered_gg_df.to_csv(self.processed_data_dir / "kegg_gene_gene_relations.csv", index=False)
            else:
                logger.warning("过滤后没有剩余的基因-基因关系可供保存。")

        # 保存基因-通路关系 (同样进行过滤)
        if self.gene_pathway_relations:
            gene_pathway_df = pd.DataFrame(self.gene_pathway_relations)
            logger.info(f"过滤前，有 {len(gene_pathway_df)} 条基因-通路关系。")

            # 关键过滤步骤：确保基因存在于我们的共有基因列表中
            filtered_gp_df = gene_pathway_df[gene_pathway_df['gene_id'].isin(final_common_genes)]
            logger.info(f"过滤后，保留了 {len(filtered_gp_df)} 条基因-通路关系。")

            if not filtered_gp_df.empty:
                filtered_gp_df.to_csv(self.processed_data_dir / "kegg_gene_pathway_relations.csv", index=False)
            else:
                logger.warning("过滤后没有剩余的基因-通路关系可供保存。")

        # 2. 获取所有数据源中出现过的全部患者ID
        all_patient_ids = set()
        if self.expression_data is not None:
            all_patient_ids.update(self.expression_data.index)
        if self.mutation_data is not None:
            all_patient_ids.update(self.mutation_data.index)
        if self.cnv_data is not None:
            all_patient_ids.update(self.cnv_data.index)
        if self.clinical_data is not None:
            all_patient_ids.update(self.clinical_data.index)
        logger.info(f"所有数据源中共包含 {len(all_patient_ids)} 个唯一的患者ID")

        # 获取KEGG通路中的基因列表 (只用符号)
        kegg_genes = set()
        if self.gene_pathway_relations:
            for relation in self.gene_pathway_relations:
                gene_symbol = str(relation["gene_id"])
                kegg_genes.add(gene_symbol)
        if self.gene_info:
            for gene_symbol in self.gene_info.keys():
                kegg_genes.add(str(gene_symbol))
        logger.info(f"KEGG通路中有 {len(kegg_genes)} 个基因符号")
        
        # 3. 保存患者组学数据（不过滤患者，只对齐基因）
        
        # 保存表达谱数据
        if self.expression_data is not None:
            # 确保所有列名（基因）都是字符串类型
            expr_genes = [str(col) for col in self.expression_data.columns]
            # 找出属于KEGG通路的基因
            valid_expr_genes = [g for g in expr_genes if g in kegg_genes]
            logger.info(f"表达数据中找到 {len(valid_expr_genes)} 个KEGG通路基因")
            
            # 直接保存所有患者的有效基因表达数据
            expr_subset = self.expression_data[valid_expr_genes]
            expr_subset.to_csv(self.processed_data_dir / "patient_gene_expression.csv")
            logger.info(f"保存了完整的表达数据，形状: {expr_subset.shape}")
            self._create_patient_gene_edge_index(expr_subset, 'expression')

        # 保存突变数据
        if self.mutation_data is not None:
            # 确保所有列名（基因）都是字符串类型
            mut_genes = [str(col) for col in self.mutation_data.columns]
            # 找出属于KEGG通路的基因
            valid_mut_genes = [g for g in mut_genes if g in kegg_genes]
            logger.info(f"突变数据中找到 {len(valid_mut_genes)} 个KEGG通路基因")
            
            # 直接保存所有患者的有效基因突变数据
            mut_subset = self.mutation_data[valid_mut_genes]
            mut_subset.to_csv(self.processed_data_dir / "patient_gene_mutation.csv")
            logger.info(f"保存了完整的突变数据，形状: {mut_subset.shape}")
            self._create_patient_gene_edge_index(mut_subset, 'mutation')
            
        # 保存CNV数据
        if self.cnv_data is not None:
            # 确保所有列名（基因）都是字符串类型
            cnv_genes = [str(col) for col in self.cnv_data.columns]
            # 找出属于KEGG通路的基因
            valid_cnv_genes = [g for g in cnv_genes if g in kegg_genes]
            logger.info(f"CNV数据中找到 {len(valid_cnv_genes)} 个KEGG通路基因")

            # 直接保存所有患者的有效基因CNV数据
            cnv_subset = self.cnv_data[valid_cnv_genes]
            cnv_subset.to_csv(self.processed_data_dir / "patient_gene_cnv.csv")
            logger.info(f"保存了完整的CNV数据，形状: {cnv_subset.shape}")
            self._create_patient_gene_edge_index(cnv_subset, 'cnv')

        # 4. 保存临床特征和生存数据
        # 首先保存生存数据
        if self.survival_data is not None:
            self.survival_data.to_csv(self.processed_data_dir / "patient_survival.csv")
            logger.info(f"保存了 {len(self.survival_data)} 条生存记录")
            
            # 获取有生存数据的患者ID列表
            survival_patients = set(self.survival_data.index)
            logger.info(f"生存数据中有 {len(survival_patients)} 个患者")
        else:
            logger.warning("没有生存数据可用，无法保存patient_survival.csv")
            survival_patients = set()
        
        # 保存临床特征数据（保存为_raw.csv，供build_hetero_graph.py读取并简化）
        if hasattr(self, 'processed_clinical_features') and self.processed_clinical_features is not None:
            output_filename = "patient_clinical_features_raw.csv"
            
            if len(survival_patients) > 0:
                # 如果有生存数据，筛选出有生存数据的患者的临床特征
                clinical_features_filtered = self.processed_clinical_features.loc[
                    self.processed_clinical_features.index.isin(survival_patients)
                ]
                logger.info(f"保存了 {len(clinical_features_filtered)} 个有生存数据的患者的临床特征到 {output_filename}")
                
                # 检查是否有患者在生存数据中但不在临床特征中
                missing_patients = survival_patients - set(clinical_features_filtered.index)
                if missing_patients:
                    logger.warning(f"有 {len(missing_patients)} 个患者在生存数据中但不在临床特征中")
            else:
                # 如果没有生存数据，保存所有患者的临床特征
                clinical_features_filtered = self.processed_clinical_features
                logger.info(f"保存了所有 {len(clinical_features_filtered)} 个患者的临床特征到 {output_filename}")
            
            # 保存临床特征
            clinical_features_filtered.to_csv(self.processed_data_dir / output_filename)
        else:
            logger.warning("没有临床特征数据可用，无法保存patient_clinical_features.csv")
        
        # 5. 更新元数据，保存所有患者和KEGG基因数量
        metadata = {
            "num_patients": len(all_patient_ids),
            "num_genes": len(kegg_genes),
            "num_pathways": len(self.pathway_info),
            "data_sources": {
                "expression": self.expression_data is not None,
                "mutation": self.mutation_data is not None,
                "cnv": self.cnv_data is not None,
                "clinical": hasattr(self, 'processed_clinical_features') and self.processed_clinical_features is not None,
                "survival": self.survival_data is not None
            }
        }
        
        # --- (修复) 明确保存ID映射 ---
        # 重新生成最终的ID映射，确保一致性
        final_patient_ids = sorted(list(all_patient_ids))
        final_gene_symbols = sorted(list(kegg_genes))
        final_pathway_ids = sorted(list(self.pathway_info.keys()))

        id_mappings = {
            'patient_id_to_idx': {pid: i for i, pid in enumerate(final_patient_ids)},
            'gene_id_to_idx': {gid: i for i, gid in enumerate(final_gene_symbols)},
            'pathway_id_to_idx': {pid: i for i, pid in enumerate(final_pathway_ids)}
        }
        
        # 保存ID映射文件
        with open(self.processed_data_dir / "id_mappings.json", 'w') as f:
            json.dump(id_mappings, f, indent=2)
        logger.info(f"ID映射文件已保存到: {self.processed_data_dir / 'id_mappings.json'}")
        
        # 更新共有基因和患者列表，以便其他方法使用
        self.common_genes = list(kegg_genes)
        self.common_patients = list(all_patient_ids)
        
        with open(self.processed_data_dir / "hetero_data_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("所有处理后的数据已保存")
    
    def _create_edge_index_files(self, relations_df, src_col, dst_col, type_col, src_type, dst_type):
        """根据关系DataFrame创建边索引文件"""
        # 为每种关系类型创建边索引
        relation_types = relations_df[type_col].unique()
        
        for rel_type in relation_types:
            # 过滤出当前关系类型的边
            edges = relations_df[relations_df[type_col] == rel_type]
            
            # 创建边索引（源节点索引和目标节点索引）
            edge_index = np.array([
                edges[src_col].values,
                edges[dst_col].values
            ])
            
            # 将关系类型中的特殊字符替换为下划线，确保文件名有效
            safe_rel_type = str(rel_type).replace('/', '_').replace('\\', '_').replace(' ', '_')
            
            # 保存边索引
            edge_file_name = f"{src_type}_{safe_rel_type}_{dst_type}_edge_index.npy"
            logger.info(f"保存边索引: {edge_file_name}")
            np.save(self.processed_data_dir / edge_file_name, edge_index)
            
            # 为反方向创建边索引
            rev_edge_file_name = f"{dst_type}_rev_{safe_rel_type}_{src_type}_edge_index.npy"
            rev_edge_index = np.array([
                edges[dst_col].values,
                edges[src_col].values
            ])
            np.save(self.processed_data_dir / rev_edge_file_name, rev_edge_index)
    
    def _create_patient_gene_edge_index(self, data_matrix, edge_type):
        """为患者-基因关系创建边索引和边属性文件"""
        logger.info(f"为 {edge_type} 创建边索引和属性文件...")
        
        # 获取患者和基因的ID映射
        patient_map = {pid: i for i, pid in enumerate(self.common_patients)}
        gene_map = {gid: i for i, gid in enumerate(self.common_genes)}
        
        edge_index = []
        edge_attr = []
        
        # 遍历数据矩阵
        for patient_id, row in data_matrix.iterrows():
            if patient_id in patient_map:
                patient_idx = patient_map[patient_id]
                for gene_id, value in row.items():
                    if gene_id in gene_map:
                        gene_idx = gene_map[gene_id]
                        
                        # 只添加有意义的边
                        if edge_type == 'mutation' and value > 0:
                            edge_index.append([patient_idx, gene_idx])
                            # 突变只有存在性，属性为1
                            edge_attr.append([1.0])
                        elif edge_type == 'cnv' and value != 0:
                            edge_index.append([patient_idx, gene_idx])
                            # CNV属性为离散值
                            edge_attr.append([float(value)])
                        elif edge_type == 'expression':
                            edge_index.append([patient_idx, gene_idx])
                            # 表达属性为连续值
                            edge_attr.append([float(value)])
        
        if edge_index:
            edge_index = np.array(edge_index).T
            edge_attr = np.array(edge_attr)
            
            # 保存文件
            base_name = f"patient_{edge_type}_gene"
            np.save(self.processed_data_dir / f"{base_name}_edge_index.npy", edge_index)
            np.save(self.processed_data_dir / f"{base_name}_edge_attr.npy", edge_attr)
            logger.info(f"保存了 {base_name} 边文件，索引形状: {edge_index.shape}，属性形状: {edge_attr.shape}")
            
            # 创建反向边
            rev_edge_index = np.array([edge_index[1], edge_index[0]])
            rev_base_name = f"gene_rev_{edge_type}_patient"
            np.save(self.processed_data_dir / f"{rev_base_name}_edge_index.npy", rev_edge_index)
            logger.info(f"保存了 {rev_base_name} 反向边索引文件，形状: {rev_edge_index.shape}")


def main():
    """主函数，用于执行数据准备流程"""
    # (新增) 设置命令行参数解析器
    parser = argparse.ArgumentParser(description="数据准备脚本，用于处理特定数据集的原始数据。")
    parser.add_argument(
        '--dataset', 
        type=str, 
        required=True,
        help=f"要处理的数据集名称。可用选项: {list(DATASET_CONFIGS.keys())}"
    )
    args = parser.parse_args()

    logger.info("="*50)
    logger.info(f"开始为数据集 '{args.dataset}' 进行数据准备")
    logger.info("="*50)
    
    # (修改) 初始化并运行数据准备流程，传入数据集名称
    preparer = DataPreparation(dataset_name=args.dataset)
    preparer.run()
    
    logger.info("="*50)
    logger.info(f"数据集 '{args.dataset}' 的数据准备完成")
    logger.info("="*50)

if __name__ == "__main__":
    main()