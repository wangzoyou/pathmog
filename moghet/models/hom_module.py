import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalOmicsModulation(nn.Module):
    """
    分层组学调制 (HOM) 模块。

    此模块不将多组学数据视为平等输入，而是将基因表达作为核心信号，
    利用CNV、突变、临床和通路上下文来动态地"调制"这个信号。
    它学习生成一个缩放因子γ和一个偏移因子β，以校准每个基因的表达特征。
    """
    def __init__(self, modulator_dim, clinical_dim, pathway_dim, hidden_dim):
        super().__init__()
        
        # 调制器网络，输入维度 = 突变/CNV(2) + 临床 + 通路
        context_dim = modulator_dim + clinical_dim + pathway_dim
        
        self.modulator_net = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.ReLU(),
            # 输出2个值，分别用于生成 gamma 和 beta
            nn.Linear(hidden_dim, 2) 
        )
        
        # 关键：初始化最后一个线性层的权重和偏置为0
        # 这使得在训练开始时，gamma ≈ 1, beta ≈ 0，
        # 整个模块近似于一个恒等变换，让模型训练更稳定。
        self.modulator_net[-1].weight.data.fill_(0)
        self.modulator_net[-1].bias.data.fill_(0)

    def forward(self, gene_features, pathway_context, clinical_context):
        """
        Args:
            gene_features (Tensor): 原始基因组学特征, 形状 [N_genes, 3] (表达, CNV, 突变)
            pathway_context (Tensor): 每个基因所属的通路上下文特征。
            clinical_context (Tensor): 每个基因所属的患者临床上下文特征。
        
        Returns:
            modulated_expression (Tensor): 调制后的基因表达特征, 形状 [N_genes, 1]。
            gamma (Tensor): 生成的缩放因子, 形状 [N_genes, 1]。
            beta (Tensor): 生成的偏移因子, 形状 [N_genes, 1]。
        """
        # === 数值稳定性检查 ===
        # 检查输入是否包含NaN或无穷大
        if torch.isnan(gene_features).any() or torch.isinf(gene_features).any():
            print("警告: HOM模块检测到输入gene_features包含NaN或无穷大值")
            gene_features = torch.nan_to_num(gene_features, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(pathway_context).any() or torch.isinf(pathway_context).any():
            print("警告: HOM模块检测到输入pathway_context包含NaN或无穷大值")
            pathway_context = torch.nan_to_num(pathway_context, nan=0.0, posinf=1e6, neginf=-1e6)
        
        if torch.isnan(clinical_context).any() or torch.isinf(clinical_context).any():
            print("警告: HOM模块检测到输入clinical_context包含NaN或无穷大值")
            clinical_context = torch.nan_to_num(clinical_context, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # 1. 将原始多组学特征拆分为核心信号和调制信号
        expression_signal = gene_features[:, 0:1] # [N_genes, 1]
        modulator_signals = gene_features[:, 1:3] # [N_genes, 2] (CNV, 突变)
        
        # 2. 组合所有上下文信息
        full_context = torch.cat([
            modulator_signals,  # 基因自身的CNV和突变状态
            pathway_context,    # 基因所处的通路环境
            clinical_context    # 基因所属的患者临床画像
        ], dim=1)
        
        # 3. 通过调制器网络生成 gamma 和 beta
        # modulation_params 的形状: [N_genes, 2]
        modulation_params = self.modulator_net(full_context)
        
        # === 数值稳定性处理 ===
        # 限制modulation_params的范围，防止数值不稳定
        modulation_params = torch.clamp(modulation_params, min=-10.0, max=10.0)
        
        # gamma 是乘性因子，我们希望它以1为中心，限制在合理范围内
        gamma_raw = modulation_params[:, 0:1]
        gamma = 1 + torch.tanh(gamma_raw) * 0.5  # 限制gamma在[0.5, 1.5]范围内
        
        # beta 是加性因子，我们希望它以0为中心
        beta = torch.tanh(modulation_params[:, 1:2]) * 2.0  # 限制beta在[-2, 2]范围内
        
        # 4. 应用FiLM调制
        modulated_expression = gamma * expression_signal + beta
        
        # === 最终数值稳定性检查 ===
        if torch.isnan(modulated_expression).any() or torch.isinf(modulated_expression).any():
            print("警告: HOM模块输出包含NaN或无穷大值，使用原始表达信号")
            modulated_expression = expression_signal
            gamma = torch.ones_like(gamma)
            beta = torch.zeros_like(beta)
        
        if torch.isnan(gamma).any() or torch.isinf(gamma).any():
            print("警告: HOM模块gamma包含NaN或无穷大值，使用默认值")
            gamma = torch.ones_like(gamma)
        
        if torch.isnan(beta).any() or torch.isinf(beta).any():
            print("警告: HOM模块beta包含NaN或无穷大值，使用默认值")
            beta = torch.zeros_like(beta)
        
        return modulated_expression, gamma, beta 