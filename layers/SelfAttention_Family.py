from ast import Import
from regex import F
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
from utils.masking import TriangularCausalMask, ProbMask
from reformer_pytorch import LSHSelfAttention
from einops import rearrange, repeat
import torch.nn.functional as Func



class DSAttention(nn.Module):
    '''De-stationary Attention'''

    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(DSAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        tau = 1.0 if tau is None else tau.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x 1
        delta = 0.0 if delta is None else delta.unsqueeze(
            1).unsqueeze(1)  # B x 1 x 1 x S

        # De-stationary Attention, rescaling pre-softmax score with learned de-stationary factors
        scores = torch.einsum("blhe,bshe->bhls", queries, keys) * tau + delta

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None


class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class Zero_FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(Zero_FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.tau = 1.0
        self.scale_zero = 0.8
        self.scale_non_zero = 0.6
        self.position_bias_scale = 0.1
    def _compute_distance_to_zero(self, zero_mask):
        """
        计算每个位置到最近零值的距离
        Args:
            zero_mask: [B, L_K]（True表示零值）
        Returns:
            distance: [B, L_K]（值越小表示离零值越近）
        """
        B, L_K = zero_mask.shape
        device = zero_mask.device

        # 生成位置索引矩阵
        positions = torch.arange(L_K, device=device).view(1, L_K).expand(B, L_K)  # [B, L_K]

        # 找到每个位置左侧和右侧最近的零值索引
        left_zero = torch.zeros_like(positions)
        right_zero = torch.zeros_like(positions)
        
        for b in range(B):
            zero_indices = torch.where(zero_mask[b])[0]
            if len(zero_indices) == 0:
                left_zero[b] = -L_K
                right_zero[b] = 2 * L_K
            else:
                # 左侧最近零值
                left_zero[b] = torch.cummax(
                    torch.where(zero_mask[b], positions[b], -L_K), dim=-1
                )[0]
                # 右侧最近零值
                right_zero[b] = L_K - torch.cummax(
                    torch.where(zero_mask[b].flip(-1), L_K - positions[b], -L_K), dim=-1
                )[0].flip(-1)

        # 计算最小距离
        distance = torch.min(
            torch.abs(positions - left_zero),
            torch.abs(right_zero - positions)
        ).float()  # [B, L_K]

        return distance
    def zero_pos_att_bias(self, scores, zero_mask):
        device = scores.device
        if zero_mask is not None:
              ##attention1
            B, H, L_Q, L_K = scores.shape
            # 计算零值距离（确保结果在相同设备）
            distance_to_zero = self._compute_distance_to_zero(zero_mask.bool()).to(device)  # [B, L_K]
            
            # 计算权重并扩展到匹配scores的形状
            zero_proximity_weight = torch.exp(-distance_to_zero / self.tau).view(B, 1, 1, -1)  # [B, 1, 1, L_K]
            zero_proximity_weight = zero_proximity_weight[:,:,-L_Q:,-L_K:]
            # 调整得分
            scores = scores + zero_proximity_weight * self.scale_zero
        
        # 新增位置偏置处理
        L_K = scores.shape[-1]
        j_indices = torch.arange(L_K, device=device).float()
        position_bias = (j_indices / L_K) * self.position_bias_scale  # 线性递增偏置
        position_bias = position_bias.view(1, 1, 1, L_K).expand(scores.size(0), scores.size(1), scores.size(2), -1)
        scores += position_bias

        return scores

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)

        scores = self.zero_pos_att_bias(scores=scores, zero_mask=attn_mask)


        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)

            scores.masked_fill_(attn_mask.mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None

class PeriodAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(PeriodAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        
        # 新增：定义周期偏置参数（1小时和2小时周期）
        self.bias_1h = nn.Parameter(torch.tensor(0.0))  # 1小时周期对应的偏置
        self.bias_2h = nn.Parameter(torch.tensor(0.0))  # 2小时周期对应的偏置
        # self.sigma = nn.Parameter(torch.tensor(0.3))    # 可学习的高斯分布标准差
        self.sigma = 2.0  # 不可学习的高斯分布标准差

    def _create_periodic_mask(self, L, S, device, period):
        """创建周期性位置掩码"""
        rows = torch.arange(L, device=device).view(-1, 1)  # 行坐标 (L,1)
        cols = torch.arange(S, device=device).view(1, -1)   # 列坐标 (1,S)
        diff = rows - cols                                  # 位置差矩阵 (L,S)
        
        # 满足两个条件的位置设为1：
        # 1. 因果掩码（只能看到过去）
        # 2. 时间差是周期（4或8步）的整数倍
        mask = (diff >= 0) & ((diff % period) == 0)
        return mask.float()  # 转换为浮点型 (L,S)
    def _create_periodic_curve(self, L, S, device, period):
        """创建周期性衰减曲线"""
        rows = torch.arange(L, device=device).view(-1, 1)  # (L,1)
        cols = torch.arange(S, device=device).view(1, -1)   # (1,S)
        diff = (rows - cols).float()                       # (L,S)
        
        # 高斯衰减曲线（在周期点处最大，向两侧衰减）
        distance_to_period = torch.abs((diff % period) - 0)  # 距离最近周期点的距离
        curve = torch.exp(-0.5 * (distance_to_period / self.sigma)**2)  # (L,S)
        
        # 因果掩码（只允许看过去）
        causal_mask = (diff >= 0).float()
        return curve * causal_mask

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / sqrt(E)

        # 步骤1: 计算基础注意力分数
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)  # (B,H,L,S)

        # 步骤2: 应用因果掩码（如果启用）
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        # 新增步骤3: 添加周期性偏置 ------------------------------------------
       
        # mask_1h = self._create_periodic_mask(L, S, queries.device, 24)  # (L,S)
        # mask_2h = self._create_periodic_mask(L, S, queries.device, 48)  # (L,S)
        # periodic_bias = self.bias_1h * mask_1h + self.bias_2h * mask_2h
        # 扩展掩码维度以匹配注意力分数形状 (B,H,L,S)
        # mask_1h = mask_1h.unsqueeze(0).unsqueeze(0)  # (1,1,L,S)
        # mask_2h = mask_2h.unsqueeze(0).unsqueeze(0)  # (1,1,L,S)
        # 创建周期性曲线 ------------------------------------------
        curve_1h = self._create_periodic_curve(L, S, queries.device, 24)  # (L,S)
        curve_2h = self._create_periodic_curve(L, S, queries.device, 48)  # (L,S)
        
        # 扩展维度并加权求和
        curve_1h = curve_1h.unsqueeze(0).unsqueeze(0)  # (1,1,L,S)
        curve_2h = curve_2h.unsqueeze(0).unsqueeze(0)  # (1,1,L,S)
        
        # 应用可学习幅度的周期性曲线
        periodic_bias = self.bias_1h * curve_1h + self.bias_2h * curve_2h
        # 计算偏置项并叠加到注意力分数
        
        scores += periodic_bias.expand_as(scores)  # 扩展到(B,H,L,S)
        
        # 步骤4: 计算最终注意力权重
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        
        # 步骤5: 计算加权值向量
        V = torch.einsum("bhls,bshd->blhd", A, values)

        if self.output_attention:
            return V.contiguous(), A
        else:
            return V.contiguous(), None
class ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def _prob_QK(self, Q, K, sample_k, n_top):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        B, H, L_K, E = K.shape
        _, _, L_Q, _ = Q.shape

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k

        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex

    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask):
        B, H, L_V, D = V.shape

        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)

        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q

        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries
        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask)

        return context.contiguous(), attn
class Zero_ProbAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(Zero_ProbAttention, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
        self.tau = 1.0
        self.scale_zero = 0.8
        self.scale_non_zero = 0.6
        self.position_bias_scale = 0.5

    def _prob_QK(self, Q, K, sample_k, n_top, zero_mask=None):  # n_top: c*ln(L_q)
        # Q [B, H, L, D]
        # zero_mask: [B, L_K]
        B, H, L_K, E = K.shape
        # print('K.shape',K.shape)
        # print('L_K',L_K)
        _, _, L_Q, _ = Q.shape
        # print('Q.shape',Q.shape)
        # print('zero_mask.shape',zero_mask.shape)
        # ------------------------------
        # 改进点：根据零值分布调整采样概率
        # ------------------------------
        if zero_mask is not None:
            # 计算每个 Key 点附近的零值密度（假设 zero_mask 是 [B, L_K]）
            # print('zero_mask',zero_mask)
            # zero_density = zero_mask.float().unfold(dimension=1, size=7, step=1).mean(dim=-1)  # [B, L_K-6]
            # weights = 1.0 - zero_density
            # weights = Func.pad(weights, (3, 3), value=0.2)  # [B, L_K]
            # weights = weights[:, :L_K]  # 确保对齐
            # print('weights',weights)
            # print('zero_mask',zero_mask.device)
            zero_mask = zero_mask.to(K.device)
            weights = torch.where(
                zero_mask, 
                torch.full_like(zero_mask, 0.5, dtype=torch.float, device=K.device),
                torch.ones_like(zero_mask, dtype=torch.float, device=K.device)
                )
            weights = weights[:, :L_K]
            # print('weights',weights)
            # 如果希望所有 Query 共享相同的采样权重（按 Batch 平均）
            weights = weights.mean(dim=0)  # [L_K]
            weights = weights.unsqueeze(0).expand(L_Q, -1)  # [L_Q, L_K]
        else:
            weights = torch.ones(L_Q, L_K, device=K.device)  # 均匀采样 [L_Q, L_K]

        # 按权重采样（每个 Query 独立采样 sample_k 个 Key）
        index_sample = torch.multinomial(weights, sample_k, replacement=True)  # [L_Q, sample_k]
        

        # calculate the sampled Q_K
        K_expand = K.unsqueeze(-3).expand(B, H, L_Q, L_K, E)
        # real U = U_part(factor*ln(L_k))*L_q
        index_sample = torch.randint(L_K, (L_Q, sample_k))
        K_sample = K_expand[:, :, torch.arange(
            L_Q).unsqueeze(1), index_sample, :]
        Q_K_sample = torch.matmul(
            Q.unsqueeze(-2), K_sample.transpose(-2, -1)).squeeze()

        # find the Top_k query with sparisty measurement
        M = Q_K_sample.max(-1)[0] - torch.div(Q_K_sample.sum(-1), L_K)
        M_top = M.topk(n_top, sorted=False)[1]

        # use the reduced Q to calculate Q_K
        Q_reduce = Q[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   M_top, :]  # factor*ln(L_q)
        Q_K = torch.matmul(Q_reduce, K.transpose(-2, -1))  # factor*ln(L_q)*L_k
        return Q_K, M_top

    def _get_initial_context(self, V, L_Q):
        B, H, L_V, D = V.shape
        if not self.mask_flag:
            # V_sum = V.sum(dim=-2)
            V_sum = V.mean(dim=-2)
            contex = V_sum.unsqueeze(-2).expand(B, H,
                                                L_Q, V_sum.shape[-1]).clone()
        else:  # use mask
            # requires that L_Q == L_V, i.e. for self-attention only
            assert (L_Q == L_V)
            contex = V.cumsum(dim=-2)
        return contex
    def _compute_distance_to_zero_new(self, zero_mask):
        """
        计算改进后的距离矩阵：
        - 非零值：到最近零值的距离
        - 零值：到次近零值的距离（排除自身）
        0430:改进方向，加大离0值近的非0值的注意力（加大对突出下雨的关注），减弱离0值近的0值的注意力（减弱多0值聚集的注意力）
        """
        B, L_K = zero_mask.shape
        device = zero_mask.device
        positions = torch.arange(L_K, device=device).view(1, L_K).expand(B, L_K)  # [B, L_K]
        distance = torch.zeros_like(zero_mask, dtype=torch.float32)
        
        for b in range(B):
            zero_indices = torch.where(zero_mask[b])[0]
            if len(zero_indices) == 0:
                distance[b] = L_K  # 无零值时所有位置距离设为最大
                continue
            
            # 计算所有位置到每个零值的距离矩阵
            pos = positions[b].unsqueeze(1)  # [L_K, 1]
            zeros = zero_indices.unsqueeze(0)  # [1, num_zeros]
            dist_matrix = torch.abs(pos - zeros)  # [L_K, num_zeros]
            
            # 对零值位置排除自身后找次近距离
            self_mask = (zeros == pos)  # [L_K, num_zeros]
            dist_matrix_self_excluded = dist_matrix.masked_fill(self_mask, L_K)
            min_dist_self_excluded = torch.min(dist_matrix_self_excluded, dim=1)[0]  # [L_K]
            
            # 对非零值取原始最近距离
            min_dist_original = torch.min(dist_matrix, dim=1)[0]  # [L_K]
            
            # 合并结果：零值用次近距离，非零用原始距离
            is_zero = zero_mask[b]
            combined_dist = torch.where(is_zero, min_dist_self_excluded, min_dist_original)
            distance[b] = combined_dist
            
        return distance  # [B, L_K]
    def _compute_distance_to_zero(self, zero_mask):
        """
        计算每个位置到最近零值的距离
        Args:
            zero_mask: [B, L_K]（True表示零值）
        Returns:
            distance: [B, L_K]（值越小表示离零值越近）
        """
        B, L_K = zero_mask.shape
        device = zero_mask.device

        # 生成位置索引矩阵
        positions = torch.arange(L_K, device=device).view(1, L_K).expand(B, L_K)  # [B, L_K]

        # 找到每个位置左侧和右侧最近的零值索引
        left_zero = torch.zeros_like(positions)
        right_zero = torch.zeros_like(positions)
        
        for b in range(B):
            zero_indices = torch.where(zero_mask[b])[0]
            if len(zero_indices) == 0:
                left_zero[b] = -L_K
                right_zero[b] = 2 * L_K
            else:
                # 左侧最近零值
                left_zero[b] = torch.cummax(
                    torch.where(zero_mask[b], positions[b], -L_K), dim=-1
                )[0]
                # 右侧最近零值
                right_zero[b] = L_K - torch.cummax(
                    torch.where(zero_mask[b].flip(-1), L_K - positions[b], -L_K), dim=-1
                )[0].flip(-1)

        # 计算最小距离
        distance = torch.min(
            torch.abs(positions - left_zero),
            torch.abs(right_zero - positions)
        ).float()  # [B, L_K]

        return distance
    def _update_context(self, context_in, V, scores, index, L_Q, attn_mask, zero_mask=None):
        B, H, L_V, D = V.shape
        # print('scores.shape',scores.shape)
        # print(attn_mask.shape)
        device = scores.device 
        # if self.mask_flag:
        #     attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
        #     scores.masked_fill_(attn_mask.mask, -np.inf)
        # ------------------------------
        # 新增：零值邻近注意力增强
        # ------------------------------
        if zero_mask is not None:
              ##attention1
            B, H, L_Q, L_K = scores.shape
            # 计算零值距离（确保结果在相同设备）
            distance_to_zero = self._compute_distance_to_zero(zero_mask.bool()).to(device)  # [B, L_K]
            
            # 计算权重并扩展到匹配scores的形状
            zero_proximity_weight = torch.exp(-distance_to_zero / self.tau).view(B, 1, 1, -1)  # [B, 1, 1, L_K]
            zero_proximity_weight = zero_proximity_weight[:,:,-L_Q:,-L_K:]
            # 调整得分
            scores = scores + zero_proximity_weight * self.scale_zero
        
        # 新增位置偏置处理
        L_K = scores.shape[-1]
        j_indices = torch.arange(L_K, device=device).float()
        position_bias = (j_indices / L_K) * self.position_bias_scale  # 线性递增偏置
        position_bias = position_bias.view(1, 1, 1, L_K).expand(scores.size(0), scores.size(1), scores.size(2), -1)
        scores += position_bias
        # if zero_mask is not None:
        #     ##attention2
        #     # 生成非零值位置的权重（非零值为1，零值为0）
        #     non_zero_mask = (~zero_mask).float()  # [B, L_K]
        #     # 扩展维度以匹配scores的形状 [B, H, L_Q, L_K]
        #     non_zero_weight = non_zero_mask.view(B, 1, 1, -1) * self.scale_zero
        #     # 调整注意力得分
        #     scores = scores + non_zero_weight.to(device)
        # if zero_mask is not None:
        #     ##attention3
            
        #     # 计算零值距离（确保结果在相同设备）
        #     distance_to_zero = self._compute_distance_to_zero(zero_mask.bool()).to(device)  # [B, L_K]
            
        #     # 计算权重并扩展到匹配scores的形状
        #     zero_proximity_weight = -torch.exp(-distance_to_zero / self.tau).view(B, 1, 1, -1)  # [B, 1, 1, L_K]
            
        #     # 调整得分
        #     scores = scores + zero_proximity_weight * self.scale_zero
        # if zero_mask is not None:
        #     ###attention4
        #     # 计算改进后的距离矩阵
        #     distance = self._compute_distance_to_zero_new(zero_mask).to(device)  # [B, L_K]
            
        #     # 分离非零值与零值掩码
        #     non_zero_mask = (~zero_mask).float().to(device) # [B, L_K]
        #     zero_mask_pos = zero_mask.float().to(device)     # [B, L_K]
            
        #     # 非零值：距离越近，正权重越大
        #     non_zero_weights = torch.exp(-distance / self.tau).to(device) * self.scale_non_zero
        #     non_zero_weights *= non_zero_mask.to(device)  # 仅作用于非零位置
            
        #     # 零值：次近距离越近，负权重越大（惩罚聚集区）
        #     zero_weights = -torch.exp(-distance / self.tau).to(device) * self.scale_zero
        #     zero_weights *= zero_mask_pos  # 仅作用于零值位置
            
        #     # 合并权重并调整注意力得分
        #     combined_weights = (non_zero_weights + zero_weights).view(B, 1, 1, -1)
        #     scores = scores + combined_weights.to(device)
        if self.mask_flag:
            attn_mask = ProbMask(B, H, L_Q, index, scores, device=V.device)
            scores.masked_fill_(attn_mask.mask, -np.inf)

        attn = torch.softmax(scores, dim=-1)  # nn.Softmax(dim=-1)(scores)
        context_in[torch.arange(B)[:, None, None],
        torch.arange(H)[None, :, None],
        index, :] = torch.matmul(attn, V).type_as(context_in)
        if self.output_attention:
            attns = (torch.ones([B, H, L_V, L_V]) /
                     L_V).type_as(attn).to(attn.device)
            attns[torch.arange(B)[:, None, None], torch.arange(H)[
                                                  None, :, None], index, :] = attn
            return context_in, attns
        else:
            return context_in, None

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L_Q, H, D = queries.shape
        _, L_K, _, _ = keys.shape

        queries = queries.transpose(2, 1)
        keys = keys.transpose(2, 1)
        values = values.transpose(2, 1)

        U_part = self.factor * \
                 np.ceil(np.log(L_K)).astype('int').item()  # c*ln(L_k)
        u = self.factor * \
            np.ceil(np.log(L_Q)).astype('int').item()  # c*ln(L_q)

        U_part = U_part if U_part < L_K else L_K
        u = u if u < L_Q else L_Q
        # print('attn_mask',attn_mask)
        scores_top, index = self._prob_QK(
            queries, keys, sample_k=U_part, n_top=u, zero_mask=attn_mask)

        # add scale factor
        scale = self.scale or 1. / sqrt(D)
        if scale is not None:
            scores_top = scores_top * scale
        # get the context
        context = self._get_initial_context(values, L_Q)
        # update the context with selected top_k queries

        context, attn = self._update_context(
            context, values, scores_top, index, L_Q, attn_mask,zero_mask=attn_mask)
        # print('context.shape',context.shape)

        return context.contiguous(), attn

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None,mix=False):
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
        self.mix = mix

    def forward(self, queries, keys, values, attn_mask, tau=None, delta=None):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(
            queries,
            keys,
            values,
            attn_mask,
            tau=tau,
            delta=delta
        )
        if self.mix:
            out = out.transpose(2,1).contiguous()
        out = out.view(B, L, -1)

        return self.out_projection(out), attn


class ReformerLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None, causal=False, bucket_size=4, n_hashes=4):
        super().__init__()
        self.bucket_size = bucket_size
        self.attn = LSHSelfAttention(
            dim=d_model,
            heads=n_heads,
            bucket_size=bucket_size,
            n_hashes=n_hashes,
            causal=causal
        )

    def fit_length(self, queries):
        # inside reformer: assert N % (bucket_size * 2) == 0
        B, N, C = queries.shape
        if N % (self.bucket_size * 2) == 0:
            return queries
        else:
            # fill the time series
            fill_len = (self.bucket_size * 2) - (N % (self.bucket_size * 2))
            return torch.cat([queries, torch.zeros([B, fill_len, C]).to(queries.device)], dim=1)

    def forward(self, queries, keys, values, attn_mask, tau, delta):
        # in Reformer: defalut queries=keys
        B, N, C = queries.shape
        queries = self.attn(self.fit_length(queries))[:, :N, :]
        return queries, None


class TwoStageAttentionLayer(nn.Module):
    '''
    The Two Stage Attention (TSA) Layer
    input/output shape: [batch_size, Data_dim(D), Seg_num(L), d_model]
    '''

    def __init__(self, configs,
                 seg_num, factor, d_model, n_heads, d_ff=None, dropout=0.1):
        super(TwoStageAttentionLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.time_attention = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                           output_attention=False), d_model, n_heads)
        self.dim_sender = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                       output_attention=False), d_model, n_heads)
        self.dim_receiver = AttentionLayer(FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                                         output_attention=False), d_model, n_heads)
        self.router = nn.Parameter(torch.randn(seg_num, factor, d_model))

        self.dropout = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)

        self.MLP1 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))
        self.MLP2 = nn.Sequential(nn.Linear(d_model, d_ff),
                                  nn.GELU(),
                                  nn.Linear(d_ff, d_model))

    def forward(self, x, attn_mask=None, tau=None, delta=None):
        # Cross Time Stage: Directly apply MSA to each dimension
        batch = x.shape[0]
        time_in = rearrange(x, 'b ts_d seg_num d_model -> (b ts_d) seg_num d_model')
        time_enc, attn = self.time_attention(
            time_in, time_in, time_in, attn_mask=None, tau=None, delta=None
        )
        dim_in = time_in + self.dropout(time_enc)
        dim_in = self.norm1(dim_in)
        dim_in = dim_in + self.dropout(self.MLP1(dim_in))
        dim_in = self.norm2(dim_in)

        # Cross Dimension Stage: use a small set of learnable vectors to aggregate and distribute messages to build the D-to-D connection
        dim_send = rearrange(dim_in, '(b ts_d) seg_num d_model -> (b seg_num) ts_d d_model', b=batch)
        batch_router = repeat(self.router, 'seg_num factor d_model -> (repeat seg_num) factor d_model', repeat=batch)
        dim_buffer, attn = self.dim_sender(batch_router, dim_send, dim_send, attn_mask=None, tau=None, delta=None)
        dim_receive, attn = self.dim_receiver(dim_send, dim_buffer, dim_buffer, attn_mask=None, tau=None, delta=None)
        dim_enc = dim_send + self.dropout(dim_receive)
        dim_enc = self.norm3(dim_enc)
        dim_enc = dim_enc + self.dropout(self.MLP2(dim_enc))
        dim_enc = self.norm4(dim_enc)

        final_out = rearrange(dim_enc, '(b seg_num) ts_d d_model -> b ts_d seg_num d_model', b=batch)

        return final_out
