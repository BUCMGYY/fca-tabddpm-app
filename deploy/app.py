"""
FCA-TabDDPM 表格数据增强平台
==============================
包含完整模型定义、训练、推理和评估功能

部署步骤:
    1. pip install -r requirements.txt
    2. 将 best_model.pt 放在项目根目录
    3. streamlit run app.py
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats as sp_stats
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import io
import time
import os
import math

# ============================================================
# 页面配置
# ============================================================
st.set_page_config(
    page_title="FCA-TabDDPM 表格数据增强平台",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 1.8rem; font-weight: bold; color: #1a1a2e;
        text-align: center; padding: 0.8rem 0;
        border-bottom: 3px solid #16213e; margin-bottom: 1rem;
    }
    .info-box {
        background-color: #f0f2f6; padding: 0.8rem; border-radius: 8px;
        border-left: 4px solid #4e79a7; margin: 0.5rem 0; font-size: 0.9rem;
    }
    .success-box {
        background-color: #d4edda; padding: 0.8rem; border-radius: 8px;
        border-left: 4px solid #28a745; margin: 0.5rem 0; font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================
# 一、模型定义
# ============================================================

class FeatureEmbedding(nn.Module):
    """特征嵌入层"""
    def __init__(self, n_cont, n_binary, cat_dims, d_f):
        super().__init__()
        self.n_cont = n_cont
        self.n_binary = n_binary
        self.cat_dims = cat_dims
        if n_cont > 0:
            self.cont_embed = nn.Linear(1, d_f)
        self.binary_embeds = nn.ModuleList([nn.Embedding(2, d_f) for _ in range(n_binary)])
        self.cat_embeds = nn.ModuleList([nn.Embedding(dim, d_f) for dim in cat_dims])

    def forward(self, x_cont, x_binary, x_cat):
        embeds = []
        if self.n_cont > 0 and x_cont is not None:
            for i in range(self.n_cont):
                embeds.append(self.cont_embed(x_cont[:, i:i+1]))
        for i, emb in enumerate(self.binary_embeds):
            embeds.append(emb(x_binary[:, i].long()))
        for i, emb in enumerate(self.cat_embeds):
            embeds.append(emb(x_cat[:, i].long()))
        return torch.stack(embeds, dim=1)


class FCABlock(nn.Module):
    """特征交叉注意力块"""
    def __init__(self, d_f, n_heads, dropout=0.1):
        super().__init__()
        self.d_f = d_f
        self.n_heads = n_heads
        self.d_k = d_f // n_heads
        self.W_Q = nn.Linear(d_f, d_f)
        self.W_K = nn.Linear(d_f, d_f)
        self.W_V = nn.Linear(d_f, d_f)
        self.W_O = nn.Linear(d_f, d_f)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_f)

    def forward(self, H):
        B, Fn, d = H.shape
        residual = H
        Q = self.W_Q(H).view(B, Fn, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_K(H).view(B, Fn, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_V(H).view(B, Fn, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, Fn, self.d_f)
        out = self.W_O(out)
        return self.layer_norm(residual + out), attn


class FCADenoiser(nn.Module):
    """FCA去噪网络"""
    def __init__(self, n_cont, n_binary, cat_dims, n_classes,
                 d_f=64, n_layers=3, n_heads=4, d_ff=256, dropout=0.1):
        super().__init__()
        self.n_cont = n_cont
        self.n_binary = n_binary
        self.cat_dims = cat_dims
        self.d_f = d_f
        self.feature_embed = FeatureEmbedding(n_cont, n_binary, cat_dims, d_f)
        self.time_embed = nn.Sequential(
            nn.Linear(d_f, d_f), nn.SiLU(), nn.Linear(d_f, d_f), nn.SiLU())
        self.class_embed = nn.Embedding(n_classes, d_f)
        self.fca_blocks = nn.ModuleList(
            [FCABlock(d_f, n_heads, dropout) for _ in range(n_layers)])
        self.ff_blocks = nn.ModuleList([nn.Sequential(
            nn.Linear(d_f, d_ff), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_f), nn.Dropout(dropout)) for _ in range(n_layers)])
        self.ff_norms = nn.ModuleList([nn.LayerNorm(d_f) for _ in range(n_layers)])
        if n_cont > 0:
            self.cont_head = nn.Linear(d_f, 1)
        self.binary_heads = nn.ModuleList([nn.Linear(d_f, 1) for _ in range(n_binary)])
        self.cat_heads = nn.ModuleList([nn.Linear(d_f, dim) for dim in cat_dims])

    def get_timestep_embedding(self, t, d):
        half = d // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
        args = t[:, None].float() * freqs[None]
        return torch.cat([torch.cos(args), torch.sin(args)], dim=-1)

    def forward(self, x_cont, x_binary, x_cat, t, y):
        H = self.feature_embed(x_cont, x_binary, x_cat)
        t_emb = self.time_embed(self.get_timestep_embedding(t, self.d_f))
        y_emb = self.class_embed(y)
        H = H + (t_emb + y_emb).unsqueeze(1)
        attns = []
        for fca, ff, ff_norm in zip(self.fca_blocks, self.ff_blocks, self.ff_norms):
            H, attn = fca(H)
            H = ff_norm(H + ff(H))
            attns.append(attn)
        outputs = {}
        idx = 0
        if self.n_cont > 0:
            outputs['cont'] = self.cont_head(H[:, idx:idx+self.n_cont])
            idx += self.n_cont
        bin_logits = [head(H[:, idx+i]).squeeze(-1) for i, head in enumerate(self.binary_heads)]
        outputs['binary'] = torch.stack(bin_logits, dim=1) if bin_logits else None
        idx += self.n_binary
        outputs['cat'] = [head(H[:, idx+i]) for i, head in enumerate(self.cat_heads)]
        outputs['attentions'] = attns
        return outputs


class FCATabDDPM:
    """FCA-TabDDPM完整扩散模型"""
    def __init__(self, n_cont, n_binary, cat_dims, n_classes,
                 d_f=64, n_layers=3, n_heads=4, d_ff=256, T=1000, device='cpu'):
        self.n_cont = n_cont
        self.n_binary = n_binary
        self.cat_dims = cat_dims
        self.n_classes = n_classes
        self.T = T
        self.device = device
        self.denoiser = FCADenoiser(
            n_cont, n_binary, cat_dims, n_classes, d_f, n_layers, n_heads, d_ff
        ).to(device)
        self.betas = torch.linspace(1e-4, 0.02, T).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def load_pretrained(self, path_or_bytes):
        state_dict = torch.load(path_or_bytes, map_location=self.device, weights_only=False)
        self.denoiser.load_state_dict(state_dict)
        self.denoiser.eval()

    def partial_load_pretrained(self, path_or_bytes):
        """部分加载预训练权重（特征数不同时自动跳过不匹配的参数）"""
        state_dict = torch.load(path_or_bytes, map_location=self.device, weights_only=False)
        model_dict = self.denoiser.state_dict()
        matched = {k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(matched)
        self.denoiser.load_state_dict(model_dict)
        return len(matched), len(model_dict)

    def save(self, path):
        torch.save(self.denoiser.state_dict(), path)

    def train_model(self, X_cont, X_binary, X_cat, y,
                    epochs=1000, batch_size=128, lr=1e-3, progress_callback=None):
        self.denoiser.train()
        optimizer = torch.optim.AdamW(self.denoiser.parameters(), lr=lr)
        cont_t = torch.FloatTensor(X_cont if X_cont is not None and len(X_cont.shape) > 1 and X_cont.shape[1] > 0
                                    else np.zeros((len(y), 0)))
        dataset = TensorDataset(cont_t, torch.LongTensor(X_binary),
                                torch.LongTensor(X_cat), torch.LongTensor(y))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

        for epoch in range(epochs):
            total_loss = 0
            for batch in loader:
                xc, xb, xm, yb = [b.to(self.device) for b in batch]
                B = yb.shape[0]
                t = torch.randint(0, self.T, (B,), device=self.device)
                ab = self.alpha_bars[t]
                loss = torch.tensor(0.0, device=self.device)

                # 连续变量加噪
                if self.n_cont > 0 and xc.shape[1] > 0:
                    noise = torch.randn_like(xc)
                    xc_n = ab.sqrt().unsqueeze(1) * xc + (1-ab).sqrt().unsqueeze(1) * noise
                else:
                    xc_n = xc

                # 二分类变量加噪
                bt = self.betas[t]
                xb_n = xb.clone()
                flip = torch.rand_like(xb.float()) < bt.unsqueeze(1)
                xb_n[flip] = 1 - xb_n[flip]

                # 多分类变量加噪
                xm_n = xm.clone()
                for j in range(len(self.cat_dims)):
                    K = self.cat_dims[j]
                    rep = torch.rand(B, device=self.device) < bt
                    xm_n[rep, j] = torch.randint(0, K, (rep.sum(),), device=self.device)

                outputs = self.denoiser(xc_n, xb_n, xm_n, t, yb)

                if self.n_cont > 0 and xc.shape[1] > 0:
                    loss = loss + F.mse_loss(outputs['cont'].squeeze(-1), noise)
                if outputs['binary'] is not None:
                    loss = loss + F.binary_cross_entropy_with_logits(outputs['binary'], xb.float())
                for j, logits in enumerate(outputs['cat']):
                    loss = loss + F.cross_entropy(logits, xm[:, j])

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.denoiser.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg = total_loss / max(len(loader), 1)
            if progress_callback:
                progress_callback(epoch, epochs, avg)

    @torch.no_grad()
    def sample(self, n_samples, y_label):
        self.denoiser.eval()
        xc = torch.randn(n_samples, max(self.n_cont, 1), device=self.device)
        xb = torch.randint(0, 2, (n_samples, self.n_binary), device=self.device)
        xm = torch.zeros(n_samples, len(self.cat_dims), dtype=torch.long, device=self.device)
        for j, K in enumerate(self.cat_dims):
            xm[:, j] = torch.randint(0, K, (n_samples,))
        y = torch.full((n_samples,), y_label, dtype=torch.long, device=self.device)

        for t_val in reversed(range(self.T)):
            t = torch.full((n_samples,), t_val, dtype=torch.long, device=self.device)
            out = self.denoiser(xc, xb, xm, t, y)
            if self.n_cont > 0:
                bt = self.betas[t_val]
                at = self.alphas[t_val]
                abt = self.alpha_bars[t_val]
                xc = (1/at.sqrt()) * (xc - bt/(1-abt).sqrt() * out['cont'].squeeze(-1))
                if t_val > 0:
                    xc += bt.sqrt() * torch.randn_like(xc)
            if out['binary'] is not None:
                xb = (torch.sigmoid(out['binary']) > 0.5).long()
            for j, logits in enumerate(out['cat']):
                xm[:, j] = logits.argmax(dim=-1)

        return {'cont': xc.cpu().numpy() if self.n_cont > 0 else None,
                'binary': xb.cpu().numpy(), 'cat': xm.cpu().numpy(), 'y': y.cpu().numpy()}

    @torch.no_grad()
    def get_attention_map(self, y_label, n_samples=50):
        self.denoiser.eval()
        xc = torch.randn(n_samples, max(self.n_cont, 1), device=self.device)
        xb = torch.randint(0, 2, (n_samples, self.n_binary), device=self.device)
        xm = torch.zeros(n_samples, len(self.cat_dims), dtype=torch.long, device=self.device)
        for j, K in enumerate(self.cat_dims):
            xm[:, j] = torch.randint(0, K, (n_samples,))
        y = torch.full((n_samples,), y_label, dtype=torch.long, device=self.device)
        t = torch.full((n_samples,), self.T // 2, dtype=torch.long, device=self.device)
        out = self.denoiser(xc, xb, xm, t, y)
        return out['attentions'][-1].mean(dim=(0, 1)).cpu().numpy()


# ============================================================
# 二、工具函数
# ============================================================

def detect_feature_types(df, target_col):
    config = {}
    for col in df.columns:
        if col == target_col:
            continue
        nu = df[col].nunique()
        if df[col].dtype in ['float64', 'float32'] or nu > 20:
            config[col] = {'type': '连续型', 'nunique': nu}
        elif nu == 2:
            config[col] = {'type': '二分类', 'nunique': nu}
        else:
            config[col] = {'type': '多分类', 'nunique': nu}
    return config


def prepare_model_data(df, target_col, feature_config):
    cont_cols = [c for c, cfg in feature_config.items() if cfg['type'] == '连续型']
    bin_cols = [c for c, cfg in feature_config.items() if cfg['type'] == '二分类']
    cat_cols = [c for c, cfg in feature_config.items() if cfg['type'] == '多分类']

    X_cont = df[cont_cols].values.astype(np.float32) if cont_cols else np.zeros((len(df), 0), dtype=np.float32)

    bin_maps = {}
    X_binary = np.zeros((len(df), len(bin_cols)), dtype=np.int64)
    for i, col in enumerate(bin_cols):
        uv = sorted(df[col].unique())
        m = {v: idx for idx, v in enumerate(uv)}
        X_binary[:, i] = df[col].map(m).values
        bin_maps[col] = {idx: v for v, idx in m.items()}

    cat_dims, cat_maps = [], {}
    X_cat = np.zeros((len(df), len(cat_cols)), dtype=np.int64)
    for i, col in enumerate(cat_cols):
        uv = sorted(df[col].unique())
        m = {v: idx for idx, v in enumerate(uv)}
        X_cat[:, i] = df[col].map(m).values
        cat_dims.append(len(uv))
        cat_maps[col] = {idx: v for v, idx in m.items()}

    y_unique = sorted(df[target_col].unique())
    y_map = {v: idx for idx, v in enumerate(y_unique)}
    y = df[target_col].map(y_map).values.astype(np.int64)

    return (X_cont, X_binary, X_cat, y,
            cont_cols, bin_cols, cat_cols, cat_dims,
            y_unique, y_map, bin_maps, cat_maps)


def samples_to_df(result, cont_cols, bin_cols, cat_cols, target_col,
                   y_unique, y_label, bin_maps, cat_maps):
    df = pd.DataFrame()
    if result['cont'] is not None and len(cont_cols) > 0:
        for i, col in enumerate(cont_cols):
            df[col] = result['cont'][:, i].round(2)
    for i, col in enumerate(bin_cols):
        vals = result['binary'][:, i]
        if col in bin_maps:
            df[col] = [bin_maps[col].get(int(v), v) for v in vals]
        else:
            df[col] = vals
    for i, col in enumerate(cat_cols):
        vals = result['cat'][:, i]
        if col in cat_maps:
            df[col] = [cat_maps[col].get(int(v), v) for v in vals]
        else:
            df[col] = vals
    df[target_col] = y_unique[y_label]
    return df


def compute_jsd(p, q, is_binary=True):
    eps = 1e-10
    if is_binary:
        P = np.array([p + eps, 1 - p + eps])
        Q = np.array([q + eps, 1 - q + eps])
    else:
        P, Q = p + eps, q + eps
    P, Q = P / P.sum(), Q / Q.sum()
    M = 0.5 * (P + Q)
    return float(0.5 * np.sum(P * np.log(P/M)) + 0.5 * np.sum(Q * np.log(Q/M)))


def evaluate_quality(real_df, syn_df, feature_config, target_col):
    results = {}
    jsd_list, jsd_detail = [], []
    for col, cfg in feature_config.items():
        if col not in syn_df.columns:
            continue
        if cfg['type'] == '二分类':
            jsd = compute_jsd(real_df[col].mean(), syn_df[col].mean(), True)
        elif cfg['type'] == '多分类':
            cats = sorted(set(real_df[col].unique()) | set(syn_df[col].unique()))
            r = real_df[col].value_counts().reindex(cats, fill_value=0).values.astype(float)
            s = syn_df[col].value_counts().reindex(cats, fill_value=0).values.astype(float)
            jsd = compute_jsd(r, s, False)
        else:
            continue
        jsd_list.append(jsd)
        jsd_detail.append({'特征': col, '类型': cfg['type'], 'JSD': jsd})
    results['avg_jsd'] = np.mean(jsd_list) if jsd_list else 0
    results['jsd_detail'] = pd.DataFrame(jsd_detail)

    wd_list = []
    for col, cfg in feature_config.items():
        if cfg['type'] == '连续型' and col in syn_df.columns and col in real_df.columns:
            r_vals = pd.to_numeric(real_df[col], errors='coerce').dropna()
            s_vals = pd.to_numeric(syn_df[col], errors='coerce').dropna()
            if len(r_vals) > 0 and len(s_vals) > 0:
                wd_list.append({'特征': col, 'WD': sp_stats.wasserstein_distance(r_vals, s_vals)})
    results['avg_wd'] = np.mean([w['WD'] for w in wd_list]) if wd_list else 0
    results['wd_detail'] = pd.DataFrame(wd_list)

    num_cols = [c for c in feature_config if c in syn_df.columns and c in real_df.columns]
    if len(num_cols) >= 2:
        try:
            rc = np.nan_to_num(real_df[num_cols].apply(pd.to_numeric, errors='coerce').corr().values, nan=0)
            sc = np.nan_to_num(syn_df[num_cols].apply(pd.to_numeric, errors='coerce').corr().values, nan=0)
            mask = np.triu(np.ones_like(rc, dtype=bool), k=1)
            results['pcd'] = float(np.abs(rc[mask] - sc[mask]).sum())
        except:
            results['pcd'] = 0
    else:
        results['pcd'] = 0

    bin_cols = [c for c, cfg in feature_config.items() if cfg['type'] == '二分类' and c in syn_df.columns]
    if len(bin_cols) >= 2:
        try:
            rb = real_df[bin_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
            sb = syn_df[bin_cols].apply(pd.to_numeric, errors='coerce').fillna(0).values.astype(float)
            co_r = (rb.T @ rb) / rb.shape[0]
            co_s = (sb.T @ sb) / sb.shape[0]
            m = np.triu(np.ones_like(co_r, dtype=bool), k=1)
            results['cooccurrence_mae'] = float(np.abs(co_r[m] - co_s[m]).mean())
            results['co_real'], results['co_syn'], results['bin_cols'] = co_r, co_s, bin_cols
        except:
            results['cooccurrence_mae'] = 0
    else:
        results['cooccurrence_mae'] = 0
    return results


# ============================================================
# 三、Session State
# ============================================================
for k, v in {'data_uploaded': False, 'df_raw': None, 'target_col': None,
             'feature_config': {}, 'model_trained': False, 'model': None,
             'synthetic_df': None, 'class_counts': {},
             'cont_cols': [], 'bin_cols': [], 'cat_cols': [], 'cat_dims_list': [],
             'y_unique': [], 'y_map': {}, 'bin_maps': {}, 'cat_maps': {}}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ============================================================
# 四、页面主体
# ============================================================
st.markdown('<div class="main-header">🧬 FCA-TabDDPM 表格数据增强平台</div>', unsafe_allow_html=True)
tab1, tab2, tab3, tab4 = st.tabs(["📤 数据上传与配置", "🔧 模型训练与生成", "📊 生成质量评估", "📖 使用说明"])


# --- Tab 1 ---
with tab1:
    st.markdown("### 📤 上传数据")
    uploaded_file = st.file_uploader("上传CSV格式的表格数据", type=['csv'])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.session_state['df_raw'] = df
        st.session_state['data_uploaded'] = True
        st.success(f"✅ 数据加载成功：{df.shape[0]} 行 × {df.shape[1]} 列")
        with st.expander("📋 数据预览", expanded=True):
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
        st.markdown("---")
        target_col = st.selectbox("🎯 选择目标变量", df.columns.tolist(), index=len(df.columns)-1)
        st.session_state['target_col'] = target_col
        cc = df[target_col].value_counts().sort_index()
        st.session_state['class_counts'] = cc.to_dict()
        ca, cb = st.columns([1, 1.5])
        with ca:
            st.markdown("#### 类别分布")
            dd = pd.DataFrame({'类别': cc.index.astype(str), '样本量': cc.values,
                                '占比': (cc.values/cc.sum()*100).round(1).astype(str)+'%'})
            st.dataframe(dd, use_container_width=True, hide_index=True)
            st.metric("不平衡比", f"{cc.max()/cc.min():.1f} : 1")
        with cb:
            fig = px.bar(x=cc.index.astype(str), y=cc.values, color=cc.index.astype(str),
                         labels={'x':'类别','y':'样本量'}, template="plotly_white", height=350)
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.markdown("### 🔍 特征类型识别")
        fc = detect_feature_types(df, target_col)
        td = pd.DataFrame([{'特征名':c,'识别类型':cfg['type'],'唯一值数':cfg['nunique']} for c,cfg in fc.items()])
        edited = st.data_editor(td, column_config={"识别类型": st.column_config.SelectboxColumn(
            options=["连续型","二分类","多分类"], required=True)},
            use_container_width=True, hide_index=True, num_rows="fixed")
        for _, row in edited.iterrows():
            fc[row['特征名']]['type'] = row['识别类型']
        st.session_state['feature_config'] = fc
        nc = sum(1 for c in fc.values() if c['type']=='连续型')
        nb = sum(1 for c in fc.values() if c['type']=='二分类')
        nm = sum(1 for c in fc.values() if c['type']=='多分类')
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("总特征",len(fc)); c2.metric("连续型",nc); c3.metric("二分类",nb); c4.metric("多分类",nm)
        st.markdown('<div class="success-box">✅ 配置完成，前往「模型训练与生成」</div>', unsafe_allow_html=True)


# --- Tab 2 ---
with tab2:
    if not st.session_state['data_uploaded']:
        st.info("💡 请先上传数据")
    else:
        df = st.session_state['df_raw']
        target_col = st.session_state['target_col']
        feature_config = st.session_state['feature_config']
        class_counts = st.session_state['class_counts']

        st.markdown("### 🔧 训练设置")
        ct1, ct2 = st.columns(2)
        with ct1:
            train_mode = st.radio("训练模式", ["预训练权重微调（推荐）","从头训练"])
            pt_file = None
            if "预训练" in train_mode:
                pt_file = st.file_uploader("上传 best_model.pt", type=['pt','pth'])
        with ct2:
            epochs = st.slider("训练轮数", 100, 5000, 1000, 100)
            lr = st.select_slider("学习率", [1e-4,3e-4,5e-4,1e-3,3e-3], value=1e-3)
            batch_size = st.selectbox("批量大小", [32,64,128,256], index=2)
        with st.expander("⚙️ 高级超参数"):
            h1,h2,h3 = st.columns(3)
            with h1: d_f=st.selectbox("d_f",[32,64,128],index=1); n_layers=st.selectbox("L",[1,2,3,4,5],index=2)
            with h2: n_heads=st.selectbox("h",[2,4,8],index=1); d_ff=st.selectbox("d_ff",[128,256,512],index=1)
            with h3: T=st.selectbox("T",[500,1000,2000],index=1)

        st.markdown("---")
        if st.button("🚀 开始训练", type="primary", use_container_width=True):
            Xc,Xb,Xm,y,ccols,bcols,mcols,cdims,yuniq,ymap,bmaps,cmaps = prepare_model_data(df,target_col,feature_config)
            st.session_state.update({'cont_cols':ccols,'bin_cols':bcols,'cat_cols':mcols,
                'cat_dims_list':cdims,'y_unique':yuniq,'y_map':ymap,'bin_maps':bmaps,'cat_maps':cmaps})
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            model = FCATabDDPM(len(ccols),len(bcols),cdims,len(yuniq),d_f,n_layers,n_heads,d_ff,T,device)

            if "预训练" in train_mode and pt_file:
                try:
                    n_m, n_t = model.partial_load_pretrained(io.BytesIO(pt_file.read()))
                    st.info(f"📦 预训练权重：{n_m}/{n_t} 参数匹配并加载")
                except Exception as e:
                    st.warning(f"⚠️ 加载失败：{e}，从头训练")

            pb = st.progress(0); st_text = st.empty(); losses = []
            def cb(ep, tot, loss):
                pb.progress((ep+1)/tot); losses.append(loss)
                if (ep+1) % max(1,tot//20)==0: st_text.text(f"Epoch {ep+1}/{tot} | Loss: {loss:.4f}")
            model.train_model(Xc,Xb,Xm,y,epochs,batch_size,lr,cb)
            st.session_state['model']=model; st.session_state['model_trained']=True
            st.markdown('<div class="success-box">✅ 训练完成</div>', unsafe_allow_html=True)
            if losses:
                st.plotly_chart(px.line(y=losses,labels={'x':'Epoch','y':'Loss'},template="plotly_white",height=300), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 增补策略")
        strategy = st.selectbox("策略", ["完全平衡（补齐至最大类）","按比例上采样","指定最小样本量","自定义每类数量"])
        mx = max(class_counts.values()); gp = {}
        if strategy.startswith("完全"):
            for c,n in class_counts.items(): gp[c]=max(0,mx-n)
        elif strategy.startswith("按比例"):
            mul = st.slider("倍数",1,5,2)
            for c,n in class_counts.items(): gp[c]=n*(mul-1)
        elif strategy.startswith("指定"):
            mt = st.number_input("最小样本量",10,mx*5,mx,10)
            for c,n in class_counts.items(): gp[c]=max(0,mt-n)
        else:
            cols=st.columns(min(len(class_counts),4))
            for i,(c,n) in enumerate(class_counts.items()):
                with cols[i%len(cols)]: gp[c]=st.number_input(f"{c}（现有{n}）",0,5000,max(0,mx-n),10,key=f"g_{c}")

        tg = sum(gp.values())
        if tg > 0:
            pf = pd.DataFrame([{'类别':str(c),'现有':n,'生成':gp.get(c,0),'增强后':n+gp.get(c,0)} for c,n in class_counts.items()])
            st.dataframe(pf, use_container_width=True, hide_index=True)
            if st.button("🔬 开始生成", type="primary", use_container_width=True):
                if not st.session_state['model_trained']:
                    st.warning("⚠️ 请先训练模型")
                else:
                    model=st.session_state['model']; ymap=st.session_state['y_map']; yuniq=st.session_state['y_unique']
                    ccols=st.session_state['cont_cols']; bcols=st.session_state['bin_cols']
                    mcols=st.session_state['cat_cols']; bmaps=st.session_state['bin_maps']; cmaps=st.session_state['cat_maps']
                    all_syn=[]; pg=st.progress(0); items=[(c,n) for c,n in gp.items() if n>0]
                    for i,(c,n) in enumerate(items):
                        lid = ymap.get(c, 0)
                        res = model.sample(n, lid)
                        all_syn.append(samples_to_df(res,ccols,bcols,mcols,target_col,yuniq,lid,bmaps,cmaps))
                        pg.progress((i+1)/len(items))
                    syn_all = pd.concat(all_syn, ignore_index=True)
                    st.session_state['synthetic_df'] = syn_all
                    st.markdown(f'<div class="success-box">✅ 生成 {len(syn_all)} 例</div>', unsafe_allow_html=True)
                    with st.expander("预览"): st.dataframe(syn_all.head(20), use_container_width=True, hide_index=True)
                    d1,d2 = st.columns(2)
                    with d1:
                        b=io.StringIO(); syn_all.to_csv(b,index=False,encoding='utf-8-sig')
                        st.download_button("📥 合成数据",b.getvalue().encode('utf-8-sig'),f"syn_{tg}.csv","text/csv",use_container_width=True)
                    with d2:
                        mg=pd.concat([df,syn_all],ignore_index=True); b2=io.StringIO()
                        mg.to_csv(b2,index=False,encoding='utf-8-sig')
                        st.download_button("📥 增强后数据",b2.getvalue().encode('utf-8-sig'),f"aug_{len(mg)}.csv","text/csv",use_container_width=True)


# --- Tab 3 ---
with tab3:
    if st.session_state['synthetic_df'] is None:
        st.info("💡 请先生成合成数据")
    else:
        syn_df=st.session_state['synthetic_df']; df=st.session_state['df_raw']
        tc=st.session_state['target_col']; fc=st.session_state['feature_config']
        st.markdown("### 📊 生成质量评估")
        scope = st.radio("范围",["全局","按类别"],horizontal=True)
        if scope=="全局":
            res = evaluate_quality(df, syn_df, fc, tc)
        else:
            cl = st.selectbox("类别", sorted(df[tc].unique()))
            rs, ss = df[df[tc]==cl], syn_df[syn_df[tc]==cl]
            res = evaluate_quality(rs, ss, fc, tc) if len(ss)>0 else None
        if res:
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("JSD ↓",f"{res['avg_jsd']:.6f}"); c2.metric("WD ↓",f"{res['avg_wd']:.4f}")
            c3.metric("PCD ↓",f"{res['pcd']:.2f}"); c4.metric("共现MAE ↓",f"{res['cooccurrence_mae']:.6f}")
            st.markdown("---")
            if len(res['jsd_detail'])>0:
                st.markdown("#### 各特征JSD")
                fig=px.bar(res['jsd_detail'].sort_values('JSD',ascending=False),x='特征',y='JSD',color='类型',
                    color_discrete_map={'二分类':'#4E79A7','多分类':'#E15759'},template="plotly_white",height=400)
                fig.update_layout(xaxis_tickangle=-45); st.plotly_chart(fig, use_container_width=True)
            if len(res['wd_detail'])>0:
                st.markdown("#### 连续变量WD")
                st.plotly_chart(px.bar(res['wd_detail'],x='特征',y='WD',color_discrete_sequence=['#76B7B2'],
                    template="plotly_white",height=350), use_container_width=True)
            if 'co_real' in res:
                st.markdown("#### 共现矩阵对比")
                bl=[c[:6] for c in res['bin_cols']]; cl2,cr2=st.columns(2)
                with cl2:
                    st.markdown("**真实数据**")
                    fr=go.Figure(data=go.Heatmap(z=res['co_real'],x=bl,y=bl,colorscale="Blues"))
                    fr.update_layout(height=500,template="plotly_white",xaxis_tickangle=-45)
                    st.plotly_chart(fr, use_container_width=True)
                with cr2:
                    st.markdown("**合成数据**")
                    fs=go.Figure(data=go.Heatmap(z=res['co_syn'],x=bl,y=bl,colorscale="Oranges"))
                    fs.update_layout(height=500,template="plotly_white",xaxis_tickangle=-45)
                    st.plotly_chart(fs, use_container_width=True)
            if st.session_state['model']:
                st.markdown("---"); st.markdown("#### 🧠 注意力热图")
                ac=st.selectbox("证型",sorted(df[tc].unique()),key='attn_c')
                ym=st.session_state.get('y_map',{})
                if ac in ym:
                    at=st.session_state['model'].get_attention_map(ym[ac])
                    fn=st.session_state['cont_cols']+st.session_state['bin_cols']+st.session_state['cat_cols']
                    sn=[n[:6] for n in fn]
                    fa=go.Figure(data=go.Heatmap(z=at,x=sn,y=sn,colorscale="RdBu_r"))
                    fa.update_layout(height=600,template="plotly_white",xaxis_tickangle=-45)
                    st.plotly_chart(fa, use_container_width=True)


# --- Tab 4 ---
with tab4:
    st.markdown("""
    ### 📖 使用说明
    #### 1. 数据上传与配置
    上传CSV数据 → 选择目标变量 → 确认特征类型（支持手动修正）
    #### 2. 模型训练与生成
    - **预训练微调**：上传 best_model.pt，自动匹配可迁移的参数，收敛更快
    - **从头训练**：完全基于用户数据训练
    - 四种增补策略：完全平衡 / 按比例 / 指定最小量 / 自定义
    #### 3. 生成质量评估
    JSD / WD / PCD / 共现MAE 四维评估 + 注意力可视化
    #### 4. 预训练权重说明
    预训练权重基于4421例颈椎病中医证型数据训练，包含38个特征（1连续+31二分类+6多分类）的嵌入参数。
    微调模式下，与用户数据维度匹配的参数将被加载，不匹配的参数随机初始化。
    ---
    *中国中医科学院望京医院*
    """)

st.markdown("---")
st.markdown('<div style="text-align:center;color:#888;font-size:0.85rem;">FCA-TabDDPM 表格数据增强平台 | 中国中医科学院望京医院</div>', unsafe_allow_html=True)
