import type { OpNode, OpKind } from "../types";

export const SIMPLE_FORMULA: Partial<Record<OpKind,string>> = {
  norm:String.raw`y=\operatorname{Norm}(x)`,linear:String.raw`y=xW^{\mathsf T}`,split:String.raw`(a,b,\ldots)=\operatorname{Split}(x)`,rope:String.raw`q'=\operatorname{RoPE}(q,\mathrm{position})`,matmul:String.raw`y=a\,b^{\mathsf T}`,scale:String.raw`y=x/\sqrt{d_h}`,mask:String.raw`y=x+\mathrm{mask}`,softmax:String.raw`p=\operatorname{softmax}(x)`,activation:String.raw`y=\bar g\odot\sigma(\bar g)\,\bar u`,route:String.raw`I=\operatorname{TopK}(\mathrm{score}(x))`,cache:String.raw`\mathrm{KV}[\mathrm{slot}]\leftarrow(k)`,add:String.raw`y=x+f(x)`,io:String.raw`y=x`,
};

export const FORMULA_NOTE: Partial<Record<OpKind,string>> = {
  norm:"把每个 token 的向量缩放到稳定范围;shape 不变。",linear:"W 是当前模块绑定的权重;最后一维由 W 的输出维决定。",split:"只切分最后一维,不做数值计算,也没有权重。",rope:"position 决定旋转角度;这里只旋转 q 的 rope 64 维与 k_pe。",matmul:"沿共同的 head_dim 相乘并求和。",scale:"量化/缩放只改数值表示,不改变语义。",mask:"不可见位置加 −∞,softmax 后概率变为 0。",softmax:"把每行 score 转为和为 1 的概率;sink 折进分母。",activation:"门控或逐元素非线性;不混合 token。",route:"只选择去哪里计算;Top-K 本身不生成专家输出。",cache:"slot 由 runtime 提供,权重不参与。",add:"残差支路与计算支路逐元素相加,shape 必须一致。",io:"这是数据入口或运行时元数据,不执行可训练计算。",
};

export type FormulaTerm = readonly [symbol:string,meaning:string];
export type FormulaStep = { title:string; formula:string; explanation:string };

export const FORMULA_TERMS_BY_KIND: Record<OpKind,readonly FormulaTerm[]> = {
  io:[["x","输入"],["y","输出"]],
  norm:[["x","输入向量"],["y","归一化输出"],["H","归一化维度"],["γ","可训练缩放权重"],["ε","数值稳定项"],["RMS(x)","均方根"]],
  linear:[["x","输入张量"],["W","投影权重"],["y","线性投影输出"]],
  split:[["x","待切分张量"],["a,b,…","沿最后一维得到的输出"]],
  rope:[["q / k","Q 或 K 向量"],["p","token position"],["dᵣ","参与旋转的维度"],["θ","旋转角基数"]],
  matmul:[["a","左输入张量"],["b","右输入张量"],["y","矩阵乘输出"]],
  scale:[["x","输入"],["q_scale","FP8 组量化 scale"],["y","缩放后张量"]],
  mask:[["x","原始 score"],["M","causal / padding mask"],["y","mask 后 score"]],
  softmax:[["x","输入 score"],["sink","每 head 可学习标量"],["p","归一化概率"]],
  activation:[["g","gate 分支"],["u","up 分支"],["c","clamp limit = 10"],["y","输出"]],
  route:[["s","路由分数"],["K","选择数量"],["I / 𝓔","选中的 token 或专家集合"]],
  cache:[["k","写入 cache 的张量"],["slot","物理 cache 位置"]],
  add:[["x","residual 分支"],["f(x)","当前计算分支"],["y","逐元素相加结果"]],
};

export const FORMULA_TERMS_BY_ID: Partial<Record<string,readonly FormulaTerm[]>> = {
  "input":[["Xₗ","层输入 hidden_states"],["T","token 数"],["C_hc","hc_mult = 4"]],
  "hcpre-attn":[["x","4 通道输入 · [T,4,H]"],["x_flat","展平 · [T,24576]"],["mixes","hc_fn 输出 · [T,8]"],["pre_i / post_i","第 i 通道 pre/post σ 门"],["y","归约输出 · [T,H]"],["ε_hc","hc_eps = 10⁻⁶"]],
  "hcpre-mlp":[["x","4 通道输入 · [T,4,H]"],["x_flat","展平 · [T,24576]"],["mixes","hc_fn 输出 · [T,8]"],["pre_i / post_i","第 i 通道 pre/post σ 门"],["y","归约输出 · [T,H]"],["ε_hc","hc_eps = 10⁻⁶"]],
  "norm1":[["x","归约后 hidden · [B,S,H]"],["y","normalized hidden"],["H","hidden_size = 6144"],["γ","input_layernorm.weight"],["ε","rms_norm_eps = 10⁻⁵"]],
  "norm2":[["x","归约后 hidden · [B,S,H]"],["y","normalized hidden Û"],["H","hidden_size = 6144"],["γpost","post_attention_layernorm.weight"],["ε","rms_norm_eps = 10⁻⁵"]],
  "hcpost1":[["x","子块输出 · [T,H]"],["post","post 门 · [T,4]"],["residual","4 通道残差 · [T,4,H]"],["m_hc","hc_magnitude = 2.0"],["y","更新后的 4 通道 hidden"]],
  "hcpost2":[["x","子块输出 · [T,H]"],["post","post 门 · [T,4]"],["residual","4 通道残差 · [T,4,H]"],["m_hc","hc_magnitude = 2.0"],["y","层输出 · [T,4,H]"]],
  "aproj":[["X̂","norm 后 hidden · [B,S,H]"],["W","fused_qkv_a_proj(运行时融合)"],["q_a","q lora · [B,S,2048]"],["kv_a","压缩向量 · [B,S,512]"],["k_pe","共享 rope key · [B,S,64]"]],
  "qan":[["q_a","q lora · [B,S,2048]"],["q̃","归一化后 q lora"],["γq","q_a_layernorm.weight"],["ε","10⁻⁵"]],
  "qb":[["q̃","norm 后 q lora"],["N_h","query heads = 64"],["D_qk","qk_head_dim = 256"],["Q","query heads · [B,64,256]"]],
  "kvn":[["kv","kv lora · [B,S,512]"],["k̃","归一化压缩向量"],["k_pe","rope key · [B,S,64](不经 norm)"],["γkv","kv_a_layernorm.weight"]],
  "kvb":[["k̃","压缩向量(含 cache 历史)"],["K_nope","每头 nope key · [B,64,192]"],["V","每头 value · [B,64,256]"],["D_v","v_head_dim = 256"]],
  "rope":[["D_rope","rope 维 = 64"],["θ","rope_theta = 10⁷"],["p","token position"],["q'/k'","旋转后的向量"],["interleaved","checkpoint PTM 布局,is_neox_style=False"]],
  "cache":[["k̃","压缩向量 · [T,512]"],["k_pe","rope key · [T,64]"],["slot","物理 cache 位置"],["fp8_ds_mla","量化部署时的压缩 cache 布局(默认 BF16)"]],
  "iwqb":[["q̃","norm 后 q lora · [T,2048]"],["N_idx","index 头数 = 32"],["D_idx","index 头维 = 128"],["Qidx","索引查询 · [T,32,128]"]],
  "iwk":[["hidden","子块输入 · [T,6144]"],["k","索引 key · [T,128]"],["w","每头权重 · [T,32]"],["融合","一次 GEMM 输出 [128|32]"]],
  "ikn":[["k","索引 key"],["μ / σ","LayerNorm 的均值 / 方差"],["γ,β","k_norm.weight / bias"],["ε","10⁻⁶"],["LayerNorm","区别于主干的 RMSNorm"]],
  "iquant":[["q","Qidx 向量"],["q_scale","每 128 元素一组的 FP8 scale(ue8m0)"],["w'","折叠 q_scale/√128/√32 后的权重"],["√128","softmax_scale = D_idx^(-1/2)"],["√32","头数归一 N_idx^(-1/2)"]],
  "iscore":[["w'","折叠后的每头权重"],["Qidx","FP8 索引查询"],["k_j","第 j 个 token 的索引 key"],["I_t","token t 的 Top-2048 集合"],["buffer","模型级共享 topk_indices_buffer"]],
  "ishared":[["buffer","最近前驱 full 层写入的索引"],["I_t","本层直接复用的候选集合"],["57","shared 层数量"]],
  "qk":[["Q","query heads · [B,64,256]"],["K","候选 K(含 k_pe)"],["D_qk","qk_head_dim = 256"],["S","候选内 scores"],["M","causal mask"]],
  "sink":[["S","候选内 scores"],["P","注意力概率"],["sink_h","第 h head 的 sink(FP32,初始 0)"],["lse","注意力 log-sum-exp(内核内部量)"]],
  "pv":[["P","注意力概率"],["V","每头 value · [B,64,256]"],["O","上下文 heads"],["𝒮","Top-2048 候选集合"]],
  "gate":[["O","注意力 heads"],["g","linear_gate(hidden)"],["σ","sigmoid"],["elementwise","64×256 逐元素相乘"]],
  "oproj":[["Y","门控后 heads · [T,16384]"],["W_O","o_proj.weight"],["Y_attn","注意力子块输出 · [T,H]"]],
  "gateup":[["Û","norm 后 hidden"],["W_gate / W_up","L0 dense FFN 权重"],["H_ffn","intermediate_size = 18432"],["packed","[T,36864]"]],
  "act":[["g","gate 分支"],["u","up 分支"],["silu","silu(x)=x·σ(x)"],["无 clamp","dense/shared 不截断"]],
  "down":[["z","激活输出"],["W_down","down_proj.weight"],["Y_ffn","FFN 输出 · [T,H]"]],
  "router":[["Û","norm 后 hidden"],["W_g","gate.weight(FP32)"],["r","router_logits · [T,256]"],["b","e_score_correction_bias → expert_bias"]],
  "select":[["s","σ(r) 打分"],["K","每 token 专家数 = 8"],["s_route","routed_scaling_factor = 2.827"],["ŵ_e","专家 e 的混合权重"],["bias","只影响选择,不进权重"]],
  "experts":[["Û","专家输入"],["E_e","第 e 个专家函数"],["c","swiglu_limit = 10(仅 routed)"],["ŵ","Top-8 混合权重"],["Y_routed","加权归并输出"]],
  "shared":[["Û","共享输入"],["E_sh","shared expert(无 clamp)"],["Y_shared","恒在支路输出"]],
  "sum":[["Y_routed","routed 加权归并输出"],["Y_shared","shared 输出"],["Y_moe","MoE 输出"]],
};

export const FORMULA_STEPS_BY_ID: Partial<Record<string,readonly FormulaStep[]>> = {
  "select":[
    {title:"1 · 打分与选择",formula:String.raw`s=\sigma(r),\qquad \mathcal E=\operatorname{TopK}_{8}(s+b)`,explanation:"r 是 256 个 FP32 logits;σ 逐元素得到分数 s。b 是 expert_bias,只在选择 Top-8 时加上;n_group=topk_group=1,无分组约束。"},
    {title:"2 · 混合权重",formula:String.raw`\hat w_e=2.827\,\frac{s_e}{\sum_{j\in\mathcal E}s_j}`,explanation:"归一化用的是未加 bias 的 s;再乘 routed_scaling_factor=2.827。bias 不进入混合权重。"},
    {title:"3 · 专家输出归并",formula:String.raw`Y_{\mathrm{routed}}=\sum_{e\in\mathcal E}\hat w_eE_e(\hat U)`,explanation:"每个入选专家对 Û 计算 clamped SwiGLU(gate 截 max=10,up 截 ±10),按 ŵ 加权求和;shared expert 的输出在此之外单独相加。"},
  ],
};

export function formulaTerms(node:OpNode){
  return FORMULA_TERMS_BY_ID[node.id]??(node.latex?[]:FORMULA_TERMS_BY_KIND[node.kind]);
}

export const LATEX_BY_ID: Record<string,string> = {
  "input":String.raw`X_l^{\mathrm{ihc}}=\operatorname{broadcast}_{C_{hc}}(X_l),\quad C_{hc}=4`,
  "hcpre-attn":String.raw`\begin{aligned}x_{\mathrm{flat}}&=\operatorname{flatten}(x)\in\mathbb R^{T\times 24576}\\r&=\left(\operatorname{mean}(x_{\mathrm{flat}}^2)+\varepsilon_{\mathrm{rms}}\right)^{-1/2}\\m&=W_{\mathrm{hc}}\,x_{\mathrm{flat}}\cdot r\in\mathbb R^{T\times 8}\\\mathrm{pre}_i&=\sigma(m_i s_0+b_i)+\varepsilon_{hc},\quad \mathrm{post}_i=2\sigma(m_{4+i}s_1+b_{4+i})+\varepsilon_{hc}\\y&=\sum_{i=1}^{4}\mathrm{pre}_i\,x_i\end{aligned}`,
  "hcpre-mlp":String.raw`\begin{aligned}x_{\mathrm{flat}}&=\operatorname{flatten}(x)\in\mathbb R^{T\times 24576}\\r&=\left(\operatorname{mean}(x_{\mathrm{flat}}^2)+\varepsilon_{\mathrm{rms}}\right)^{-1/2}\\m&=W_{\mathrm{hc}}\,x_{\mathrm{flat}}\cdot r\in\mathbb R^{T\times 8}\\\mathrm{pre}_i&=\sigma(m_i s_0+b_i)+\varepsilon_{hc},\quad \mathrm{post}_i=2\sigma(m_{4+i}s_1+b_{4+i})+\varepsilon_{hc}\\y&=\sum_{i=1}^{4}\mathrm{pre}_i\,x_i\end{aligned}`,
  "norm1":String.raw`\operatorname{RMS}(x)=\sqrt{\tfrac1H\textstyle\sum_j x_j^2+\varepsilon},\qquad y_i=\frac{x_i}{\operatorname{RMS}(x)}\gamma_i`,
  "norm2":String.raw`\operatorname{RMS}(x)=\sqrt{\tfrac1H\textstyle\sum_j x_j^2+\varepsilon},\qquad y_i=\frac{x_i}{\operatorname{RMS}(x)}\gamma_i`,
  "hcpost1":String.raw`y[n,i,d]=\mathrm{post}_i\,x[n,d]+\mathrm{residual}[n,i,d]`,
  "hcpost2":String.raw`y[n,i,d]=\mathrm{post}_i\,x[n,d]+\mathrm{residual}[n,i,d]`,
  "aproj":String.raw`[\,q_a\mid k_v\mid k_{pe}\,]=\hat X\,W_a^{\mathsf T}\in\mathbb R^{T\times(2048+512+64)}`,
  "qan":String.raw`\tilde q=\frac{q_a}{\operatorname{RMS}(q_a)}\odot\gamma_q`,
  "qb":String.raw`Q=\tilde q\,W_{qb}^{\mathsf T}\in\mathbb R^{T\times 64\times 256},\quad 192\ \mathrm{nope}+64\ \mathrm{rope}`,
  "kvn":String.raw`\tilde k=\frac{k_v}{\operatorname{RMS}(k_v)}\odot\gamma_{kv},\qquad k_{pe}\ \text{不经 norm}`,
  "kvb":String.raw`[\,K^{\mathrm{nope}}\mid V\,]=\tilde k\,W_{kvb}^{\mathsf T}\in\mathbb R^{T\times 64\times(192+256)}`,
  "rope":String.raw`(Q_{:,\,192:},k_{pe})=\operatorname{RoPE}_{\mathrm{interleaved}}(Q_{:,\,192:},k_{pe};\theta{=}10^{7})`,
  "cache":String.raw`\mathcal C[\mathrm{slot}(t)]\leftarrow[\,\tilde k_t\mid k_{pe,t}\,]\in\mathbb R^{576}`,
  "iwqb":String.raw`Q^{\mathrm{idx}}=\tilde q\,W_{wqb}^{\mathsf T}\in\mathbb R^{T\times 32\times 128}`,
  "iwk":String.raw`[\,k\mid w\,]=X\,W_{wk}^{\mathsf T},\quad k\in\mathbb R^{T\times128},\ w\in\mathbb R^{T\times32}`,
  "ikn":String.raw`k'=\frac{k-\mu(k)}{\sqrt{\operatorname{var}(k)+10^{-6}}}\odot\gamma+\beta`,
  "iquant":String.raw`q_{fp8}=\operatorname{quant}_{128}(q),\qquad w'=w\cdot q_{\mathrm{scale}}\cdot 128^{-1/2}\cdot 32^{-1/2}`,
  "iscore":String.raw`s_j=\sum_h w'_h\,\langle q^{\mathrm{idx}}_h,k'_j\rangle,\qquad \mathcal I_t=\operatorname{TopK}_{2048}(s)`,
  "ishared":String.raw`\mathcal I_t=\mathrm{buffer}[\,\text{nearest full layer}\,]`,
  "qk":String.raw`S_{h}=\frac{Q_h\,[K_h\Vert k_{pe}]^{\mathsf T}}{\sqrt{256}}+M_{\mathrm{causal}},\quad j\in\mathcal I_t`,
  "sink":String.raw`\mathrm{out}\leftarrow\mathrm{out}\cdot\frac{e^{\mathrm{lse}}}{e^{\mathrm{lse}}+e^{\mathrm{sink}_h}},\qquad P=\operatorname{softmax}(S)`,
  "pv":String.raw`O_h=\sum_{j\in\mathcal I_t}P_{h,j}\,V_{h,j},\qquad O\in\mathbb R^{T\times64\times256}`,
  "gate":String.raw`Y=O\odot\sigma(g),\qquad g=\hat X\,W_{gate}^{\mathsf T}\in\mathbb R^{T\times64\times256}`,
  "oproj":String.raw`Y_{\mathrm{attn}}=Y\,W_o^{\mathsf T}\in\mathbb R^{T\times 6144}`,
  "gateup":String.raw`[\,g\mid u\,]=\hat U\,[W_{gate}\Vert W_{up}]^{\mathsf T}\in\mathbb R^{T\times36864}`,
  "act":String.raw`z=\operatorname{silu}(g)\odot u,\qquad \operatorname{silu}(x)=x\,\sigma(x)`,
  "down":String.raw`Y_{\mathrm{ffn}}=z\,W_{down}^{\mathsf T}\in\mathbb R^{T\times 6144}`,
  "router":String.raw`r=\hat U W_g^{\mathsf T}\in\mathbb R^{T\times 256}\quad(\mathrm{fp32})`,
  "select":String.raw`s=\sigma(r),\quad\mathcal E=\operatorname{TopK}_8(s+b),\quad\hat w_e=2.827\,\frac{s_e}{\sum_{j\in\mathcal E}s_j}`,
  "experts":String.raw`Y_{\mathrm{routed}}=\sum_{e\in\mathcal E}\hat w_e\,W_{2,e}\!\left[\operatorname{clip}(g_{,e},10)\odot\sigma(\operatorname{clip}(g_{,e},10))\odot\operatorname{clip}(u_{,e},-10,10)\right]`,
  "shared":String.raw`Y_{\mathrm{shared}}=W_{2}\,[\operatorname{silu}(g)\odot u]\quad(\text{无 clamp})`,
  "sum":String.raw`Y_{\mathrm{moe}}=Y_{\mathrm{routed}}+Y_{\mathrm{shared}}`,
};
