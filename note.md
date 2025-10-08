# 什么是 Depthwise Separable Convolution（深度可分离卷积）
它把一个普通卷积（比如 64→64，kernel=3×3）拆成两步：
Depthwise Convolution：对每个通道单独做卷积（groups=in_channels）。
Pointwise Convolution：用 1×1 卷积来做通道混合。
这样参数量和计算量都大幅减少

# Depthwise Separable Convolution 替换范围分析

## 📌 EDSR 的结构回顾
1. **Head**：输入卷积，把 3 通道（RGB）变成 n_feats（通常 64 或 256）  
2. **Body**：一堆残差块（主要计算量 & 参数都在这里）  
3. **Tail**：上采样模块（PixelShuffle + 卷积），输出回 3 通道  

---

## 📌 替换范围的区别

### ✅ 方案 A：只替换 Body（当前做法）
- **优点**：
  - Body 占了 EDSR 绝大多数卷积层，参数量和 FLOPs 降低很多。  
  - Head/Tail 保持普通卷积，输入输出稳定，不会出现表示能力不足的问题。  
  - **比较安全**，效果下降相对可控。  
- **缺点**：
  - 模型依然保留了 Head/Tail 的计算开销（虽然这部分占比小）。  

---

### ✅ 方案 B：Head + Body + Tail 全部替换
- **优点**：
  - 参数量和计算量进一步减少（极致轻量化）。  
- **缺点**：
  - Head 如果也换成 Depthwise，**输入 RGB → 特征的映射能力可能不足**，会损失很多表示力。  
  - Tail 如果换了，**上采样质量下降**，容易导致 PSNR/SSIM 明显掉。  
  - 整体模型性能可能掉得比较厉害。  

---

## 📌 哪种更适合论文
- **硕士论文 / SCI 四区小论文**：推荐 **只替换 Body**。  
  👉 显著减少参数量，但 PSNR 下降不会太离谱，可作为轻量化改进。  
- **消融实验**：  
  同时做两组：  
  - Body-only 替换  
  - 全部替换（Head+Body+Tail）  
  对比后得出结论：  
  - 参数量减少更多，但 PSNR 掉得更多  
  - **只替换 Body 是更合理的折中方案**  

---

## 📊 假设实验结果示例（论文表格）
| 模型版本             | 参数量 | FLOPs | Set5 PSNR | Set14 PSNR | 结论 |
|----------------------|--------|-------|-----------|------------|------|
| Baseline EDSR (x2)   | 1.5M   | 150G  | 38.1 dB   | 33.7 dB    | 基线 |
| EDSR + DWConv (Body) | 0.9M   | 95G   | 37.8 dB   | 33.5 dB    | 较优折中 |
| EDSR + DWConv (All)  | 0.7M   | 70G   | 37.1 dB   | 32.8 dB    | 太轻量，精度下降 |

---

## ✅ 总结
- **只替换 Body**：轻量化 + 性能下降小 → 更适合论文的“改进点”。  
- **全部替换**：更极端的轻量化，可作为消融实验对比，但效果差。  

# DWConv 轻量化

## 1. 标准卷积计算量

输入特征图大小：$H \times W$  
输入通道数：$C_{in}$  
输出通道数：$C_{out}$  
卷积核大小：$K \times K$

标准卷积的计算量（乘加操作数 FLOPs）为：

$$
FLOPs_{conv} = H \times W \times C_{in} \times C_{out} \times K^2
$$

---

## 2. 深度可分离卷积（DWConv + PWConv）

DWConv 分为两步：

1. **Depthwise Convolution（逐通道卷积）**：每个输入通道单独做卷积  
   $$
   FLOPs_{DW} = H \times W \times C_{in} \times K^2
   $$

2. **Pointwise Convolution（逐点卷积，$1\times1$ 卷积）**：做通道融合  
   $$
   FLOPs_{PW} = H \times W \times C_{in} \times C_{out}
   $$

因此总计算量为：

$$
FLOPs_{DWConv} = FLOPs_{DW} + FLOPs_{PW} = H \times W \times (C_{in} \times K^2 + C_{in} \times C_{out})
$$

---

## 3. 对比标准卷积和 DWConv

计算量比值：

$$
\frac{FLOPs_{DWConv}}{FLOPs_{Conv}} = \frac{C_{in} \times K^2 + C_{in} \times C_{out}}{C_{in} \times C_{out} \times K^2}
$$

当 $K=3$ 时：

$$
\frac{FLOPs_{DWConv}}{FLOPs_{Conv}} \approx \frac{1}{C_{out}} + \frac{1}{K^2}
$$

例如 $K=3, C_{out}=64$：

$$
\frac{FLOPs_{DWConv}}{FLOPs_{Conv}} \approx \frac{1}{64} + \frac{1}{9} \approx 0.12
$$

也就是说，DWConv 的计算量大约只有标准卷积的 **12%**，轻量化效果显著。

---

## 4. 参数量对比

标准卷积参数量：

$$
Params_{Conv} = C_{in} \times C_{out} \times K^2
$$

DWConv 参数量：

$$
Params_{DWConv} = C_{in} \times K^2 + C_{in} \times C_{out}
$$

对比结果和 FLOPs 的比例相似，DWConv 参数量也显著减少。

---

## 5. 结论

- **为什么轻量化**：  
  DWConv 将卷积分解为逐通道卷积 + 通道融合，避免了全连接的卷积核，减少了大量冗余计算。

- **如何证明**：  
  通过 FLOPs 和参数量公式对比，可以量化说明 DWConv 的计算复杂度远小于标准卷积。  
  在 $3\times3$ 卷积、$C_{out}=64$ 的场景下，计算量降低至 **12%**，参数量相应减少。

这个结果已经包含了三大核心指标：
	1.	参数量 (Params) → 模型复杂度和存储需求。
	2.	FLOPs → 理论计算复杂度，体现计算量大小。
	3.	推理时间 / FPS → 实际运行效率，反映部署端性能