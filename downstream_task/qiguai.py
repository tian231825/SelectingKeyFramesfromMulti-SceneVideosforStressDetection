import torch
import torch.nn as nn
import torch
import torch.nn as nn

# 假设 transformer_output 是你的 Transformer 模型的输出张量，形状为 (batch, 300, 768)
transformer_output = torch.randn(32, 300, 768)  # 举例中的 batch size 为 32

# 定义一个自注意力层
self_attention = nn.MultiheadAttention(embed_dim=768, num_heads=12)  # 假设使用了 12 个头

# 使用自注意力层对 transformer_output 进行自注意力操作
self_attn_output, _ = self_attention(transformer_output.transpose(0, 1), transformer_output.transpose(0, 1), transformer_output.transpose(0, 1))

# 获取CLS token的输出，即第一个位置的输出
cls_output = self_attn_output[0, :, :]  # 获取第一个位置的输出

# 输出结果
print(cls_output.size())  # 输出结果的形状为 (batch, 768)