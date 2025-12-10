import torch
import cv2
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2
def main():
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]}
    }
    encoder = 'vits'
    max_depth = 20

    model = DepthAnythingV2(**{**model_configs[encoder], 'max_depth': max_depth})
    checkpoint_path = 'checkpoints/depth_anything_v2_metric_hypersim_vits.pth'
    model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
    model.eval()

    # 3. 准备示例输入（需与模型输入尺寸一致，默认 518x518）
    input_size = 518
    # 创建随机输入张量（形状：[batch=1, channel=3, height=518, width=518]）
    dummy_input = torch.randn(1, 3, input_size, input_size)

    # 4. 导出 ONNX
    output_path = 'depth_anything_v2_metric_hypersim_vits.onnx'
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=12,  # 推荐使用 12+ 版本以支持更多算子
        do_constant_folding=True,
        input_names=['input'],  # 输入节点名称
        output_names=['output'],  # 输出节点名称
        dynamic_axes={  # 支持动态尺寸（可选，根据需求设置）
            'input': {2: 'height', 3: 'width'},
            'output': {1: 'height', 2: 'width'}
        }
    )
    print(f"ONNX 模型已导出至：{output_path}")

if __name__ == '__main__':
    main()