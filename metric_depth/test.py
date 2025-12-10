from thop import profile
from ptflops import get_model_complexity_info
import torch
import numpy as np
from depth_anything_v2.dpt import DepthAnythingV2

def test_efficientvit_model():
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"使用设备: {DEVICE}")

    # 1. 配置模型参数
    model_configs = {
        'efficientvit_b0': {
            'encoder': 'efficientvit_b0',  # 改为EfficientViT标识
            'features': 64,  # 根据EfficientViT输出特征维度调整
            'out_channels': [48, 96, 192, 384],  # 需与EfficientViT中间层输出通道匹配
            'max_depth': 20.0  # 保持与原metric模型一致
        }
    }
    
    # 2. 初始化模型（假设已在DepthAnythingV2中集成EfficientViT）
    try:
        model = DepthAnythingV2(**model_configs['efficientvit_b2'])
        model = model.to(DEVICE)  # 将模型移到指定设备
        print("模型初始化并移动到设备成功")
    except Exception as e:
        print(f"模型初始化失败：{e}")
        return
    
    # 3. 加载权重（若有修改后的权重，若无则测试随机权重能否运行）
    try:
        model.eval()
        print("使用随机权重测试")
    except Exception as e:
        print(f"权重加载失败（继续测试随机权重）：{e}")
    
    # 4. 准备测试图像
    try:
        # 生成随机图像
        raw_img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)  # 随机生成480x640图像
        print(f"测试图像尺寸：{raw_img.shape[:2]}")
    except Exception as e:
        print(f"图像准备失败：{e}")
        return
    
    # 5. 模型推理（测试前向传播）
    try:
        with torch.no_grad():
            # 调用infer_image方法，内部包含预处理和后处理
            depth_map = model.infer_image(raw_img, input_size=518)  # 可调整input_size
        
        # 验证输出格式
        assert isinstance(depth_map, np.ndarray), "输出不是numpy数组"
        assert depth_map.ndim == 2, "输出不是2D深度图"
        assert depth_map.shape == raw_img.shape[:2], f"深度图尺寸与输入不匹配（输入：{raw_img.shape[:2]}, 输出：{depth_map.shape}）"
        assert np.all(depth_map >= 0) and np.all(depth_map <= model.max_depth), "深度值超出合理范围"
        
        print("="*50)
        print("测试成功！")
        print(f"输出深度图形状：{depth_map.shape}")
        print(f"深度值范围：[{depth_map.min():.2f}, {depth_map.max():.2f}]米")
        print("="*50)
    
    except Exception as e:
        print(f"前向传播失败：{e}")
        return

    # 计算Flops和Params（输入尺寸为518x518）
    flops, params = get_model_complexity_info(
        model,
        input_res=(3, 518, 518),  # (C, H, W)
        as_strings=False,  # 不返回字符串，返回数值
        print_per_layer_stat=False,  # 不打印每层统计
    )

    print(f"Flops: {flops / 1e9:.2f} G")
    print(f"Params: {params / 1e6:.2f} M")

if __name__ == '__main__':
    test_efficientvit_model()