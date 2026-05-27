# C Implementation Test Dataset

## 📊 Dataset Overview

这个数据集包含用于测试C momentum compressor实现的真实神经网络训练梯度数据。

### 数据来源
- **来源**: CIFAR-10 联邦学习训练
- **模型**: ResNet18 (11M参数) 和 ResNet50 (23.5M参数)
- **场景**: 单个client的3轮训练梯度
- **框架**: APPFL federated learning

### 数据用途
- ✅ 验证C实现的**算法正确性**
- ✅ 测试**动量预测**功能（需要多轮数据）
- ✅ 评估**压缩性能**和压缩比
- ✅ 调试**误差控制**和压缩策略

## 📁 Dataset Structure

```
/home/exouser/compressor/final/dataset/
│
├── README.md                    # 本文件
│
├── ResNet18 (3 rounds × 43MB = 129MB)
│   ├── round_0_client_0.bin    # Round 0: 直接压缩baseline
│   ├── round_1_client_0.bin    # Round 1: 动量预测启动
│   └── round_2_client_0.bin    # Round 2: 动量预测稳定
│
└── ResNet50 (3 rounds × 90MB = 270MB)
    ├── round_0_client_0.bin    # Round 0: 直接压缩baseline
    ├── round_1_client_0.bin    # Round 1: 动量预测启动
    └── round_2_client_0.bin    # Round 2: 动量预测稳定

Total: ~400MB
```

## 📦 Data Format

### Binary File Format (.bin)

每个.bin文件包含一个训练轮次的完整模型梯度：

```
Header (76 bytes):
  - round_num      (4 bytes, uint32)
  - client_id      (64 bytes, char[64])
  - layer_count    (8 bytes, uint64)

For each layer:
  - layer_name     (256 bytes, char[256])
  - shape          (64 bytes, uint64[8])
  - ndim           (8 bytes, uint64)
  - dtype          (4 bytes, uint32)
  - data_size      (8 bytes, uint64)
  - data           (variable, raw bytes)
```

### Layer Data

每个二进制文件包含101-200层的网络参数梯度：

**ResNet18 (101 layers):**
- Convolutional layers: conv1, layer1.x.conv{1,2}, layer2.x.conv{1,2}, ...
- Batch normalization: bn1, layer1.x.bn{1,2}, ...
- Fully connected: linear.weight

**ResNet50 (更多层):**
- 包含Bottleneck结构的更深网络
- Conv layers, BN layers, shortcuts, FC layer

### Data Types
- `DTYPE_FLOAT32 = 0`: 32位浮点数（大部分参数）
- `DTYPE_FLOAT64 = 1`: 64位浮点数（极少）
- `DTYPE_INT32 = 2`: 32位整数
- `DTYPE_INT64 = 3`: 64位整数
- `DTYPE_UINT8 = 4`: 8位无符号整数

## 🎯 Usage Example

### C Program
```c
#include "momentum_compressor.h"

// Load data
TestLayer* layers = load_test_data("dataset/round_0_client_0.bin", &num_layers);

// Initialize compressor
CompressorConfig config = momentum_compressor_default_config();
MomentumCompressor* compressor = momentum_compressor_create(&config);

// Compress layers
for (size_t i = 0; i < num_layers; i++) {
    NDArray* gradient = ndarray_create(layers[i].shape, layers[i].ndim, layers[i].dtype);
    memcpy(gradient->data, layers[i].data, layers[i].data_size);

    CompressedLayerData* compressed = momentum_compressor_compress_layer(
        compressor, layers[i].layer_name, gradient);

    // Process compressed data...

    compressed_layer_data_free(compressed);
    ndarray_destroy(gradient);
}

momentum_compressor_destroy(compressor);
free_test_data(layers, num_layers);
```

### Run Test
```bash
cd /home/exouser/compressor/final
export LD_LIBRARY_PATH="/home/exouser/.appfl/.compressor/SZ3/lib:$LD_LIBRARY_PATH"
./test_c_real
```

## 📊 Expected Results

### Round 0 (Baseline)
```
Type: direct
Codec: zstd
Ratio: 1.08-1.10x
```

### Round 1 (Momentum Prediction Starts)
```
Type: momentum_predicted
Codec: zstd
Ratio: 1.08-1.10x
Predicted kernels: 15-20%
Sign mismatch: 18-22%
```

### Round 2 (Momentum Prediction Stable)
```
Type: momentum_predicted
Codec: zstd
Ratio: 1.08-1.10x
Predicted kernels: 15-20%
Sign mismatch: 18-22%
```

## 🔧 Regenerating Dataset

如果需要重新生成数据集：

```bash
cd /home/exouser/compressor/final/EB-FaLCom

# Step 1: 捕获真实训练数据（3轮，1个client）
python capture_3rounds_simple.py

# Step 2: 转换为C可读的二进制格式
python convert_to_binary.py

# Step 3: 复制到测试目录
cp dataset/resnet18/round_*_client_0.bin /home/exouser/compressor/final/dataset/
cp dataset/resnet50/round_*_client_0.bin /home/exouser/compressor/final/dataset/
```

详细步骤见 `DATA_GENERATION_GUIDE.md`

## 📈 Dataset Statistics

### ResNet18
- **Total Layers**: 101
- **Total Parameters**: ~11M
- **Data Size**: 42.66 MB per round
- **Binary Size**: 42.69 MB per round (with metadata)
- **Compression Ratio**: 1.08x (ZSTD)

### ResNet50
- **Total Layers**: ~200
- **Total Parameters**: ~23.5M
- **Data Size**: 90.00 MB per round
- **Binary Size**: 90.03 MB per round (with metadata)
- **Compression Ratio**: 1.08-1.10x (ZSTD)

## 🎪 Validation Checklist

使用此数据集验证C实现时，应确认：

- [ ] ✅ **Round 0 直接压缩**: 成功建立baseline
- [ ] ✅ **Round 1 动量预测**: 预测功能激活，有预测统计
- [ ] ✅ **Round 2 动量稳定**: 预测比例稳定在15-20%
- [ ] ✅ **符号预测**: 符号不匹配率在18-22%范围
- [ ] ✅ **压缩比**: 达到1.08x以上
- [ ] ✅ **所有层类型**: conv, bn, fc层均正确处理
- [ ] ✅ **内存管理**: 无内存泄漏
- [ ] ✅ **错误处理**: 边界情况正确处理

## 📝 Notes

- **单Client策略**: 只使用1个client的数据足够测试算法正确性
- **真实数据**: 来自实际CIFAR-10训练，包含真实梯度分布
- **多轮必要性**: 动量预测需要历史数据，至少需要2-3轮
- **数据大小**: ~400MB total，可接受的测试数据量
- **可复现性**: 使用固定random seed，结果可复现

## 🔗 Related Files

- `capture_3rounds_simple.py` - 数据捕获脚本
- `convert_to_binary.py` - 格式转换脚本
- `test_c_real.c` - C测试程序
- `DATA_GENERATION_GUIDE.md` - 详细生成指南
