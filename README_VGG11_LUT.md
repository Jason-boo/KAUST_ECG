# VGG-11 + 8-bit量化激活 + LUT ReLU 对比实验

本目录新增了一个可直接运行的脚本：

- `train_eval_vgg11_quant_lut.py`

它会自动完成以下流程：

1. 读取 `LUT_ReLU.xlsx`（两列：输入、输出，原始16点映射）。
2. 将 `[-1, 1]` 范围内的16点LUT插值为8-bit（256点）LUT，并保存为：
   - `outputs_vgg11_lut/lut_relu_interpolated_8bit.csv`
3. 用标准 VGG-11 结构（所有激活均为 ReLU）分别在 MNIST 和 CIFAR-10 上训练。
4. 训练过程中，每一层激活输入/输出都会被限制在 `[-1,1]` 并进行8-bit量化。
5. 保存训练好的两个模型：
   - `outputs_vgg11_lut/vgg11_mnist_quant8_standard_relu.pth`
   - `outputs_vgg11_lut/vgg11_cifar10_quant8_standard_relu.pth`
6. 分别对两个模型进行推理：
   - 标准量化ReLU推理
   - 用插值后的8-bit LUT替换所有ReLU后的推理
7. 输出4个准确率：
   - MNIST(标准ReLU)
   - MNIST(LUT ReLU)
   - CIFAR10(标准ReLU)
   - CIFAR10(LUT ReLU)

## 运行方式

先确保依赖可用：

```bash
pip install torch torchvision pandas openpyxl numpy
```

然后在仓库根目录执行：

```bash
python train_eval_vgg11_quant_lut.py --epochs 10 --batch-size 128 --lut-xlsx ./LUT_ReLU.xlsx
```

如果你想快速冒烟测试：

```bash
python train_eval_vgg11_quant_lut.py --epochs 1 --batch-size 64 --lut-xlsx ./LUT_ReLU.xlsx
```
