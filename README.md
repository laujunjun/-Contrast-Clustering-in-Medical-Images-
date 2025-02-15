# -Contrast-Clustering-in-Medical-Images-

##  一、DL-CC（Deep Learning with Contrastive Clustering）框架

使用论文“Histopathology images-based deep learning prediction of prognosis and therapeutic response in small cell lung cancer”的对比聚类方法对医学图像进行聚类。

该论文深度学习特征提取模块的实现是通过DL-CC（Deep Learning with Contrastive Clustering）框架来完成。以下是该模块具体实现的详细步骤：

1. 数据准备与预处理

- 图像分割：首先，从全切片图像（WSI）中提取肿瘤区域，病理学家在20倍放大下提取肿瘤的中心区域，形成肿瘤组织微阵列（TMA）。

- 图像切割：将每个TMA分割成不重叠的224 × 224像素的图块，以便进行后续处理。

- 背景去除：使用Otsu阈值法将白色背景与组织区域分离，确保仅保留组织覆盖率超过60%的图块。

- 数据增强：为了提高模型的鲁棒性，应用六种随机图像增强技术，包括翻转、旋转、对比度调整、缩放、HSV调整和噪声添加。这些增强技术在预处理阶段应用，增加训练数据集的多样性。

2. 特征提取模块

- 非冗余向量提取模块：
  使用一对共享权重的ResNet50网络处理不同的增强图像。这种结构允许模型从不同的图像增强版本中提取特征，捕捉复杂的组织形态信息。
  该模块的输出是一个2048维的特征向量，表示每个图块的特征。

- 实例级对比特征映射模块：
  该模块负责将提取的特征映射到一个50维的潜在空间。通过对比学习的方式，模型能够学习到不同图像之间的相似性和差异性。
  具体来说，使用对比损失函数来优化模型，使得相似的图像特征在特征空间中靠近，而不相似的图像特征则远离。

3. 聚类与特征构建

- 聚类分析：
  将提取的特征进行聚类，形成多个组织形态表型簇（HPCs）。在该研究中，识别了50个复杂的HPCs作为病理特征。
  每个TMA的图块被聚类到不同的簇中，形成特征向量。
  
- 特征向量构建：
  计算每个簇在每个TMA中的比例，这些比例构成了该TMA的特征向量。
  这些特征向量随后用于Cox回归模型，以建立组织表型与临床注释之间的关联。

4. 模型训练与优化

- 训练过程：
  使用标注的训练数据集对模型进行训练，优化模型参数以提高特征提取的准确性。
  通过交叉验证等方法评估模型的性能，确保其在不同数据集上的泛化能力。

5. 结果评估

- 性能评估：
  通过比较不同风险组的生存曲线，使用Kaplan-Meier方法和Log-rank检验来评估PathoSig的预测能力。
  结果显示，PathoSig在不同的验证队列中表现出显著的预测性能和稳健性。

通过以上步骤，深度学习特征提取模块能够有效地从组织病理图像中提取有价值的特征，为后续的预后预测和治疗反应评估提供支持。

===================== 以上为对原文内容的解析 ====================

## 二、复现
1. 数据预处理
  - 将WSI分割在ROI区域内分割成patch，注意WSI的level与patch的尺寸大小。
  - 准备wsi标签文件：脚本```generate_slide_label.py```
2. 训练
  - 代码： ```train_resNet.py```
  - 输入：配置文件路径（包含输入数据与参数）
  - 输出：每一轮的模型参数文件(.pth)
  - 模型构建与优化：
    
  	•	使用 resnet 获取ResNet50架构，并加载到CUDA设备（如果可用）。

  	•	初始化了自动混合精度训练的 GradScaler，可以提高训练效率和稳定性。

  	•	optimizer 使用的是AdamW优化器，并应用学习率调度函数（lr_schedule），根据训练进度逐步调整学习率。

  - 训练过程：
    
  	•	进入训练循环，每个epoch中遍历训练数据。

  	•	每次计算损失（实例损失、聚类损失、Barlow损失），并根据加权的总损失进行反向传播，更新模型权重。

  	•	每10000个batch输出一次训练进度，包括学习率、各类损失等。

  	•	每个epoch结束时，会进行一次验证，并计算验证损失。
  - 模型保存与恢复：
    
  	•	支持从之前的检查点加载模型权重（initial_checkpoint），恢复训练进度。

  	•	每个epoch结束时保存当前模型的状态字典（state_dict）。
3. 推理
  - 代码：```test_resNet.py```
  - 输入：最好轮次的模型参数
  - 输出：128维特征（可在配置文件中调整）、聚类类别
  - 加载对比学习模型
    
	•	res = resnet.get_resnet("ResNet50")：获取 ResNet50 作为主干网络。

	•	解析 YAML 配置文件，获取模型参数 args。

	•	```model = model.Net(arg=args,resnet=res).to(device)```：构建自定义深度学习模型，加载预训练权重。










