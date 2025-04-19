# 小规模语音识别系统

这是一个基于LibriSpeech数据集的小规模语音识别系统，使用PyTorch实现。

## 环境要求

- Python 3.6+
- PyTorch 1.7+
- torchaudio
- pandas
- numpy
- tqdm

可以通过以下命令安装依赖：

```bash
pip install torch torchaudio pandas numpy tqdm
```

### 注意

原始项目使用了ctcdecode包进行波束搜索解码，但该包可能在某些环境下安装困难。本项目已对代码进行调整，使用了简化版解码器（`Utils/decoder_simple.py`），无需安装ctcdecode即可运行训练。

如果您需要使用高级解码功能（如带语言模型的波束搜索），可以尝试安装ctcdecode：
```bash
# 首先确保安装了PyTorch
pip install torch torchaudio
# 然后安装ctcdecode
pip install ctcdecode
```

如果安装失败，可以使用项目提供的简化版解码器，只会影响评估准确性，不会影响模型训练。

## 数据准备

1. 下载LibriSpeech数据集：

```bash
mkdir -p data
cd data
# 下载LibriSpeech数据集子集
wget https://www.openslr.org/resources/12/dev-clean.tar.gz
wget https://www.openslr.org/resources/12/dev-other.tar.gz
wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
# 解压数据
tar -xzf dev-clean.tar.gz
tar -xzf dev-other.tar.gz
tar -xzf train-clean-100.tar.gz
cd ..
```

2. 处理数据，生成JSON文件：

```bash
python prepare_data.py --librispeech_path ./data/LibriSpeech --output_dir ./data
```

## 训练模型

1. 创建初始模型：

```bash
python create_initial_model.py
```

2. 训练模型：

```bash
python TrainAcoustMain.py
```

## 项目结构

- `TrainAcoustMain.py`：主训练脚本
- `Utils/`：工具函数和模型定义
  - `Acousticmod.py`：声学模型定义
  - `DataProcess.py`：数据处理类
  - `Tokenizer.py`：文本标记化
  - `Preprocessing.py`：音频预处理
- `prepare_data.py`：数据准备脚本
- `create_initial_model.py`：创建初始模型

## 模型结构

该模型使用了CNN、Transformer和全连接层的组合架构进行语音识别，模型训练使用CTC损失函数。

## 使用训练好的模型进行语音识别

### 1. 训练模型

按照上述步骤训练模型后，模型文件将保存在 `models/` 目录中，例如 `models/modelMelN100.pth`。

### 2. 转换模型格式

将模型转换为TorchScript格式，便于推理使用：

```bash
python convert_model.py --input models/modelMelN100.pth --output models/model.zip
```

### 3. 下载测试音频

```bash
python download_test_audio.py
```

这将下载LibriSpeech测试数据集的一小部分，解压并列出几个示例音频文件路径。

### 4. 进行语音识别

使用转换后的模型识别音频：

```bash
python recognize_speech.py --model models/model.zip --audio test_audio/LibriSpeech/test-clean/61/70968/61-70968-0000.flac
```

如果你有语言模型，可以使用波束搜索进行解码以提高准确率：

```bash
python recognize_speech.py --model models/model.zip --audio test_audio/LibriSpeech/test-clean/61/70968/61-70968-0000.flac --beam_search --lm path/to/your/language_model.bin
```

## 处理音频文件
该系统支持多种音频格式，如果你有自己的音频文件想要识别，确保:

1. 最好是单声道音频
2. 采样率为16kHz（系统会自动重采样）
3. 音频长度适中（过长的音频可能导致性能问题）

