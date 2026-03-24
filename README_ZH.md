# TCWMSDT

<p align="center">
  <img src="https://img.shields.io/badge/许可证-Apache%202.0-blue.svg" alt="License">
  <img src="https://img.shields.io/badge/Python-3.10+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/中西医 benchmark-辨证分型-orange.svg" alt="TCWM">
</p>

一个用于评估大语言模型（LLM）在中医辨证论治任务中能力的综合基准测试框架，专注于脾胃病（消化系统疾病）领域。

## 📋 项目简介

TCWMSDT（Traditional Chinese and Western Medicine Benchmark for Syndrome Differentiation Task，中医辨证论治任务基准测试）旨在系统评估大语言模型在中医临床诊断，特别是辨证论治方面的能力。本基准测试聚焦于脾胃病，这是中医临床实践中最常见的疾病类型之一。

### 核心特性

- **多维度评估**：涵盖中医诊断的10个关键维度
- **偏差抑制**：采用三轮选项随机化评估，有效降低位置偏差
- **双模式支持**：支持API调用和本地模型部署
- **LLM作为评判员**：使用大模型评估病因、病机等复杂维度
- **思维链分析**：评估推理过程的完整性和准确性
- **综合报告生成**：输出详细的HTML评估报告

## 🏗️ 项目架构

```
TCWMSDT/
├── tcwm_benchmark.py          # 主测试框架入口
├── tools/
│   ├── model_interface.py    # 模型抽象层（API & 本地）
│   ├── data_loader.py        # 数据加载与预处理
│   ├── utils.py               # 工具函数
│   └── report_generator.py   # HTML报告生成
├── evaluators/
│   ├── multiple_choice_evaluator.py  # 选择题评估器
│   └── llm_judge_evaluator.py        # LLM评判评估器
├── prompts/                  # 评估提示词
├── finall_exam_data/         # 基准测试数据集
└── results/                  # 评估结果输出
```

## 📊 评估维度

基准测试从以下10个中医诊断维度评估模型能力：

| 维度 | 说明 | 评估方法 |
|--------|------|----------|
| 证型 | 辨证类型 | 选择题（多选） |
| 病性 | 疾病性质 | 选择题（单选） |
| 病位 | 病变部位 | 选择题（多选） |
| 治则治法 | 治疗原则 | 选择题（多选） |
| 病因 | 发病原因 | LLM评判 |
| 病机 | 病理机制 | LLM评判 |
| 治疗方法 | 治疗手段 | LLM评判 |
| 注意事项 | 医嘱建议 | LLM评判 |
| COT完备性 | 患者信息利用的全面性 | LLM评判 |
| COT准确性 | 检查幻觉 | LLM评判 |

### 附加评估类别

- **中医药学**：中医基础知识
- **西医药学**：西医知识
- **医学伦理**：医学伦理考量
- **安全评估**：模型安全性与对齐

## 🚀 快速开始

### 环境要求

```bash
Python 3.10+
pip install -r requirements.txt
```

### 安装

```bash
git clone https://github.com/your-org/TCWMSDT.git
cd TCWMSDT
pip install -r requirements.txt
```

### 运行评估

#### API模式（适用于云端模型）

```bash
python tcwm_benchmark.py \
    --model_type api \
    --api_url "http://localhost:8000/v1" \
    --model_name "your-model" \
    --api_key "your-api-key" \
    --output_dir "results/your-model-run"
```

#### 本地模式（适用于自托管模型）

```bash
python tcwm_benchmark.py \
    --model_type local \
    --model_path "/path/to/your/model" \
    --output_dir "results/local-model-run"
```

### 命令行参数

| 参数 | 说明 | 默认值 |
|--------|------|----------|
| `--model_type` | 模型类型：`api` 或 `local` | `api` |
| `--api_url` | API端点URL | `http://localhost:8005/v1` |
| `--model_name` | 模型标识符 | `lora_5e6` |
| `--api_key` | API认证密钥 | `-` |
| `--model_path` | 本地模型路径 | `-` |
| `--output_dir` | 结果输出目录 | `results/piweibing_test1` |
| `--config_file` | 配置文件路径 | 配置文件路径 |
| `--resume` | 从断点恢复 | `True` |
| `--skip_think` | 跳过思维链评估（适用于非CoT模型） | `False` |

## 📁 数据格式

### 辨证论治案例

```json
{
  "id": "case_001",
  "instruction": "患者症状描述，包含舌象、脉象...",
  "output": {
    "证型": "脾胃虚弱",
    "证型答案": "A;B",
    "证型选项": "A:脾胃虚弱;B:脾胃湿热;...",
    "病性": "虚证",
    "病性答案": "A",
    "病性选项": "A:虚证;B:实证;...",
    "病位": "脾胃",
    "病位答案": "A;B",
    "病位选项": "A:脾;B:胃;...",
    "治则治法": "健脾益气",
    "治则治法答案": "A;B",
    "治则治法选项": "A:健脾益气;B:清热化湿;...",
    "病因": "饮食不节,劳倦过度",
    "病机": "脾胃虚弱,运化失常",
    "治疗方法": "中药汤剂,针灸",
    "注意事项": "饮食调护,情志调节"
  },
  "disease_cn": "脾胃病",
  "disease_en": "Spleen and Stomach Disease",
  "exam_class": "中西医辩证分型"
}
```

### 选择题案例

```json
{
  "id": "question_001",
  "question": "题目内容...",
  "answer": "A",
  "option": {"A": "选项1", "B": "选项2"},
  "question_type": "单项选择题",
  "exam_class": "中医药学"
}
```

## 📈 评分体系

最终得分采用加权综合计算：

| 任务类别 | 权重 |
|---------------|--------|
| 辨证论治（中西医辩证分型） | 40% |
| 中医基础（中医药学） | 20% |
| 西医知识（西医药学） | 20% |
| 医学伦理 | 10% |
| 内容安全（安全评估） | 10% |

### 辨证分型评分

- **选择题维度**：采用Sp指标：`Sp = |A∩B| / (|A| + |Ā∩B|)`
  - 其中A为标准答案集合，B为模型选择集合
- **LLM评判维度**：0-100分制，归一化至0-1

## 🔧 配置

编辑 `configs/config_example.json` 自定义评估设置：

```json
{
  "data_path": "finall_exam_data/TCWM-PIWEIBING.json",
  "llm_judge_api_host": "127.0.0.1",
  "llm_judge_api_port": 8002,
  "llm_judge_model_name": "Qwen3-32B",
  "llm_judge_api_key": "",
  "max_retries": 3,
  "checkpoint_interval": 10,
  "random_seed": 42
}
```

## 📝 输出结果

基准测试生成全面的评估报告：

```
results/your-model-run/
├── checkpoint.json          # 断点恢复文件
├── detailed_results.json    # 完整评估结果
├── evaluation_report.html  # 可视化HTML报告
└── logs_YYYYMMDD-HHMMSS.log # 执行日志
```

## 🛠️ 依赖项

- `transformers` - 模型加载
- `torch` - 深度学习后端
- `openai` - API客户端
- `httpx` - HTTP客户端
- `tqdm` - 进度条
- `matplotlib` - 可视化
- `numpy` - 数值计算
- `json-repair` - JSON解析

## 📄 许可证

本项目基于Apache License 2.0许可证 - 详见 [LICENSE](LICENSE) 文件。

## 🤝 贡献

欢迎提交贡献！请随时提交Pull Request。

## ⚠️ 免责声明

本基准测试仅用于研究和教育目的。评估结果不应作为医疗建议使用。请始终咨询合格的医疗专业人员以获取医疗决策建议。

---

<p align="center">❤️ 用心支持中医人工智能研究</p>
