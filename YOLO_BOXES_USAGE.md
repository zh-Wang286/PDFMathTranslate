# YOLO检测框可视化功能使用说明

## 功能概述

PDF2zh支持在原始PDF页面上可视化YOLO模型的布局检测结果。该功能可以帮助用户：

- **调试版面分析**：直观查看YOLO模型在原始文档上的检测结果
- **验证检测准确性**：在原始内容基础上评估布局分析的正确性
- **模型评估**：分析YOLO检测的准确性和置信度
- **区域分类理解**：清晰了解不同类型的内容区域分布
- **保持翻译清洁**：翻译页面无检测框干扰，确保最佳阅读体验

## 输出特点

**原始页面检测框显示**：
- **原始PDF页面**：显示YOLO检测框，便于验证原始文档的布局分析结果
- **翻译PDF页面**：保持简洁无检测框，确保翻译内容的可读性不受干扰
- **Dual模式**：原始页面（含检测框）和翻译页面（无检测框）交替显示，便于调试对比

**优化的标签显示**：
- **透明背景**：标签不使用白色背景，避免遮挡原始文本内容
- **描边效果**：使用白色描边和黑色填充的双重文字效果，确保标签在各种背景下都清晰可读
- **智能定位**：标签显示在检测框的左上角位置，减少对重要内容的遮挡

## 命令行使用

### 基本用法

```bash
# 启用检测框可视化
python -m pdf2zh.pdf2zh "document.pdf" --draw-layout-boxes

# 完整的翻译命令（推荐）
python -m pdf2zh.pdf2zh "document.pdf" --draw-layout-boxes --service azure-openai -li en -lo zh
```

### 完整参数示例

```bash
# 翻译英文PDF为中文，并显示检测框
python -m pdf2zh.pdf2zh \
    "research_paper.pdf" \
    --draw-layout-boxes \
    --service azure-openai \
    --lang-in en \
    --lang-out zh \
    --output ./translated_docs/
```

## 程序化使用

### Python API调用

```python
from pdf2zh.high_level import translate
from pdf2zh.doclayout import OnnxModel

# 加载YOLO模型
model = OnnxModel()

# 翻译并绘制检测框
result_files = translate(
    files=["document.pdf"],
    output="./output/",
    lang_in="en",
    lang_out="zh", 
    service="azure-openai",
    model=model,
    draw_layout_boxes=True  # 启用检测框绘制
)

# result_files包含生成的文件路径
mono_pdf, dual_pdf = result_files[0]
```

### 流式处理

```python
from pdf2zh.high_level import translate_stream
from pdf2zh.doclayout import OnnxModel
import io

# 读取PDF字节流
with open("document.pdf", "rb") as f:
    pdf_bytes = f.read()

model = OnnxModel()

# 流式翻译并绘制检测框
mono_pdf, dual_pdf, token_stats, para_stats, table_stats = translate_stream(
    stream=pdf_bytes,
    lang_in="en",
    lang_out="zh",
    service="azure-openai", 
    model=model,
    draw_layout_boxes=True
)

# 保存结果
with open("output_mono.pdf", "wb") as f:
    f.write(mono_pdf)
with open("output_dual.pdf", "wb") as f:
    f.write(dual_pdf)
```

## 检测框说明

### 颜色编码

检测框采用简化的颜色方案，按翻译逻辑分类：

**🟢 绿色 - 会被翻译的区域：**
- **Text**: 常规文本区域
- **Table**: 表格区域（粗边框突出显示）

**🔴 红色 - 不会被翻译的区域：**
- **Figure**: 图像区域
- **Abandon**: 废弃区域
- **Isolate_formula**: 独立公式
- **Formula_caption**: 公式标题
- **Reserved**: 其他保留区域

### 标签信息

每个检测框包含标签，显示：
- **区域类型**: text, table, figure等
- **区域编号**: 同页面中该类型的序号
- **置信度**: YOLO检测的置信度分数（如果<1.0会显示）

示例标签格式：
- `text_1`: 第一个文本区域
- `table_2(0.95)`: 第二个表格区域，置信度95%
- `figure_1`: 第一个图像区域

### 标签显示优化

- **透明背景**：标签不使用白色背景，避免遮挡原始文本内容
- **描边效果**：使用白色描边 + 黑色填充的双重文字效果，确保在各种背景色上都清晰可读
- **智能定位**：标签显示在检测框的左上角外侧，最大程度减少对重要内容的遮挡
- **自适应大小**：标签使用8pt字体，在清晰度和占用空间之间达到平衡

## 输出文件

启用检测框功能后，会生成以下文件：

```
output/
├── document-mono.pdf    # 仅翻译版本（无检测框）
└── document-dual.pdf    # 原文+翻译对照版本（仅原始页面带检测框）
```

### 推荐使用dual版本

**dual版本的优势**：
- **原始页面**：显示原始内容+检测框，便于验证YOLO分析结果
- **翻译页面**：显示纯净翻译内容，无检测框干扰，确保最佳阅读体验
- **页面交替**：奇数页原文（带检测框），偶数页译文（无检测框），便于调试对比
- **最佳平衡**：既能进行技术分析，又能获得清洁的翻译阅读体验

## 应用场景

### 1. 模型调试
```bash
# 处理测试文档并查看检测结果
python -m pdf2zh.pdf2zh "test_document.pdf" --draw-layout-boxes --service google
```

### 2. 翻译质量评估
```bash
# 生成带检测框的双语对照PDF
python -m pdf2zh.pdf2zh "academic_paper.pdf" --draw-layout-boxes --service azure-openai -li en -lo zh
```

### 3. 批量处理与分析
```python
import glob
from pdf2zh.high_level import translate
from pdf2zh.doclayout import OnnxModel

model = OnnxModel()

# 批量处理并生成可视化结果
pdf_files = glob.glob("input/*.pdf")
for pdf_file in pdf_files:
    translate(
        files=[pdf_file],
        output="./analyzed_output/",
        draw_layout_boxes=True,
        model=model
    )
```

## 性能影响

- **检测框绘制**：对翻译性能影响极小（<1%）
- **文件大小**：增加约5-10KB（检测框为矢量图形）
- **渲染兼容性**：生成标准PDF矢量图形，兼容所有PDF阅读器

## 技术实现

### 检测框生成
- 使用PDF原生矢量图形指令
- 颜色编码区分不同区域类型
- 动态标签显示检测信息
- 支持置信度显示

### 双页面实现
- **原始页面处理**：在原始PDF页面添加检测框覆盖层，便于调试版面分析
- **翻译页面处理**：保持纯净翻译内容，移除所有检测框元素
- **用户体验优化**：原始页面用于技术分析，翻译页面专注于内容阅读

## 故障排除

### 检测框未显示
1. 确认使用了 `--draw-layout-boxes` 参数
2. 检查PDF查看器是否支持图形显示
3. 查看控制台日志中的错误信息

### 标签文字模糊
- 这是正常的，标签使用小字体以避免干扰主要内容
- 可以在PDF查看器中放大查看

### 性能问题
- 检测框绘制对性能影响很小
- 如果遇到问题，可以先不使用此功能进行翻译，然后单独生成可视化版本

## 注意事项

1. **文件大小增加**：启用检测框会略微增加PDF文件大小
2. **可视化目的**：检测框主要用于调试和分析，不影响实际翻译质量
3. **兼容性**：生成的PDF文件与标准PDF完全兼容
4. **页面布局**：dual版本中，奇数页为原文，偶数页为译文 