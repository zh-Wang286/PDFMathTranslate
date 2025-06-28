import concurrent.futures
import logging
import re
import threading
import unicodedata
from dataclasses import dataclass
from enum import Enum
from string import Template
from typing import Dict, List, Tuple, Optional

import numpy as np
from pdfminer.converter import PDFConverter
from pdfminer.layout import LTChar, LTFigure, LTLine, LTPage
from pdfminer.pdffont import PDFCIDFont, PDFUnicodeNotDefined
from pdfminer.pdfinterp import PDFGraphicState, PDFResourceManager
from pdfminer.utils import apply_matrix_pt, mult_matrix
from pymupdf import Font
from tenacity import retry, wait_fixed

from pdf2zh.translator import (
    AnythingLLMTranslator,
    ArgosTranslator,
    AzureOpenAITranslator,
    AzureTranslator,
    BaseTranslator,
    BingTranslator,
    DashScopeTranslator,
    DeepLTranslator,
    DeepLXTranslator,
    DeepseekTranslator,
    DifyTranslator,
    GeminiTranslator,
    GoogleTranslator,
    GrokTranslator,
    GroqTranslator,
    ModelScopeTranslator,
    OllamaTranslator,
    OpenAIlikedTranslator,
    OpenAITranslator,
    QwenMtTranslator,
    SiliconTranslator,
    TencentTranslator,
    XinferenceTranslator,
    ZhipuTranslator,
)

from pdf2zh.utils import count_tokens

logger = logging.getLogger(__name__)


class PDFConverterEx(PDFConverter):
    def __init__(
        self,
        rsrcmgr: PDFResourceManager,
    ) -> None:
        PDFConverter.__init__(self, rsrcmgr, None, "utf-8", 1, None)

    def begin_page(self, page, ctm) -> None:
        # 重载替换 cropbox
        (x0, y0, x1, y1) = page.cropbox
        (x0, y0) = apply_matrix_pt(ctm, (x0, y0))
        (x1, y1) = apply_matrix_pt(ctm, (x1, y1))
        mediabox = (0, 0, abs(x0 - x1), abs(y0 - y1))
        self.cur_item = LTPage(page.pageno, mediabox)

    def end_page(self, page):
        # 重载返回指令流
        return self.receive_layout(self.cur_item)

    def begin_figure(self, name, bbox, matrix) -> None:
        # 重载设置 pageid
        self._stack.append(self.cur_item)
        self.cur_item = LTFigure(name, bbox, mult_matrix(matrix, self.ctm))
        self.cur_item.pageid = self._stack[-1].pageid

    def end_figure(self, _: str) -> None:
        # 重载返回指令流
        fig = self.cur_item
        assert isinstance(self.cur_item, LTFigure), str(type(self.cur_item))
        self.cur_item = self._stack.pop()
        self.cur_item.add(fig)
        return self.receive_layout(fig)

    def render_char(
        self,
        matrix,
        font,
        fontsize: float,
        scaling: float,
        rise: float,
        cid: int,
        ncs,
        graphicstate: PDFGraphicState,
    ) -> float:
        # 重载设置 cid 和 font
        try:
            text = font.to_unichr(cid)
            assert isinstance(text, str), str(type(text))
        except PDFUnicodeNotDefined:
            text = self.handle_undefined_char(font, cid)
        textwidth = font.char_width(cid)
        textdisp = font.char_disp(cid)
        item = LTChar(
            matrix,
            font,
            fontsize,
            scaling,
            rise,
            text,
            textwidth,
            textdisp,
            ncs,
            graphicstate,
        )
        self.cur_item.add(item)
        item.cid = cid  # hack 插入原字符编码
        item.font = font  # hack 插入原字符字体
        return item.adv


class Paragraph:
    def __init__(self, y, x, x0, x1, y0, y1, size, brk):
        self.y: float = y  # 初始纵坐标
        self.x: float = x  # 初始横坐标
        self.x0: float = x0  # 左边界
        self.x1: float = x1  # 右边界
        self.y0: float = y0  # 上边界
        self.y1: float = y1  # 下边界
        self.size: float = size  # 字体大小
        self.brk: bool = brk  # 换行标记


class TableCell:
    """表格单元格类"""
    def __init__(self, x0: float, y0: float, x1: float, y1: float, text: str = "", font_size: float = 12.0):
        self.x0: float = x0  # 左边界
        self.y0: float = y0  # 上边界  
        self.x1: float = x1  # 右边界
        self.y1: float = y1  # 下边界
        self.text: str = text  # 单元格文本内容
        self.font_size: float = font_size  # 字体大小
        self.chars: list = []  # 包含的字符对象
        
    def contains_point(self, x: float, y: float) -> bool:
        """判断点是否在单元格内"""
        return self.x0 <= x <= self.x1 and self.y0 <= y <= self.y1
        
    def add_char(self, char):
        """添加字符到单元格"""
        self.chars.append(char)
        self.text += char.get_text()
        if char.size > self.font_size:
            self.font_size = char.size


class TableRegion:
    """表格区域类"""
    def __init__(self, table_id: int, bbox: tuple):
        self.table_id: int = table_id
        self.x0, self.y0, self.x1, self.y1 = bbox
        self.cells: list[TableCell] = []
        self.lines: list = []  # 表格线条
        
    def extract_table_structure(self, chars: list, lines: list):
        """
        从字符和线条中提取表格结构
        使用简单的坐标分析方法识别行列结构
        """
        # 收集表格区域内的字符
        table_chars = []
        for char in chars:
            if (self.x0 <= char.x0 <= self.x1 and 
                self.y0 <= char.y0 <= self.y1):
                table_chars.append(char)
        
        logger.debug(f"[表格结构] 表格 {self.table_id} 区域内发现 {len(table_chars)} 个字符")
        
        if not table_chars:
            logger.warning(f"[表格结构] 表格 {self.table_id} 区域内未发现字符")
            return []
            
        # 收集表格区域内的线条
        table_lines = []
        for line in lines:
            line_x0, line_y0 = line.pts[0]
            line_x1, line_y1 = line.pts[1]
            if (self.x0 <= line_x0 <= self.x1 and self.y0 <= line_y0 <= self.y1 and
                self.x0 <= line_x1 <= self.x1 and self.y0 <= line_y1 <= self.y1):
                table_lines.append(line)
        
        logger.debug(f"[表格结构] 表格 {self.table_id} 区域内发现 {len(table_lines)} 条线条")
        
        # 按Y坐标对字符进行分组（识别行）
        rows = {}
        for char in table_chars:
            y_key = round(char.y0, 1)  # 使用rounded y坐标作为行的key
            if y_key not in rows:
                rows[y_key] = []
            rows[y_key].append(char)
        
        logger.debug(f"[表格结构] 表格 {self.table_id} 识别出 {len(rows)} 行")
        
        # 按行处理，进一步按X坐标分组（识别列）
        cells = []
        for row_idx, y_key in enumerate(sorted(rows.keys(), reverse=True)):  # 从上到下处理
            row_chars = sorted(rows[y_key], key=lambda c: c.x0)  # 从左到右排序
            
            # 简单的列分组：基于字符间的间距
            if not row_chars:
                continue
                
            current_cell_chars = [row_chars[0]]
            row_cells = []
            
            for i in range(1, len(row_chars)):
                char = row_chars[i]
                prev_char = current_cell_chars[-1]
                
                # 如果字符间距较大，则认为是新的单元格
                if char.x0 - prev_char.x1 > 20:  # 20是经验阈值
                    # 完成当前单元格
                    cell_text = ''.join(c.get_text() for c in current_cell_chars)
                    if cell_text.strip():  # 只处理非空单元格
                        cell_x0 = min(c.x0 for c in current_cell_chars)
                        cell_y0 = min(c.y0 for c in current_cell_chars) 
                        cell_x1 = max(c.x1 for c in current_cell_chars)
                        cell_y1 = max(c.y1 for c in current_cell_chars)
                        cell_font_size = max(c.size for c in current_cell_chars)
                        
                        cell = TableCell(cell_x0, cell_y0, cell_x1, cell_y1, cell_text, cell_font_size)
                        cell.chars = current_cell_chars.copy()
                        cells.append(cell)
                        row_cells.append(cell_text.strip())
                    
                    # 开始新单元格
                    current_cell_chars = [char]
                else:
                    current_cell_chars.append(char)
            
            # 处理最后一个单元格
            if current_cell_chars:
                cell_text = ''.join(c.get_text() for c in current_cell_chars)
                if cell_text.strip():
                    cell_x0 = min(c.x0 for c in current_cell_chars)
                    cell_y0 = min(c.y0 for c in current_cell_chars)
                    cell_x1 = max(c.x1 for c in current_cell_chars)
                    cell_y1 = max(c.y1 for c in current_cell_chars)
                    cell_font_size = max(c.size for c in current_cell_chars)
                    
                    cell = TableCell(cell_x0, cell_y0, cell_x1, cell_y1, cell_text, cell_font_size)
                    cell.chars = current_cell_chars.copy()
                    cells.append(cell)
                    row_cells.append(cell_text.strip())
            
            if row_cells:
                logger.debug(f"[表格结构] 第 {row_idx+1} 行包含 {len(row_cells)} 个单元格: {row_cells}")
        
        logger.debug(f"[表格结构] 表格 {self.table_id} 提取完成，共 {len(cells)} 个单元格")
        self.cells = cells
        return cells


class OpType(Enum):
    TEXT = "text"
    LINE = "line"


@dataclass
class CellTranslationTask:
    """单元格翻译任务数据结构"""
    cell_idx: int
    cell: 'TableCell'
    text: str
    row_idx: int
    col_idx: int
    

@dataclass
class SpatialConstraint:
    """空间约束信息"""
    max_width: float
    max_height: float
    adjacent_cells: List['TableCell']
    

@dataclass  
class TableLayoutConfig:
    """表格布局配置"""
    min_font_size: float = 6.0
    max_font_size: float = 24.0
    font_size_step: float = 0.5
    overlap_tolerance: float = 2.0  # 允许的重叠像素容忍度

class AnalysisConverter(PDFConverterEx):
    """
    A converter for analyzing PDF layout and content without translation.
    It counts pages, paragraphs, tables, cells, and tokens for both.
    """
    def __init__(self, rsrcmgr, layout={}, pages_to_process=None) -> None:
        super().__init__(rsrcmgr)
        self.layout = layout
        self.pages_to_process = pages_to_process if pages_to_process is not None else []
        self.processed_pages = set()  # 跟踪实际处理的页面
        self.stats = {
            "page_count": 0,  # 将在处理过程中更新
            "pages": {},
            "total_paragraph_tokens": 0,
            "total_table_tokens": 0,
            "total_paragraph_count": 0,
            "total_table_cell_count": 0,
        }

    def receive_layout(self, ltpage: LTPage):
        # 如果指定了页面范围，且当前页面不在范围内，则跳过
        if self.pages_to_process and ltpage.pageid not in self.pages_to_process:
            return None
            
        # 记录这个页面已被处理
        self.processed_pages.add(ltpage.pageid)
        # 更新实际处理的页面数
        self.stats["page_count"] = len(self.processed_pages)
            
        page_items = list(ltpage)
        all_chars = [item for item in page_items if isinstance(item, LTChar)]
        all_lines = [item for item in page_items if isinstance(item, LTLine)]

        # 1. Analyze Tables first
        table_regions = []
        table_cells = []
        table_token_count = 0
        if 'table_regions' in self.layout and ltpage.pageid in self.layout['table_regions']:
            table_layout_info = self.layout['table_regions'][ltpage.pageid]
            for table_info_item in table_layout_info:
                table_id = table_info_item['id']
                bbox = table_info_item['bbox']
                table_region = TableRegion(table_id, bbox)
                # Use a copy of all_chars and all_lines for each table if needed,
                # but here they are just read, so it's fine.
                cells = table_region.extract_table_structure(all_chars, all_lines)
                if cells:
                    table_cells.extend(cells)
                    table_regions.append(table_region)

        for cell in table_cells:
            table_token_count += count_tokens(cell.text.strip())

        # 2. Analyze Paragraphs
        sstk: list[str] = []
        pstk: list[Paragraph] = []
        vstk: list[LTChar] = []
        xt: LTChar = None
        xt_cls: int = -1
        vmax: float = ltpage.width / 4

        def vflag(font: str, char: str):
            if isinstance(font, bytes):
                try: font = font.decode('utf-8')
                except UnicodeDecodeError: font = ""
            font = font.split("+")[-1]
            if re.match(r"\(cid:", char): return True
            if re.match(r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)", font): return True
            if (char and char != " " and (unicodedata.category(char[0]) in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"] or ord(char[0]) in range(0x370, 0x400))): return True
            return False

        for child in page_items:
            if not isinstance(child, LTChar): continue
            
            cur_v = False
            layout_map = self.layout[ltpage.pageid]
            h, w = layout_map.shape
            cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
            cls = layout_map[cy, cx]
            if cls < 0: continue # Skip chars in table regions

            if child.get_text() == "•": cls = 0
            if (cls == 0 or (sstk and cls == xt_cls and len(sstk[-1].strip()) > 1 and child.size < pstk[-1].size * 0.79) or vflag(child.fontname, child.get_text()) or (child.matrix[0] == 0 and child.matrix[3] == 0)):
                cur_v = True
            
            if not cur_v:
                if vstk and child.get_text() in "()": cur_v = True
            
            if (not cur_v or cls != xt_cls or (sstk and sstk[-1] != "" and abs(child.x0 - xt.x0) > vmax)):
                if vstk:
                    if sstk and sstk[-1] == "": xt_cls = -1
                    if sstk: sstk[-1] += "{v}"
                    vstk = []

            if not vstk:
                if cls == xt_cls:
                    if child.x0 > xt.x1 + 1: sstk[-1] += " "
                    elif child.x1 < xt.x0: sstk[-1] += " "; pstk[-1].brk = True
                else:
                    sstk.append("")
                    pstk.append(Paragraph(child.y0, child.x0, child.x0, child.x0, child.y0, child.y1, child.size, False))
            
            if not cur_v:
                if (child.size > pstk[-1].size or len(sstk[-1].strip()) == 1) and child.get_text() != " ":
                    pstk[-1].y -= child.size - pstk[-1].size
                    pstk[-1].size = child.size
                sstk[-1] += child.get_text()
            else:
                vstk.append(child)
            
            pstk[-1].x0 = min(pstk[-1].x0, child.x0); pstk[-1].x1 = max(pstk[-1].x1, child.x1)
            pstk[-1].y0 = min(pstk[-1].y0, child.y0); pstk[-1].y1 = max(pstk[-1].y1, child.y1)
            xt = child
            xt_cls = cls

        if vstk:
            if sstk: sstk[-1] += "{v}"

        # 3. Calculate and store stats
        paragraph_token_count = 0
        paragraph_count = 0
        for para_text in sstk:
            clean_text = para_text.strip()
            if clean_text and not re.fullmatch(r"\{v\d*\}", clean_text):
                paragraph_count += 1
                paragraph_token_count += count_tokens(clean_text)

        self.stats['pages'][ltpage.pageid] = {
            'paragraph_count': paragraph_count,
            'table_count': len(table_regions),
            'table_cell_count': len(table_cells),
            'paragraph_token_count': paragraph_token_count,
            'table_token_count': table_token_count
        }
        self.stats['total_paragraph_tokens'] += paragraph_token_count
        self.stats['total_table_tokens'] += table_token_count
        self.stats['total_paragraph_count'] += paragraph_count
        self.stats['total_table_cell_count'] += len(table_cells)
        return None


# fmt: off
class TranslateConverter(PDFConverterEx):
    def __init__(
        self,
        rsrcmgr,
        vfont: str = None,
        vchar: str = None,
        thread: int = 0,
        layout={},
        lang_in: str = "",
        lang_out: str = "",
        service: str = "",
        noto_name: str = "",
        noto: Font = None,
        envs: Dict = None,
        prompt: Template = None,
        ignore_cache: bool = False,
        use_concurrent_table_translation: bool = True,
    ) -> None:
        super().__init__(rsrcmgr)
        self.vfont = vfont
        self.vchar = vchar
        self.thread = thread
        self.layout = layout
        self.noto_name = noto_name
        self.noto = noto
        self.translator: BaseTranslator = None
        self.use_concurrent_table_translation = use_concurrent_table_translation
        # e.g. "ollama:gemma2:9b" -> ["ollama", "gemma2:9b"]
        param = service.split(":", 1)
        service_name = param[0]
        service_model = param[1] if len(param) > 1 else None
        if not envs:
            envs = {}
        for translator in [GoogleTranslator, BingTranslator, DeepLTranslator, DeepLXTranslator, OllamaTranslator, XinferenceTranslator, AzureOpenAITranslator,
                           OpenAITranslator, ZhipuTranslator, ModelScopeTranslator, SiliconTranslator, GeminiTranslator, AzureTranslator, TencentTranslator, DifyTranslator, AnythingLLMTranslator, ArgosTranslator, GrokTranslator, GroqTranslator, DeepseekTranslator, OpenAIlikedTranslator, DashScopeTranslator, QwenMtTranslator,]:
            if service_name == translator.name:
                self.translator = translator(lang_in, lang_out, service_model, envs=envs, prompt=prompt, ignore_cache=ignore_cache)
        if not self.translator:
            raise ValueError("Unsupported translation service")

        # 添加段落统计信息
        self.paragraph_stats = {
            "total_paragraphs": 0,  # 总段落数
            "skipped_empty": 0,     # 跳过的空白段落数
            "skipped_formula": 0,   # 跳过的公式段落数
            "skipped_no_text": 0,   # 跳过的无中英文段落数
            "translated": 0,         # 实际翻译的段落数
        }
        
        # 添加表格单元格统计信息
        self.table_stats = {
            "total_cells": 0,       # 总单元格数
            "skipped_empty": 0,     # 跳过的空白单元格数
            "skipped_no_text": 0,   # 跳过的无中英文单元格数
            "translated": 0,         # 实际翻译的单元格数
        }
        
        # 初始化并发表格翻译器
        self.concurrent_table_translator = None
        self.serial_table_translator = None

    def get_paragraph_stats(self) -> dict:
        """获取段落统计信息"""
        return self.paragraph_stats.copy()
        
    def get_table_stats(self) -> dict:
        """获取表格统计信息"""
        return self.table_stats.copy()

    def receive_layout(self, ltpage: LTPage):
        # 段落
        sstk: list[str] = []            # 段落文字栈
        pstk: list[Paragraph] = []      # 段落属性栈
        vbkt: int = 0                   # 段落公式括号计数
        # 公式组
        vstk: list[LTChar] = []         # 公式符号组
        vlstk: list[LTLine] = []        # 公式线条组
        vfix: float = 0                 # 公式纵向偏移
        # 公式组栈
        var: list[list[LTChar]] = []    # 公式符号组栈
        varl: list[list[LTLine]] = []   # 公式线条组栈
        varf: list[float] = []          # 公式纵向偏移栈
        vlen: list[float] = []          # 公式宽度栈
        # 全局
        lstk: list[LTLine] = []         # 全局线条栈
        xt: LTChar = None               # 上一个字符
        xt_cls: int = -1                # 上一个字符所属段落，保证无论第一个字符属于哪个类别都可以触发新段落
        vmax: float = ltpage.width / 4  # 行内公式最大宽度
        ops: str = ""                   # 渲染结果
        
        # 表格处理
        table_regions = {}              # 存储表格区域
        table_cells = {}                # 存储表格单元格翻译结果
        
        # 获取表格区域信息
        layout = self.layout[ltpage.pageid]
        if 'table_regions' in self.layout and ltpage.pageid in self.layout['table_regions']:
            table_info = self.layout['table_regions'][ltpage.pageid]
            
            # 收集所有字符和线条用于表格分析
            all_chars = []
            all_lines = []
            for child in ltpage:
                if isinstance(child, LTChar):
                    all_chars.append(child)
                elif isinstance(child, LTLine):
                    all_lines.append(child)
            
            # 处理每个表格区域
            for table_info_item in table_info:
                table_id = table_info_item['id']
                bbox = table_info_item['bbox']
                
                # 创建表格区域对象并提取结构
                table_region = TableRegion(table_id, bbox)
                cells = table_region.extract_table_structure(all_chars, all_lines)
                
                if cells:
                    self.table_stats["total_cells"] += len(cells)
                    if self.use_concurrent_table_translation:
                        if self.concurrent_table_translator is None:
                            self.concurrent_table_translator = ConcurrentTableTranslator(
                                translator=self.translator,
                                fontmap=self.fontmap,
                                noto=self.noto,
                                noto_name=self.noto_name,
                                thread_count=self.thread,
                            )
                        try:
                            table_region = self.concurrent_table_translator.translate_table_concurrent(table_region, self.table_stats)
                            table_regions[table_id] = table_region
                            logger.info(f"完成表格 {table_id} 的并发翻译，共处理 {len(cells)} 个单元格")
                        except Exception as e:
                            logger.error(f"表格 {table_id} 并发翻译失败，回退到串行翻译: {e}")
                            if self.serial_table_translator is None:
                                self.serial_table_translator = SerialTableTranslator(translator=self.translator)
                            table_region = self.serial_table_translator.translate_table(table_region, self.table_stats)
                            table_regions[table_id] = table_region
                    else:
                        if self.serial_table_translator is None:
                            self.serial_table_translator = SerialTableTranslator(translator=self.translator)
                        table_region = self.serial_table_translator.translate_table(table_region, self.table_stats)
                        table_regions[table_id] = table_region

        def vflag(font: str, char: str):    # 匹配公式（和角标）字体
            if isinstance(font, bytes):     # 不一定能 decode，直接转 str
                try:
                    font = font.decode('utf-8')  # 尝试使用 UTF-8 解码
                except UnicodeDecodeError:
                    font = ""
            font = font.split("+")[-1]      # 字体名截断
            if re.match(r"\(cid:", char):
                return True
            # 基于字体名规则的判定
            if self.vfont:
                if re.match(self.vfont, font):
                    return True
            else:
                if re.match(                                            # latex 字体
                    r"(CM[^R]|MS.M|XY|MT|BL|RM|EU|LA|RS|LINE|LCIRCLE|TeX-|rsfs|txsy|wasy|stmary|.*Mono|.*Code|.*Ital|.*Sym|.*Math)",
                    font,
                ):
                    return True
            # 基于字符集规则的判定
            if self.vchar:
                if re.match(self.vchar, char):
                    return True
            else:
                if (
                    char
                    and char != " "                                     # 非空格
                    and (
                        unicodedata.category(char[0])
                        in ["Lm", "Mn", "Sk", "Sm", "Zl", "Zp", "Zs"]   # 文字修饰符、数学符号、分隔符号
                        or ord(char[0]) in range(0x370, 0x400)          # 希腊字母
                    )
                ):
                    return True
            return False

        ############################################################
        # A. 原文档解析
        for child in ltpage:
            if isinstance(child, LTChar):
                cur_v = False
                layout = self.layout[ltpage.pageid]
                # ltpage.height 可能是 fig 里面的高度，这里统一用 layout.shape
                h, w = layout.shape
                # 读取当前字符在 layout 中的类别
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                
                # 跳过表格区域的字符，它们会在表格处理中单独处理
                if cls < 0:  # 表格区域使用负数ID标识
                    continue
                    
                # 锚定文档中 bullet 的位置
                if child.get_text() == "•":
                    cls = 0
                # 判定当前字符是否属于公式
                if (                                                                                        # 判定当前字符是否属于公式
                    cls == 0                                                                                # 1. 类别为保留区域
                    or (cls == xt_cls and len(sstk[-1].strip()) > 1 and child.size < pstk[-1].size * 0.79)  # 2. 角标字体，有 0.76 的角标和 0.799 的大写，这里用 0.79 取中，同时考虑首字母放大的情况
                    or vflag(child.fontname, child.get_text())                                              # 3. 公式字体
                    or (child.matrix[0] == 0 and child.matrix[3] == 0)                                      # 4. 垂直字体
                ):
                    cur_v = True
                # 判定括号组是否属于公式
                if not cur_v:
                    if vstk and child.get_text() == "(":
                        cur_v = True
                        vbkt += 1
                    if vbkt and child.get_text() == ")":
                        cur_v = True
                        vbkt -= 1
                if (                                                        # 判定当前公式是否结束
                    not cur_v                                               # 1. 当前字符不属于公式
                    or cls != xt_cls                                        # 2. 当前字符与前一个字符不属于同一段落
                    # or (abs(child.x0 - xt.x0) > vmax and cls != 0)        # 3. 段落内换行，可能是一长串斜体的段落，也可能是段内分式换行，这里设个阈值进行区分
                    # 禁止纯公式（代码）段落换行，直到文字开始再重开文字段落，保证只存在两种情况
                    # A. 纯公式（代码）段落（锚定绝对位置）sstk[-1]=="" -> sstk[-1]=="{v*}"
                    # B. 文字开头段落（排版相对位置）sstk[-1]!=""
                    or (sstk[-1] != "" and abs(child.x0 - xt.x0) > vmax)    # 因为 cls==xt_cls==0 一定有 sstk[-1]==""，所以这里不需要再判定 cls!=0
                ):
                    if vstk:
                        if (                                                # 根据公式右侧的文字修正公式的纵向偏移
                            not cur_v                                       # 1. 当前字符不属于公式
                            and cls == xt_cls                               # 2. 当前字符与前一个字符属于同一段落
                            and child.x0 > max([vch.x0 for vch in vstk])    # 3. 当前字符在公式右侧
                        ):
                            vfix = vstk[0].y0 - child.y0
                        if sstk[-1] == "":
                            xt_cls = -1 # 禁止纯公式段落（sstk[-1]=="{v*}"）的后续连接，但是要考虑新字符和后续字符的连接，所以这里修改的是上个字符的类别
                        sstk[-1] += f"{{v{len(var)}}}"
                        var.append(vstk)
                        varl.append(vlstk)
                        varf.append(vfix)
                        vstk = []
                        vlstk = []
                        vfix = 0
                # 当前字符不属于公式或当前字符是公式的第一个字符
                if not vstk:
                    if cls == xt_cls:               # 当前字符与前一个字符属于同一段落
                        if child.x0 > xt.x1 + 1:    # 添加行内空格
                            sstk[-1] += " "
                        elif child.x1 < xt.x0:      # 添加换行空格并标记原文段落存在换行
                            sstk[-1] += " "
                            pstk[-1].brk = True
                    else:                           # 根据当前字符构建一个新的段落
                        sstk.append("")
                        pstk.append(Paragraph(child.y0, child.x0, child.x0, child.x0, child.y0, child.y1, child.size, False))
                if not cur_v:                                               # 文字入栈
                    if (                                                    # 根据当前字符修正段落属性
                        child.size > pstk[-1].size                          # 1. 当前字符比段落字体大
                        or len(sstk[-1].strip()) == 1                       # 2. 当前字符为段落第二个文字（考虑首字母放大的情况）
                    ) and child.get_text() != " ":                          # 3. 当前字符不是空格
                        pstk[-1].y -= child.size - pstk[-1].size            # 修正段落初始纵坐标，假设两个不同大小字符的上边界对齐
                        pstk[-1].size = child.size
                    sstk[-1] += child.get_text()
                else:                                                       # 公式入栈
                    if (                                                    # 根据公式左侧的文字修正公式的纵向偏移
                        not vstk                                            # 1. 当前字符是公式的第一个字符
                        and cls == xt_cls                                   # 2. 当前字符与前一个字符属于同一段落
                        and child.x0 > xt.x0                                # 3. 前一个字符在公式左侧
                    ):
                        vfix = child.y0 - xt.y0
                    vstk.append(child)
                # 更新段落边界，因为段落内换行之后可能是公式开头，所以要在外边处理
                pstk[-1].x0 = min(pstk[-1].x0, child.x0)
                pstk[-1].x1 = max(pstk[-1].x1, child.x1)
                pstk[-1].y0 = min(pstk[-1].y0, child.y0)
                pstk[-1].y1 = max(pstk[-1].y1, child.y1)
                # 更新上一个字符
                xt = child
                xt_cls = cls
            elif isinstance(child, LTFigure):   # 图表
                pass
            elif isinstance(child, LTLine):     # 线条
                layout = self.layout[ltpage.pageid]
                # ltpage.height 可能是 fig 里面的高度，这里统一用 layout.shape
                h, w = layout.shape
                # 读取当前线条在 layout 中的类别
                cx, cy = np.clip(int(child.x0), 0, w - 1), np.clip(int(child.y0), 0, h - 1)
                cls = layout[cy, cx]
                if vstk and cls == xt_cls:      # 公式线条
                    vlstk.append(child)
                else:                           # 全局线条
                    lstk.append(child)
            else:
                pass
        # 处理结尾
        if vstk:    # 公式出栈
            sstk[-1] += f"{{v{len(var)}}}"
            var.append(vstk)
            varl.append(vlstk)
            varf.append(vfix)
        logger.debug("\n==========[VSTACK]==========\n")
        for id, v in enumerate(var):  # 计算公式宽度
            l = max([vch.x1 for vch in v]) - v[0].x0
            logger.debug(f'< {l:.1f} {v[0].x0:.1f} {v[0].y0:.1f} {v[0].cid} {v[0].fontname} {len(varl[id])} > v{id} = {"".join([ch.get_text() for ch in v])}')
            vlen.append(l)

        ############################################################
        # B. 段落翻译
        logger.debug("\n==========[SSTACK]==========\n")

        @retry(wait=wait_fixed(1))
        def worker(s: str):  # 多线程翻译
            # 更新总段落计数
            self.paragraph_stats["total_paragraphs"] += 1
            
            # 检查并统计跳过的情况
            if not s.strip():
                self.paragraph_stats["skipped_empty"] += 1
                return s
            if re.match(r"^\{v\d+\}$", s):
                self.paragraph_stats["skipped_formula"] += 1
                return s
            if not re.search(r'[\u4e00-\u9fff]|[a-zA-Z]', s):
                self.paragraph_stats["skipped_no_text"] += 1
                return s
                
            try:
                logger.debug(f"[段落翻译] 输入: '{s.strip()}'")
                new = self.translator.translate(s)
                logger.debug(f"[段落翻译] 输出: '{new.strip()}'")
                # 更新已翻译段落计数
                self.paragraph_stats["translated"] += 1
                return new
            except BaseException as e:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.exception(e)
                else:
                    logger.exception(e, exc_info=False)
                raise e
        
        logger.info(f"开始翻译 {len(sstk)} 个段落")
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=self.thread
        ) as executor:
            news = list(executor.map(worker, sstk))
        logger.debug(f"完成段落翻译")

        ############################################################
        # C. 新文档排版
        def raw_string(fcur: str, cstk: str):  # 编码字符串
            if fcur == self.noto_name:
                return "".join(["%04x" % self.noto.has_glyph(ord(c)) for c in cstk])
            elif isinstance(self.fontmap[fcur], PDFCIDFont):  # 判断编码长度
                return "".join(["%04x" % ord(c) for c in cstk])
            else:
                return "".join(["%02x" % ord(c) for c in cstk])

        # 根据目标语言获取默认行距
        LANG_LINEHEIGHT_MAP = {
            "zh-cn": 1.4, "zh-tw": 1.4, "zh-hans": 1.4, "zh-hant": 1.4, "zh": 1.4,
            "ja": 1.1, "ko": 1.2, "en": 1.2, "ar": 1.0, "ru": 0.8, "uk": 0.8, "ta": 0.8
        }
        default_line_height = LANG_LINEHEIGHT_MAP.get(self.translator.lang_out.lower(), 1.1) # 小语种默认1.1
        _x, _y = 0, 0
        ops_list = []

        def gen_op_txt(font, size, x, y, rtxt):
            return f"/{font} {size:f} Tf 1 0 0 1 {x:f} {y:f} Tm [<{rtxt}>] TJ "

        def gen_op_line(x, y, xlen, ylen, linewidth):
            return f"ET q 1 0 0 1 {x:f} {y:f} cm [] 0 d 0 J {linewidth:f} w 0 0 m {xlen:f} {ylen:f} l S Q BT "

        for id, new in enumerate(news):
            x: float = pstk[id].x                       # 段落初始横坐标
            y: float = pstk[id].y                       # 段落初始纵坐标
            x0: float = pstk[id].x0                     # 段落左边界
            x1: float = pstk[id].x1                     # 段落右边界
            height: float = pstk[id].y1 - pstk[id].y0   # 段落高度
            size: float = pstk[id].size                 # 段落字体大小
            brk: bool = pstk[id].brk                    # 段落换行标记
            cstk: str = ""                              # 当前文字栈
            fcur: str = None                            # 当前字体 ID
            lidx = 0                                    # 记录换行次数
            tx = x
            fcur_ = fcur
            ptr = 0
            logger.debug(f"< {y} {x} {x0} {x1} {size} {brk} > {sstk[id]} | {new}")

            ops_vals: list[dict] = []
            
            # [MODIFIED] 动态调整字号以适应翻译后的文本长度
            final_size = size
            final_lidx = 0
            
            def calculate_layout(font_size, text_to_layout, has_brk):
                """预计算布局，返回需要的行数"""
                _lidx = 0
                _x = x0
                _ptr = 0
                _fcur_ = None
                # 如果原文是单行，允许译文自动换行
                if not has_brk:
                    has_brk = True

                while _ptr < len(text_to_layout):
                    _vy_regex = re.match(r"\{\s*v([\d\s]+)\}", text_to_layout[_ptr:], re.IGNORECASE)
                    if _vy_regex:
                        _ptr += len(_vy_regex.group(0))
                        try:
                            _vid = int(_vy_regex.group(1).replace(" ", ""))
                            _adv = vlen[_vid]
                        except Exception:
                            continue
                    else:
                        _ch = text_to_layout[_ptr]
                        _fcur_ = None
                        try:
                            if self.fontmap["tiro"].to_unichr(ord(_ch)) == _ch: _fcur_ = "tiro"
                        except Exception: pass
                        if _fcur_ is None: _fcur_ = self.noto_name
                        
                        if _fcur_ == self.noto_name:
                            _adv = self.noto.char_lengths(_ch, font_size)[0]
                        else:
                            _adv = self.fontmap[_fcur_].char_width(ord(_ch)) * font_size
                        _ptr += 1
                    
                    if has_brk and _x + _adv > x1 + 0.1 * font_size and _x > x0:
                        _x = x0
                        _lidx += 1
                    _x += _adv
                return _lidx

            # 迭代寻找合适的字号
            # 首先，使用原始字号进行一次预计算
            lidx = calculate_layout(final_size, new, brk)
            required_height = (lidx + 1) * final_size * default_line_height

            # 如果高度超出，则逐步减小字号
            if required_height > height + size * 0.2: # 增加一点容忍度
                 while final_size > 6: # 设置最小字号为6pt
                    final_size -= 0.5
                    lidx = calculate_layout(final_size, new, brk)
                    required_height = (lidx + 1) * final_size * default_line_height
                    if required_height <= height + final_size * 0.2:
                        break # 找到合适的字号
            
            # 使用最终确定的字号和行数进行排版
            size = final_size
            lidx = 0 # 重置 lidx 给实际排版使用
            if not brk: # 如果原文不换行，则强制换行
                brk = True

            while ptr < len(new):
                vy_regex = re.match(
                    r"\{\s*v([\d\s]+)\}", new[ptr:], re.IGNORECASE
                )  # 匹配 {vn} 公式标记
                mod = 0  # 文字修饰符
                if vy_regex:  # 加载公式
                    ptr += len(vy_regex.group(0))
                    try:
                        vid = int(vy_regex.group(1).replace(" ", ""))
                        adv = vlen[vid]
                    except Exception:
                        continue  # 翻译器可能会自动补个越界的公式标记
                    if var[vid][-1].get_text() and unicodedata.category(var[vid][-1].get_text()[0]) in ["Lm", "Mn", "Sk"]:  # 文字修饰符
                        mod = var[vid][-1].width
                else:  # 加载文字
                    ch = new[ptr]
                    fcur_ = None
                    try:
                        if fcur_ is None and self.fontmap["tiro"].to_unichr(ord(ch)) == ch:
                            fcur_ = "tiro"  # 默认拉丁字体
                    except Exception:
                        pass
                    if fcur_ is None:
                        fcur_ = self.noto_name  # 默认非拉丁字体
                    if fcur_ == self.noto_name: # FIXME: change to CONST
                        adv = self.noto.char_lengths(ch, size)[0]
                    else:
                        adv = self.fontmap[fcur_].char_width(ord(ch)) * size
                    ptr += 1
                if (                                # 输出文字缓冲区
                    fcur_ != fcur                   # 1. 字体更新
                    or vy_regex                     # 2. 插入公式
                    or x + adv > x1 + 0.1 * size    # 3. 到达右边界（可能一整行都被符号化，这里需要考虑浮点误差）
                ):
                    if cstk:
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": fcur,
                            "size": size,
                            "x": tx,
                            "dy": 0,
                            "rtxt": raw_string(fcur, cstk),
                            "lidx": lidx
                        })
                        cstk = ""
                if brk and x + adv > x1 + 0.1 * size and x > x0:  # 到达右边界且原文段落存sdDYPK8s1VzC在换行. x > x0 避免单字符就换行
                    x = x0
                    lidx += 1
                if vy_regex:  # 插入公式
                    fix = 0
                    if fcur is not None:  # 段落内公式修正纵向偏移
                        fix = varf[vid]
                    for vch in var[vid]:  # 排版公式字符
                        vc = chr(vch.cid)
                        ops_vals.append({
                            "type": OpType.TEXT,
                            "font": self.fontid[vch.font],
                            "size": vch.size,
                            "x": x + vch.x0 - var[vid][0].x0,
                            "dy": fix + vch.y0 - var[vid][0].y0,
                            "rtxt": raw_string(self.fontid[vch.font], vc),
                            "lidx": lidx
                        })
                        if logger.isEnabledFor(logging.DEBUG):
                            lstk.append(LTLine(0.1, (_x, _y), (x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0)))
                            _x, _y = x + vch.x0 - var[vid][0].x0, fix + y + vch.y0 - var[vid][0].y0
                    for l in varl[vid]:  # 排版公式线条
                        if l.linewidth < 5:  # hack 有的文档会用粗线条当图片背景
                            ops_vals.append({
                                "type": OpType.LINE,
                                "x": l.pts[0][0] + x - var[vid][0].x0,
                                "dy": l.pts[0][1] + fix - var[vid][0].y0,
                                "linewidth": l.linewidth,
                                "xlen": l.pts[1][0] - l.pts[0][0],
                                "ylen": l.pts[1][1] - l.pts[0][1],
                                "lidx": lidx
                            })
                else:  # 插入文字缓冲区
                    if not cstk:  # 单行开头
                        tx = x
                        if x == x0 and ch == " ":  # 消除段落换行空格
                            adv = 0
                        else:
                            cstk += ch
                    else:
                        cstk += ch
                adv -= mod # 文字修饰符
                fcur = fcur_
                x += adv
                if logger.isEnabledFor(logging.DEBUG):
                    lstk.append(LTLine(0.1, (_x, _y), (x, y)))
                    _x, _y = x, y
            # 处理结尾
            if cstk:
                ops_vals.append({
                    "type": OpType.TEXT,
                    "font": fcur,
                    "size": size,
                    "x": tx,
                    "dy": 0,
                    "rtxt": raw_string(fcur, cstk),
                    "lidx": lidx
                })

            line_height = default_line_height

            while (lidx + 1) * size * line_height > height and line_height >= 1:
                line_height -= 0.05

            for vals in ops_vals:
                if vals["type"] == OpType.TEXT:
                    ops_list.append(gen_op_txt(vals["font"], vals["size"], vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["rtxt"]))
                elif vals["type"] == OpType.LINE:
                    ops_list.append(gen_op_line(vals["x"], vals["dy"] + y - vals["lidx"] * size * line_height, vals["xlen"], vals["ylen"], vals["linewidth"]))

        for l in lstk:  # 排版全局线条
            if l.linewidth < 5:  # hack 有的文档会用粗线条当图片背景
                ops_list.append(gen_op_line(l.pts[0][0], l.pts[0][1], l.pts[1][0] - l.pts[0][0], l.pts[1][1] - l.pts[0][1], l.linewidth))

        ############################################################
        # D. 表格排版（支持并发翻译优化结果）
        translation_mode = "并发" if self.use_concurrent_table_translation else "串行"
        logger.info(f"开始排版 {len(table_regions)} 个表格区域 ({translation_mode}模式)")
        for table_id, table_region in table_regions.items():
            logger.info(f"排版表格 {table_id}，包含 {len(table_region.cells)} 个单元格")
            for cell_idx, cell in enumerate(table_region.cells):
                if hasattr(cell, 'translated_text') and cell.translated_text:
                    # 对每个单元格进行排版
                    cell_x = cell.x0
                    cell_y = cell.y0
                    text = cell.translated_text
                    
                    # 使用优化后的字体和大小（如果可用），否则使用默认值
                    if hasattr(cell, 'optimized_font') and cell.optimized_font:
                        fcur = cell.optimized_font
                        logger.debug(f"[表格排版] 单元格 {cell_idx+1} 使用优化字体: {fcur}")
                    else:
                        # 传统字体选择逻辑
                        fcur = None
                    if cell.chars:
                        try:
                            if self.fontmap["tiro"].to_unichr(ord(text[0])) == text[0]:
                                fcur = "tiro"
                        except Exception:
                            pass
                    if fcur is None:
                        fcur = self.noto_name
                    
                    # 使用优化后的字体大小（如果经过空间优化）
                    cell_size = cell.font_size
                    
                    # 生成文本操作指令
                    rtxt = raw_string(fcur, text)
                    ops_list.append(gen_op_txt(fcur, cell_size, cell_x, cell_y, rtxt))
                    
                    # 增强的调试信息
                    optimization_info = ""
                    if hasattr(cell, 'optimized_font'):
                        optimization_info = f" [优化后: 字体={getattr(cell, 'optimized_font', 'N/A')}, 大小={cell_size:.1f}]"
                    
                    logger.debug(f"[表格排版] 单元格 {cell_idx+1}: '{text}' 位置({cell_x:.1f}, {cell_y:.1f}) 字体:{fcur} 大小:{cell_size:.1f}{optimization_info}")
                    logger.debug(f"Table cell rendered: '{text}' at ({cell_x:.1f}, {cell_y:.1f}) size {cell_size:.1f}")
        
        if len(table_regions) > 0:
            logger.info(f"完成 {len(table_regions)} 个表格的排版 ({translation_mode}模式)")

        ops = f"BT {''.join(ops_list)}ET "
        return ops
    

class SerialTableTranslator:
    """
    串行表格翻译器
    
    负责处理表格单元格的串行翻译。
    这是一个简单的实现，用于与并发翻译器对齐。
    """
    def __init__(self, translator: BaseTranslator):
        self.translator = translator

    def translate_table(self, table_region: 'TableRegion', table_stats: dict) -> 'TableRegion':
        """
        串行翻译一个表格区域。
        
        Args:
            table_region: 要翻译的表格区域对象。
            table_stats: 用于累计统计信息的可变字典。
            
        Returns:
            翻译完成的表格区域对象。
        """
        cells = table_region.cells
        logger.info(f"开始串行翻译表格 {table_region.table_id} 的 {len(cells)} 个单元格")
        table_stats["total_cells"] += len(cells)

        for cell in cells:
            cell_text = cell.text.strip()
            if not cell_text:
                table_stats["skipped_empty"] += 1
                cell.translated_text = cell_text
                continue
            
            if re.search(r'[\u4e00-\u9fff]|[a-zA-Z]', cell_text):
                try:
                    translated_text = self.translator.translate(cell_text)
                    cell.translated_text = translated_text
                    table_stats["translated"] += 1
                except Exception as e:
                    logger.warning(f"Failed to translate table cell '{cell_text}': {e}")
                    cell.translated_text = cell_text
            else:
                table_stats["skipped_no_text"] += 1
                cell.translated_text = cell_text
        
        logger.info(f"表格 {table_region.table_id} 的串行翻译完成，共处理 {len(cells)} 个单元格")
        return table_region
    

class ConcurrentTableTranslator:
    """
    并发表格翻译器
    
    负责处理表格单元格的并发翻译，同时保证：
    1. 空间冲突检测与解决
    2. 字体一致性管理  
    3. 表格结构完整性
    4. 线程安全性
    """
    
    def __init__(self, translator: BaseTranslator, fontmap: dict, noto: Font, noto_name: str, thread_count: int = 0):
        self.translator = translator
        self.fontmap = fontmap
        self.noto = noto
        self.noto_name = noto_name
        self.thread_count = thread_count
        self.layout_config = TableLayoutConfig()
        self._lock = threading.Lock()
        
    def translate_table_concurrent(self, table_region: 'TableRegion', table_stats: dict) -> 'TableRegion':
        """
        并发翻译表格区域
        
        Args:
            table_region: 表格区域对象
            table_stats: 用于累计统计信息的可变字典。
            
        Returns:
            翻译完成的表格区域对象
        """
        cells = table_region.cells
        if not cells:
            return table_region
            
        logger.info(f"开始并发翻译表格 {table_region.table_id}，共 {len(cells)} 个单元格")
        table_stats["total_cells"] += len(cells)
        
        # 1. 预处理：分析表格结构和空间约束，并准备翻译任务
        tasks = []
        grid_layout = self._analyze_grid_structure(cells)
        
        for cell_idx, cell in enumerate(cells):
            cell_text = cell.text.strip()
            # 处理不需要翻译的单元格并更新统计
            if not cell_text:
                table_stats["skipped_empty"] += 1
                cell.translated_text = ""
                continue
            if not re.search(r'[\u4e00-\u9fff]|[a-zA-Z]', cell_text):
                table_stats["skipped_no_text"] += 1
                cell.translated_text = cell_text
                continue
            
            # 为需要翻译的单元格创建任务
            row_idx, col_idx = self._find_cell_position(cell, grid_layout)
            task = CellTranslationTask(
                cell_idx=cell_idx,
                cell=cell,
                text=cell_text,
                row_idx=row_idx,
                col_idx=col_idx
            )
            tasks.append(task)
        
        logger.debug(f"准备了 {len(tasks)} 个翻译任务")
        
        # 2. 并发翻译阶段
        translated_tasks = self._execute_concurrent_translation(tasks, table_stats)
        
        # 3. 后处理：空间冲突检测和布局优化
        optimized_cells = self._optimize_layout(translated_tasks, grid_layout, table_region)
        
        # 4. 更新表格区域
        table_region.cells = optimized_cells
        
        logger.info(f"完成表格 {table_region.table_id} 的并发翻译和布局优化")
        return table_region
        
    def _prepare_translation_tasks(self, cells: List['TableCell']) -> Tuple[List[CellTranslationTask], dict]:
        """
        准备翻译任务和网格布局分析
        
        Returns:
            tuple: (翻译任务列表, 网格布局信息)
        """
        tasks = []
        grid_layout = self._analyze_grid_structure(cells)
        
        for cell_idx, cell in enumerate(cells):
            cell_text = cell.text.strip()
            if not cell_text:
                continue
                
            if re.search(r'[\u4e00-\u9fff]|[a-zA-Z]', cell_text):
                row_idx, col_idx = self._find_cell_position(cell, grid_layout)
                task = CellTranslationTask(
                    cell_idx=cell_idx,
                    cell=cell,
                    text=cell_text,
                    row_idx=row_idx,
                    col_idx=col_idx
                )
                tasks.append(task)
                
        logger.debug(f"准备了 {len(tasks)} 个翻译任务")
        return tasks, grid_layout
        
    def _analyze_grid_structure(self, cells: List['TableCell']) -> dict:
        """
        分析表格的网格结构
        
        Returns:
            dict: 包含行列信息的网格布局
        """
        # 按Y坐标分组识别行
        rows = {}
        for cell in cells:
            y_key = round(cell.y0, 1)
            if y_key not in rows:
                rows[y_key] = []
            rows[y_key].append(cell)
            
        # 按X坐标分组识别列  
        cols = {}
        for cell in cells:
            x_key = round(cell.x0, 1)
            if x_key not in cols:
                cols[x_key] = []
            cols[x_key].append(cell)
            
        # 构建网格映射
        sorted_row_keys = sorted(rows.keys(), reverse=True)
        sorted_col_keys = sorted(cols.keys())
        
        grid_layout = {
            'rows': {i: rows[key] for i, key in enumerate(sorted_row_keys)},
            'cols': {i: cols[key] for i, key in enumerate(sorted_col_keys)},
            'row_keys': sorted_row_keys,
            'col_keys': sorted_col_keys
        }
        
        logger.debug(f"分析得到网格结构: {len(sorted_row_keys)} 行 x {len(sorted_col_keys)} 列")
        return grid_layout
        
    def _find_cell_position(self, cell: 'TableCell', grid_layout: dict) -> Tuple[int, int]:
        """查找单元格在网格中的位置"""
        # 简化的位置查找算法
        row_idx = 0
        col_idx = 0
        
        # 查找行位置
        for i, row_key in enumerate(grid_layout['row_keys']):
            if abs(cell.y0 - row_key) < 5:  # 5像素的容忍度
                row_idx = i
                break
                
        # 查找列位置  
        for i, col_key in enumerate(grid_layout['col_keys']):
            if abs(cell.x0 - col_key) < 5:  # 5像素的容忍度
                col_idx = i
                break
                
        return row_idx, col_idx
        
    def _execute_concurrent_translation(self, tasks: List[CellTranslationTask], table_stats: dict) -> List[CellTranslationTask]:
        """
        执行并发翻译
        
        Args:
            tasks: 翻译任务列表
            table_stats: 用于累计统计信息的可变字典。
            
        Returns:
            完成翻译的任务列表
        """
        if not tasks:
            return []
            
        @retry(wait=wait_fixed(1))
        def translate_single_cell(task: CellTranslationTask) -> CellTranslationTask:
            """翻译单个单元格的线程安全函数"""
            try:
                with self._lock:
                    logger.debug(f"[并发表格翻译] 单元格 {task.cell_idx} 输入: '{task.text}'")
                
                translated_text = self.translator.translate(task.text)
                task.cell.translated_text = translated_text
                
                with self._lock:
                    logger.debug(f"[并发表格翻译] 单元格 {task.cell_idx} 输出: '{translated_text}'")
                    table_stats["translated"] += 1
                
                return task
            except Exception as e:
                logger.warning(f"单元格 {task.cell_idx} 翻译失败: '{task.text}' -> {e}")
                task.cell.translated_text = task.text
                return task
        
        # 执行并发翻译
        max_workers = self.thread_count if self.thread_count > 0 else min(len(tasks), 4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            translated_tasks = list(executor.map(translate_single_cell, tasks))
            
        logger.info(f"并发翻译完成，处理了 {len(translated_tasks)} 个单元格")
        return translated_tasks
        
    def _optimize_layout(self, tasks: List[CellTranslationTask], grid_layout: dict, table_region: 'TableRegion') -> List['TableCell']:
        """
        布局优化：检测空间冲突并调整字体大小
        
        Args:
            tasks: 已翻译的任务列表
            grid_layout: 网格布局信息
            table_region: 表格区域对象
            
        Returns:
            优化后的单元格列表
        """
        # 创建任务字典以便快速查找
        task_by_cell_idx = {task.cell_idx: task for task in tasks}
        
        # 检测和解决空间冲突
        optimized_cells = []
        for cell in table_region.cells:
            if cell in [task.cell for task in tasks]:
                # 找到对应的翻译任务
                task = next((t for t in tasks if t.cell == cell), None)
                if task and hasattr(cell, 'translated_text'):
                    # 进行空间冲突检测和字体调整
                    optimized_cell = self._resolve_spatial_conflicts(task, grid_layout)
                    optimized_cells.append(optimized_cell)
                else:
                    optimized_cells.append(cell)
            else:
                # 未翻译的单元格直接添加
                optimized_cells.append(cell)
                
        return optimized_cells
        
    def _resolve_spatial_conflicts(self, task: CellTranslationTask, grid_layout: dict) -> 'TableCell':
        """
        解决空间冲突：调整字体大小以适应单元格边界
        
        Args:
            task: 翻译任务
            grid_layout: 网格布局信息
            
        Returns:
            调整后的单元格
        """
        cell = task.cell
        translated_text = cell.translated_text
        
        if not translated_text or not translated_text.strip():
            return cell
            
        # 计算单元格可用空间
        available_width = cell.x1 - cell.x0
        available_height = cell.y1 - cell.y0
        
        # 选择合适的字体
        font_name = self._select_optimal_font(translated_text)
        
        # 迭代调整字体大小以适应空间
        optimal_font_size = self._find_optimal_font_size(
            translated_text, 
            font_name, 
            available_width, 
            available_height,
            cell.font_size
        )
        
        # 更新单元格字体信息
        cell.font_size = optimal_font_size
        cell.optimized_font = font_name
        
        logger.debug(f"单元格 {task.cell_idx} 空间优化: 字体 {font_name}, 大小 {optimal_font_size:.1f}")
        return cell
        
    def _select_optimal_font(self, text: str) -> str:
        """选择最适合的字体"""
        # 简化的字体选择逻辑
        try:
            if self.fontmap["tiro"].to_unichr(ord(text[0])) == text[0]:
                return "tiro"
        except Exception:
            pass
        return self.noto_name
        
    def _find_optimal_font_size(self, text: str, font_name: str, max_width: float, max_height: float, original_size: float) -> float:
        """
        寻找最优字体大小
        
        Args:
            text: 文本内容
            font_name: 字体名称
            max_width: 最大宽度
            max_height: 最大高度  
            original_size: 原始字体大小
            
        Returns:
            优化后的字体大小
        """
        # 从原始大小开始尝试
        font_size = min(original_size, self.layout_config.max_font_size)
        min_size = self.layout_config.min_font_size
        
        while font_size >= min_size:
            # 计算文本在当前字体大小下的预期宽度
            predicted_width = self._predict_text_width(text, font_name, font_size)
            predicted_height = font_size * 1.2  # 简化的行高计算
            
            # 检查是否适合单元格
            if (predicted_width <= max_width - self.layout_config.overlap_tolerance and 
                predicted_height <= max_height - self.layout_config.overlap_tolerance):
                break
                
            font_size -= self.layout_config.font_size_step
            
        return max(font_size, min_size)
        
    def _predict_text_width(self, text: str, font_name: str, font_size: float) -> float:
        """
        预测文本宽度
        
        Args:
            text: 文本内容
            font_name: 字体名称
            font_size: 字体大小
            
        Returns:
            预测的文本宽度
        """
        total_width = 0.0
        
        for char in text:
            try:
                if font_name == self.noto_name:
                    char_width = self.noto.char_lengths(char, font_size)[0]
                else:
                    char_width = self.fontmap[font_name].char_width(ord(char)) * font_size
                total_width += char_width
            except Exception:
                # 使用平均字符宽度作为后备
                total_width += font_size * 0.6
                
        return total_width

    def _process_non_translation_cells(self, cells: List['TableCell']) -> None:
        """
        处理不需要翻译的单元格，确保它们也有translated_text属性
        
        Args:
            cells: 单元格列表
        """
        for cell in cells:
            cell_text = cell.text.strip()
            if not hasattr(cell, 'translated_text'):
                if not cell_text:
                    # 空单元格
                    cell.translated_text = ""
                elif not re.search(r'[\u4e00-\u9fff]|[a-zA-Z]', cell_text):
                    # 不包含中英文的单元格（如数字、符号等）
                    cell.translated_text = cell_text
                # 需要翻译的单元格将在并发翻译阶段处理

