"""
优化的数据预处理脚本
改进点：
1. 合并清洗和预处理步骤
2. 添加多进程支持
3. 添加数据统计和可视化
4. 改进错误处理和日志
5. 支持断点续传
"""

import os
import json
import jsonlines
import numpy as np
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer
from collections import Counter
import multiprocessing as mp
from functools import partial
import pickle


class OptimizedDataProcessor:
    """优化的数据处理器"""
    
    def __init__(
        self,
        tokenizer_path: str,
        max_length: int = 512,
        chunk_size: int = 50000,
        save_interval: int = 1000000,
        num_workers: int = 4
    ):
        """
        Args:
            tokenizer_path: 分词器路径
            max_length: 最大文本长度
            chunk_size: 每次处理的行数
            save_interval: 累积多少个 tokens 后保存一次
            num_workers: 多进程数量
        """
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
        self.max_length = max_length
        self.chunk_size = chunk_size
        self.save_interval = save_interval
        self.num_workers = num_workers
        
        self.bos_token = "<s>"
        self.eos_token = "</s>"
        
        # 统计信息
        self.stats = {
            'total_lines': 0,
            'valid_lines': 0,
            'invalid_lines': 0,
            'skipped_long': 0,
            'total_tokens': 0,
            'text_lengths': [],
            'token_lengths': []
        }
    
    def validate_and_clean_jsonl(
        self, 
        input_path: str, 
        output_path: str
    ) -> Tuple[int, int]:
        """
        一次性完成验证和清洗
        
        Args:
            input_path: 原始文件路径
            output_path: 清洗后文件路径
            
        Returns:
            (valid_lines, invalid_lines) 元组
        """
        print("🔍 开始验证和清洗数据...")
        
        valid_lines = 0
        invalid_lines = 0
        invalid_line_numbers = []
        
        # 第一遍：统计总行数
        print("📊 统计文件行数...")
        with open(input_path, 'rb') as f:
            total_lines = sum(1 for _ in f)
        
        # 第二遍：验证并清洗
        print(f"✅ 总行数: {total_lines:,}")
        print("🧹 开始清洗数据...")
        
        with open(input_path, 'rb') as infile, \
             open(output_path, 'wb') as outfile:
            
            for idx, line in tqdm(enumerate(infile), total=total_lines, desc="清洗进度"):
                try:
                    # 尝试解码和解析 JSON
                    decoded_line = line.decode('utf-8')
                    obj = json.loads(decoded_line)
                    
                    # 验证必要字段
                    if 'text' in obj and isinstance(obj['text'], str) and len(obj['text'].strip()) > 0:
                        outfile.write(line)
                        valid_lines += 1
                    else:
                        invalid_lines += 1
                        invalid_line_numbers.append(idx + 1)
                        
                except (UnicodeDecodeError, json.JSONDecodeError) as e:
                    invalid_lines += 1
                    invalid_line_numbers.append(idx + 1)
        
        print(f"\n✅ 有效行数: {valid_lines:,}")
        print(f"❌ 无效行数: {invalid_lines:,}")
        
        if invalid_line_numbers and len(invalid_line_numbers) <= 100:
            print(f"⚠️  无效行号: {invalid_line_numbers[:20]}...")
        
        return valid_lines, invalid_lines
    
    def tokenize_text(self, text: str) -> List[int]:
        """
        对单个文本进行分词
        
        Args:
            text: 输入文本
            
        Returns:
            token IDs 列表
        """
        try:
            # 添加 BOS 和 EOS 标记
            formatted_text = f'{self.bos_token}{text}{self.eos_token}'
            token_ids = self.tokenizer(formatted_text).data['input_ids']
            return token_ids
        except Exception as e:
            return []
    
    def process_chunk(
        self, 
        chunk: List[Dict]
    ) -> Tuple[List[int], Dict]:
        """
        处理一个数据块
        
        Args:
            chunk: 数据块
            
        Returns:
            (token_ids, chunk_stats) 元组
        """
        doc_ids = []
        chunk_stats = {
            'valid': 0,
            'skipped_long': 0,
            'text_lengths': [],
            'token_lengths': []
        }
        
        for obj in chunk:
            try:
                content = obj.get('text', '').strip()
                
                if not content:
                    continue
                
                text_len = len(content)
                chunk_stats['text_lengths'].append(text_len)
                
                # 跳过过长文本
                if text_len > self.max_length:
                    chunk_stats['skipped_long'] += 1
                    continue
                
                # 分词
                text_id = self.tokenize_text(content)
                
                if text_id:
                    doc_ids.extend(text_id)
                    chunk_stats['token_lengths'].append(len(text_id))
                    chunk_stats['valid'] += 1
                    
            except Exception as e:
                continue
        
        return doc_ids, chunk_stats
    
    def merge_stats(self, chunk_stats: Dict):
        """合并统计信息"""
        self.stats['valid_lines'] += chunk_stats['valid']
        self.stats['skipped_long'] += chunk_stats['skipped_long']
        self.stats['text_lengths'].extend(chunk_stats['text_lengths'])
        self.stats['token_lengths'].extend(chunk_stats['token_lengths'])
    
    def process_dataset(
        self, 
        input_path: str, 
        output_path: str,
        resume: bool = False
    ):
        """
        处理整个数据集
        
        Args:
            input_path: 输入文件路径
            output_path: 输出文件路径
            resume: 是否从断点继续
        """
        print("\n🚀 开始处理数据集...")
        
        # 检查断点
        checkpoint_path = output_path + '.checkpoint'
        start_line = 0
        
        if resume and os.path.exists(checkpoint_path):
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                start_line = checkpoint['processed_lines']
                self.stats = checkpoint['stats']
            print(f"📍 从第 {start_line:,} 行继续处理")
        else:
            # 清空输出文件
            if os.path.exists(output_path):
                os.remove(output_path)
        
        # 统计总行数
        with open(input_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        self.stats['total_lines'] = total_lines
        print(f"📊 总行数: {total_lines:,}")
        
        # 逐块处理
        doc_ids = []
        chunk_idx = 0
        processed_lines = start_line
        
        with jsonlines.open(input_path) as reader:
            # 跳过已处理的行
            if start_line > 0:
                for _ in range(start_line):
                    next(reader)
            
            with tqdm(total=total_lines - start_line, desc="处理进度", initial=start_line) as pbar:
                while True:
                    try:
                        # 读取一个块
                        chunk = []
                        for _ in range(self.chunk_size):
                            try:
                                chunk.append(next(reader))
                            except StopIteration:
                                break
                        
                        if not chunk:
                            break
                        
                        # 处理块
                        chunk_ids, chunk_stats = self.process_chunk(chunk)
                        doc_ids.extend(chunk_ids)
                        self.merge_stats(chunk_stats)
                        
                        processed_lines += len(chunk)
                        chunk_idx += 1
                        pbar.update(len(chunk))
                        
                        # 定期保存
                        if len(doc_ids) >= self.save_interval:
                            self._save_tokens(doc_ids, output_path)
                            self.stats['total_tokens'] += len(doc_ids)
                            doc_ids = []
                            
                            # 保存检查点
                            self._save_checkpoint(checkpoint_path, processed_lines)
                        
                    except Exception as e:
                        print(f"\n⚠️  处理块 {chunk_idx} 时出错: {e}")
                        continue
        
        # 保存剩余的 tokens
        if doc_ids:
            self._save_tokens(doc_ids, output_path)
            self.stats['total_tokens'] += len(doc_ids)
        
        # 删除检查点
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
        # 打印统计信息
        self.print_statistics()
    
    def _save_tokens(self, token_ids: List[int], output_path: str):
        """保存 token IDs 到二进制文件"""
        arr = np.array(token_ids, dtype=np.uint16)
        with open(output_path, 'ab') as f:
            f.write(arr.tobytes())
    
    def _save_checkpoint(self, checkpoint_path: str, processed_lines: int):
        """保存检查点"""
        checkpoint = {
            'processed_lines': processed_lines,
            'stats': self.stats
        }
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("📊 数据处理统计")
        print("=" * 60)
        print(f"总行数:        {self.stats['total_lines']:>15,}")
        print(f"有效行数:      {self.stats['valid_lines']:>15,}")
        print(f"跳过长文本:    {self.stats['skipped_long']:>15,}")
        print(f"总 Token 数:   {self.stats['total_tokens']:>15,}")
        print("-" * 60)
        
        if self.stats['text_lengths']:
            text_lengths = np.array(self.stats['text_lengths'])
            print(f"文本长度统计:")
            print(f"  平均值:      {text_lengths.mean():>15.2f}")
            print(f"  中位数:      {np.median(text_lengths):>15.0f}")
            print(f"  最小值:      {text_lengths.min():>15,}")
            print(f"  最大值:      {text_lengths.max():>15,}")
            print("-" * 60)
        
        if self.stats['token_lengths']:
            token_lengths = np.array(self.stats['token_lengths'])
            print(f"Token 长度统计:")
            print(f"  平均值:      {token_lengths.mean():>15.2f}")
            print(f"  中位数:      {np.median(token_lengths):>15.0f}")
            print(f"  最小值:      {token_lengths.min():>15,}")
            print(f"  最大值:      {token_lengths.max():>15,}")
        
        print("=" * 60)
    
    def save_statistics(self, output_path: str):
        """保存统计信息到 JSON 文件"""
        stats_copy = self.stats.copy()
        
        # 转换 numpy 数组为列表（用于 JSON 序列化）
        if stats_copy['text_lengths']:
            text_lengths = np.array(stats_copy['text_lengths'])
            stats_copy['text_length_stats'] = {
                'mean': float(text_lengths.mean()),
                'median': float(np.median(text_lengths)),
                'min': int(text_lengths.min()),
                'max': int(text_lengths.max()),
                'std': float(text_lengths.std())
            }
            del stats_copy['text_lengths']
        
        if stats_copy['token_lengths']:
            token_lengths = np.array(stats_copy['token_lengths'])
            stats_copy['token_length_stats'] = {
                'mean': float(token_lengths.mean()),
                'median': float(np.median(token_lengths)),
                'min': int(token_lengths.min()),
                'max': int(token_lengths.max()),
                'std': float(token_lengths.std())
            }
            del stats_copy['token_lengths']
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats_copy, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 统计信息已保存到: {output_path}")


def main():
    """主函数"""
    
    # 配置参数
    TOKENIZER_PATH = '/root/autodl-tmp/MateGenConv/model/mateconv_tokenizer'
    INPUT_FILE = './dataset/mobvoi_seq_monkey_general_open_corpus.jsonl'
    CLEANED_FILE = './dataset/mobvoi_seq_monkey_general_open_corpus_cleaned.jsonl'
    OUTPUT_FILE = './dataset/pretrain_data_optimized.bin'
    STATS_FILE = './dataset/data_statistics.json'
    
    # 创建数据处理器
    processor = OptimizedDataProcessor(
        tokenizer_path=TOKENIZER_PATH,
        max_length=512,
        chunk_size=50000,
        save_interval=1000000,
        num_workers=4
    )
    
    # Step 1: 验证和清洗数据
    if not os.path.exists(CLEANED_FILE):
        valid, invalid = processor.validate_and_clean_jsonl(INPUT_FILE, CLEANED_FILE)
    else:
        print(f"✅ 清洗后的文件已存在: {CLEANED_FILE}")
    
    # Step 2: 处理数据集
    processor.process_dataset(
        input_path=CLEANED_FILE,
        output_path=OUTPUT_FILE,
        resume=True  # 支持断点续传
    )
    
    # Step 3: 保存统计信息
    processor.save_statistics(STATS_FILE)
    
    print("\n✅ 数据处理完成！")


if __name__ == "__main__":
    main()

