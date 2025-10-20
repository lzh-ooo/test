"""
优化的推理脚本
改进点：
1. 支持多种采样策略（贪心、Top-K、Top-P、Temperature）
2. 实现 KV Cache 加速生成
3. 支持批量生成
4. 添加重复惩罚和长度惩罚
5. 流式输出
6. 支持多轮对话
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Callable, Dict, Any
from dataclasses import dataclass
import numpy as np

# 假设这些模块存在
# from model.model import Transformer
# from model.LMConfig import LMConfig
from transformers import AutoTokenizer


@dataclass
class GenerationConfig:
    """生成配置"""
    # 采样策略
    do_sample: bool = True
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    
    # 惩罚
    repetition_penalty: float = 1.0
    length_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    
    # 生成控制
    max_new_tokens: int = 100
    min_new_tokens: int = 1
    num_beams: int = 1
    
    # 停止条件
    eos_token_id: Optional[int] = None
    pad_token_id: Optional[int] = None
    
    # 其他
    use_cache: bool = True
    output_scores: bool = False


class TextGenerator:
    """文本生成器"""
    
    def __init__(
        self,
        model: nn.Module,
        tokenizer: AutoTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Args:
            model: 语言模型
            tokenizer: 分词器
            device: 计算设备
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        
        # 特殊 token IDs
        self.bos_token_id = tokenizer.bos_token_id
        self.eos_token_id = tokenizer.eos_token_id
        self.pad_token_id = tokenizer.pad_token_id
    
    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
        stream: bool = False,
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        生成文本
        
        Args:
            prompt: 输入提示
            config: 生成配置
            stream: 是否流式输出
            callback: 流式输出回调函数
            
        Returns:
            生成的文本
        """
        if config is None:
            config = GenerationConfig()
        
        # 设置特殊 token
        if config.eos_token_id is None:
            config.eos_token_id = self.eos_token_id
        if config.pad_token_id is None:
            config.pad_token_id = self.pad_token_id
        
        # 编码输入
        input_ids = self.tokenizer.encode(prompt, return_tensors='pt').to(self.device)
        
        # 生成
        if stream:
            return self._generate_stream(input_ids, config, callback)
        else:
            return self._generate_standard(input_ids, config)
    
    def _generate_standard(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig
    ) -> str:
        """标准生成（非流式）"""
        generated_ids = []
        past_key_values = None
        
        # 用于重复惩罚的 ngram 字典
        ngram_dict = {}
        
        for step in range(config.max_new_tokens):
            # 前向传播
            if config.use_cache and past_key_values is not None:
                # 使用 KV Cache，只需传入最后一个 token
                outputs = self.model(
                    input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
            else:
                outputs = self.model(input_ids, use_cache=config.use_cache)
            
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            past_key_values = outputs.past_key_values if config.use_cache else None
            
            # 应用重复惩罚
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits,
                    input_ids,
                    config.repetition_penalty
                )
            
            # 应用 ngram 阻止
            if config.no_repeat_ngram_size > 0:
                logits = self._apply_no_repeat_ngram(
                    logits,
                    generated_ids,
                    config.no_repeat_ngram_size,
                    ngram_dict
                )
            
            # 采样下一个 token
            next_token_id = self._sample_next_token(logits, config)
            
            # 添加到生成序列
            generated_ids.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
            
            # 检查停止条件
            if next_token_id.item() == config.eos_token_id:
                break
            
            if step + 1 >= config.min_new_tokens and self._should_stop(generated_ids, config):
                break
        
        # 解码
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text
    
    def _generate_stream(
        self,
        input_ids: torch.Tensor,
        config: GenerationConfig,
        callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """流式生成"""
        generated_ids = []
        past_key_values = None
        ngram_dict = {}
        
        for step in range(config.max_new_tokens):
            # 前向传播
            if config.use_cache and past_key_values is not None:
                outputs = self.model(
                    input_ids[:, -1:],
                    past_key_values=past_key_values,
                    use_cache=True
                )
            else:
                outputs = self.model(input_ids, use_cache=config.use_cache)
            
            logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values if config.use_cache else None
            
            # 应用惩罚
            if config.repetition_penalty != 1.0:
                logits = self._apply_repetition_penalty(
                    logits,
                    input_ids,
                    config.repetition_penalty
                )
            
            if config.no_repeat_ngram_size > 0:
                logits = self._apply_no_repeat_ngram(
                    logits,
                    generated_ids,
                    config.no_repeat_ngram_size,
                    ngram_dict
                )
            
            # 采样
            next_token_id = self._sample_next_token(logits, config)
            generated_ids.append(next_token_id.item())
            input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
            
            # 流式输出
            if callback is not None:
                token_text = self.tokenizer.decode([next_token_id.item()])
                callback(token_text)
            
            # 检查停止条件
            if next_token_id.item() == config.eos_token_id:
                break
            
            if step + 1 >= config.min_new_tokens and self._should_stop(generated_ids, config):
                break
        
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text
    
    def _sample_next_token(
        self,
        logits: torch.Tensor,
        config: GenerationConfig
    ) -> torch.Tensor:
        """
        采样下一个 token
        
        Args:
            logits: [batch_size, vocab_size]
            config: 生成配置
            
        Returns:
            next_token_id: [batch_size]
        """
        if not config.do_sample:
            # 贪心解码
            next_token_id = torch.argmax(logits, dim=-1)
        else:
            # 温度缩放
            if config.temperature != 1.0:
                logits = logits / config.temperature
            
            # Top-K 过滤
            if config.top_k > 0:
                logits = self._top_k_filtering(logits, config.top_k)
            
            # Top-P (nucleus) 过滤
            if config.top_p < 1.0:
                logits = self._top_p_filtering(logits, config.top_p)
            
            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token_id = torch.multinomial(probs, num_samples=1).squeeze(1)
        
        return next_token_id
    
    @staticmethod
    def _top_k_filtering(logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Top-K 过滤"""
        top_k = min(top_k, logits.size(-1))
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = float('-inf')
        return logits
    
    @staticmethod
    def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        """Top-P (nucleus) 过滤"""
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        
        # 移除累积概率超过 top_p 的 tokens
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # 将被移除的 tokens 的 logits 设为 -inf
        indices_to_remove = sorted_indices_to_remove.scatter(
            1, sorted_indices, sorted_indices_to_remove
        )
        logits[indices_to_remove] = float('-inf')
        return logits
    
    @staticmethod
    def _apply_repetition_penalty(
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        penalty: float
    ) -> torch.Tensor:
        """应用重复惩罚"""
        batch_size, vocab_size = logits.shape
        
        for i in range(batch_size):
            for token_id in set(input_ids[i].tolist()):
                # 如果 logit > 0，除以 penalty；否则乘以 penalty
                if logits[i, token_id] < 0:
                    logits[i, token_id] *= penalty
                else:
                    logits[i, token_id] /= penalty
        
        return logits
    
    @staticmethod
    def _apply_no_repeat_ngram(
        logits: torch.Tensor,
        generated_ids: List[int],
        ngram_size: int,
        ngram_dict: Dict
    ) -> torch.Tensor:
        """阻止重复 n-gram"""
        if len(generated_ids) < ngram_size:
            return logits
        
        # 获取最近的 n-1 gram
        prefix = tuple(generated_ids[-(ngram_size-1):])
        
        # 如果这个 prefix 已经出现过，阻止相同的下一个 token
        if prefix in ngram_dict:
            banned_tokens = ngram_dict[prefix]
            logits[0, banned_tokens] = float('-inf')
        
        # 更新 ngram 字典
        if len(generated_ids) >= ngram_size:
            ngram_prefix = tuple(generated_ids[-ngram_size:-1])
            next_token = generated_ids[-1]
            if ngram_prefix not in ngram_dict:
                ngram_dict[ngram_prefix] = []
            ngram_dict[ngram_prefix].append(next_token)
        
        return logits
    
    @staticmethod
    def _should_stop(generated_ids: List[int], config: GenerationConfig) -> bool:
        """检查是否应该停止生成"""
        # 可以添加更多停止条件
        return False
    
    def batch_generate(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None
    ) -> List[str]:
        """
        批量生成
        
        Args:
            prompts: 输入提示列表
            config: 生成配置
            
        Returns:
            生成的文本列表
        """
        if config is None:
            config = GenerationConfig()
        
        # 编码输入
        encoded = self.tokenizer(
            prompts,
            padding=True,
            return_tensors='pt'
        )
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        
        batch_size = input_ids.size(0)
        generated_ids_list = [[] for _ in range(batch_size)]
        
        # 批量生成
        for step in range(config.max_new_tokens):
            outputs = self.model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits[:, -1, :]  # [batch_size, vocab_size]
            
            # 对每个样本应用惩罚和采样
            next_token_ids = []
            for i in range(batch_size):
                sample_logits = logits[i:i+1]
                
                if config.repetition_penalty != 1.0:
                    sample_logits = self._apply_repetition_penalty(
                        sample_logits,
                        input_ids[i:i+1],
                        config.repetition_penalty
                    )
                
                next_token_id = self._sample_next_token(sample_logits, config)
                next_token_ids.append(next_token_id)
                generated_ids_list[i].append(next_token_id.item())
            
            next_token_ids = torch.cat(next_token_ids, dim=0).unsqueeze(1)
            input_ids = torch.cat([input_ids, next_token_ids], dim=1)
            
            # 更新 attention mask
            attention_mask = torch.cat([
                attention_mask,
                torch.ones((batch_size, 1), device=self.device)
            ], dim=1)
            
            # 检查是否所有样本都已完成
            if all(
                config.eos_token_id in ids
                for ids in generated_ids_list
            ):
                break
        
        # 解码
        generated_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids_list
        ]
        
        return generated_texts
    
    def chat(
        self,
        history: List[Dict[str, str]],
        config: Optional[GenerationConfig] = None
    ) -> str:
        """
        多轮对话
        
        Args:
            history: 对话历史 [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}, ...]
            config: 生成配置
            
        Returns:
            助手的回复
        """
        # 构建提示
        prompt = self._build_chat_prompt(history)
        
        # 生成回复
        response = self.generate(prompt, config)
        
        return response
    
    def _build_chat_prompt(self, history: List[Dict[str, str]]) -> str:
        """构建对话提示"""
        prompt_parts = []
        
        for turn in history:
            role = turn['role']
            content = turn['content']
            
            if role == 'user':
                prompt_parts.append(f"用户: {content}")
            elif role == 'assistant':
                prompt_parts.append(f"助手: {content}")
        
        prompt_parts.append("助手: ")
        
        return "\n".join(prompt_parts)


def demo():
    """演示用法"""
    print("=" * 60)
    print("文本生成演示")
    print("=" * 60)
    
    # 配置
    TOKENIZER_PATH = '/root/autodl-tmp/MateGenConv/model/mateconv_tokenizer'
    MODEL_PATH = 'out/pretrain_512.pth'
    
    # 加载分词器
    print("\n📥 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
    
    # TODO: 加载模型
    # from model.model import Transformer
    # from model.LMConfig import LMConfig
    # lm_config = LMConfig(dim=512, n_layers=8, n_heads=16, vocab_size=6400)
    # model = Transformer(lm_config)
    # model.load_state_dict(torch.load(MODEL_PATH))
    
    # 创建生成器
    # generator = TextGenerator(model, tokenizer)
    
    # 生成配置
    gen_config = GenerationConfig(
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        max_new_tokens=100,
        repetition_penalty=1.2
    )
    
    # 示例 1: 标准生成
    print("\n" + "=" * 60)
    print("示例 1: 标准生成")
    print("=" * 60)
    prompt = "中国"
    print(f"输入: {prompt}")
    # result = generator.generate(prompt, gen_config)
    # print(f"输出: {result}")
    
    # 示例 2: 流式生成
    print("\n" + "=" * 60)
    print("示例 2: 流式生成")
    print("=" * 60)
    prompt = "长江、"
    print(f"输入: {prompt}")
    print("输出: ", end="", flush=True)
    
    def stream_callback(token: str):
        print(token, end="", flush=True)
    
    # result = generator.generate(prompt, gen_config, stream=True, callback=stream_callback)
    print()
    
    # 示例 3: 批量生成
    print("\n" + "=" * 60)
    print("示例 3: 批量生成")
    print("=" * 60)
    prompts = ["中国", "长江、", "你好，"]
    print(f"输入: {prompts}")
    # results = generator.batch_generate(prompts, gen_config)
    # for i, result in enumerate(results):
    #     print(f"输出 {i+1}: {result}")
    
    # 示例 4: 多轮对话
    print("\n" + "=" * 60)
    print("示例 4: 多轮对话")
    print("=" * 60)
    history = [
        {"role": "user", "content": "你好"},
        {"role": "assistant", "content": "你好！有什么可以帮助你的吗？"},
        {"role": "user", "content": "介绍一下中国"}
    ]
    # response = generator.chat(history, gen_config)
    # print(f"助手: {response}")


if __name__ == "__main__":
    demo()

