"""
简化的推理启动脚本
用法:
    # 单次生成
    python inference.py --prompt "中国" --model_path out/pretrain_512.pth
    
    # 交互式对话
    python inference.py --model_path out/pretrain_512.pth --interactive
    
    # 批量生成
    python inference.py --model_path out/pretrain_512.pth --batch_file prompts.txt
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# 导入自定义模块
try:
    from model.model import Transformer
    from model.LMConfig import LMConfig
except ImportError:
    print("⚠️  警告: 无法导入模型模块")
    print("   这是一个示例脚本，需要您的实际模型代码")
    
    class LMConfig:
        def __init__(self):
            self.dim = 512
            self.n_layers = 8
            self.n_heads = 16
            self.vocab_size = 6400
            self.max_seq_len = 512
            self.dropout = 0.1
            self.use_moe = False
    
    class Transformer:
        pass

from transformers import AutoTokenizer
from utils import load_config, get_device
from optimized_inference import TextGenerator, GenerationConfig


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MateConv 文本生成")
    
    # 模型相关
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="模型权重路径"
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default="/root/autodl-tmp/MateGenConv/model/mateconv_tokenizer",
        help="分词器路径"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径"
    )
    
    # 生成参数
    parser.add_argument("--prompt", type=str, default=None, help="输入提示")
    parser.add_argument("--max_new_tokens", type=int, default=100, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.8, help="温度参数")
    parser.add_argument("--top_k", type=int, default=50, help="Top-K 采样")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-P 采样")
    parser.add_argument("--repetition_penalty", type=float, default=1.2, help="重复惩罚")
    parser.add_argument("--no_repeat_ngram_size", type=int, default=3, help="禁止重复的 n-gram 大小")
    parser.add_argument("--do_sample", action="store_true", default=True, help="是否采样")
    parser.add_argument("--greedy", action="store_true", help="使用贪心解码（覆盖 do_sample）")
    
    # 模式选择
    parser.add_argument("--interactive", action="store_true", help="交互式对话模式")
    parser.add_argument("--batch_file", type=str, default=None, help="批量生成文件")
    parser.add_argument("--stream", action="store_true", help="流式输出")
    
    # 设备
    parser.add_argument("--device", type=str, default="auto", help="计算设备")
    
    return parser.parse_args()


def load_model(model_path: str, tokenizer_path: str, device: torch.device):
    """
    加载模型和分词器
    
    Args:
        model_path: 模型路径
        tokenizer_path: 分词器路径
        device: 计算设备
        
    Returns:
        (model, tokenizer) 元组
    """
    print("📥 加载分词器...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False)
    print(f"✅ 分词器加载成功 (词表大小: {len(tokenizer)})")
    
    print("\n🏗️  初始化模型...")
    lm_config = LMConfig()
    
    try:
        model = Transformer(lm_config)
        print("✅ 模型初始化成功")
        
        print(f"\n📂 加载模型权重: {model_path}")
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        print("✅ 模型权重加载成功")
        
        model.to(device)
        model.eval()
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"📊 模型参数量: {total_params / 1e6:.2f}M")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise


def single_generation(generator: TextGenerator, args):
    """单次生成"""
    if not args.prompt:
        print("❌ 请提供输入提示 (--prompt)")
        return
    
    # 创建生成配置
    gen_config = GenerationConfig(
        do_sample=(not args.greedy),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_new_tokens=args.max_new_tokens
    )
    
    print("\n" + "=" * 80)
    print("💬 文本生成")
    print("=" * 80)
    print(f"输入: {args.prompt}")
    print("-" * 80)
    
    if args.stream:
        print("输出: ", end="", flush=True)
        
        def callback(token: str):
            print(token, end="", flush=True)
        
        result = generator.generate(
            args.prompt,
            gen_config,
            stream=True,
            callback=callback
        )
        print("\n" + "=" * 80)
    else:
        result = generator.generate(args.prompt, gen_config)
        print(f"输出: {result}")
        print("=" * 80)


def interactive_mode(generator: TextGenerator, args):
    """交互式对话模式"""
    print("\n" + "=" * 80)
    print("🤖 交互式对话模式")
    print("=" * 80)
    print("输入 'exit'、'quit' 或 'q' 退出")
    print("输入 'clear' 清空对话历史")
    print("=" * 80 + "\n")
    
    # 创建生成配置
    gen_config = GenerationConfig(
        do_sample=(not args.greedy),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_new_tokens=args.max_new_tokens
    )
    
    history = []
    
    while True:
        try:
            # 获取用户输入
            user_input = input("👤 用户: ").strip()
            
            if not user_input:
                continue
            
            # 检查退出命令
            if user_input.lower() in ['exit', 'quit', 'q']:
                print("\n👋 再见！")
                break
            
            # 检查清空命令
            if user_input.lower() == 'clear':
                history = []
                print("✅ 对话历史已清空\n")
                continue
            
            # 添加到历史
            history.append({"role": "user", "content": user_input})
            
            # 生成回复
            if args.stream:
                print("🤖 助手: ", end="", flush=True)
                
                def callback(token: str):
                    print(token, end="", flush=True)
                
                response = generator.chat(history, gen_config)
                print()
            else:
                response = generator.chat(history, gen_config)
                print(f"🤖 助手: {response}")
            
            # 添加回复到历史
            history.append({"role": "assistant", "content": response})
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 再见！")
            break
        except Exception as e:
            print(f"\n❌ 生成出错: {e}")
            continue


def batch_generation(generator: TextGenerator, args):
    """批量生成"""
    if not args.batch_file or not os.path.exists(args.batch_file):
        print(f"❌ 批量文件不存在: {args.batch_file}")
        return
    
    print(f"\n📄 读取批量文件: {args.batch_file}")
    
    # 读取提示
    with open(args.batch_file, 'r', encoding='utf-8') as f:
        prompts = [line.strip() for line in f if line.strip()]
    
    print(f"✅ 共 {len(prompts)} 个提示")
    
    # 创建生成配置
    gen_config = GenerationConfig(
        do_sample=(not args.greedy),
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        max_new_tokens=args.max_new_tokens
    )
    
    # 批量生成
    print("\n🚀 开始批量生成...\n")
    print("=" * 80)
    
    results = generator.batch_generate(prompts, gen_config)
    
    for i, (prompt, result) in enumerate(zip(prompts, results), 1):
        print(f"\n[{i}/{len(prompts)}]")
        print(f"输入: {prompt}")
        print(f"输出: {result}")
        print("-" * 80)
    
    # 保存结果
    output_file = args.batch_file.replace('.txt', '_output.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        for prompt, result in zip(prompts, results):
            f.write(f"输入: {prompt}\n")
            f.write(f"输出: {result}\n")
            f.write("-" * 80 + "\n")
    
    print(f"\n💾 结果已保存到: {output_file}")
    print("=" * 80)


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🚀 MateConv 文本生成脚本")
    print("=" * 80 + "\n")
    
    # 解析参数
    args = parse_args()
    
    # 获取设备
    device = get_device(args.device)
    print(f"💻 使用设备: {device}")
    
    # 加载模型
    try:
        model, tokenizer = load_model(args.model_path, args.tokenizer_path, device)
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        print("   请确保模型路径正确且模型文件存在")
        return
    
    # 创建生成器
    print("\n⚙️  创建文本生成器...")
    generator = TextGenerator(model, tokenizer, device=str(device))
    print("✅ 生成器创建成功")
    
    # 根据模式执行
    try:
        if args.interactive:
            # 交互式模式
            interactive_mode(generator, args)
            
        elif args.batch_file:
            # 批量生成模式
            batch_generation(generator, args)
            
        else:
            # 单次生成模式
            single_generation(generator, args)
            
    except Exception as e:
        print(f"\n❌ 执行出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

