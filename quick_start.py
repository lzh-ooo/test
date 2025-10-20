"""
快速启动脚本 - 用于测试优化后的训练和推理流程
这个脚本展示了如何使用优化后的代码
"""

import os
import torch
from transformers import AutoTokenizer


def demo_data_processing():
    """演示数据处理"""
    print("\n" + "=" * 80)
    print("📊 数据处理演示")
    print("=" * 80)
    
    from optimized_data_processing import OptimizedDataProcessor
    
    # 配置
    TOKENIZER_PATH = '/root/autodl-tmp/MateGenConv/model/mateconv_tokenizer'
    INPUT_FILE = './dataset/mobvoi_seq_monkey_general_open_corpus.jsonl'
    CLEANED_FILE = './dataset/cleaned_data.jsonl'
    OUTPUT_FILE = './dataset/pretrain_data_optimized.bin'
    STATS_FILE = './dataset/data_stats.json'
    
    print("\n创建数据处理器...")
    processor = OptimizedDataProcessor(
        tokenizer_path=TOKENIZER_PATH,
        max_length=512,
        chunk_size=50000,
        save_interval=1000000
    )
    
    # Step 1: 清洗数据
    if not os.path.exists(CLEANED_FILE):
        print("\n🧹 步骤 1: 清洗数据...")
        valid, invalid = processor.validate_and_clean_jsonl(INPUT_FILE, CLEANED_FILE)
        print(f"✅ 有效: {valid:,}, 无效: {invalid:,}")
    else:
        print(f"\n✅ 清洗后的文件已存在: {CLEANED_FILE}")
    
    # Step 2: 处理数据
    print("\n🚀 步骤 2: 处理数据集...")
    processor.process_dataset(
        input_path=CLEANED_FILE,
        output_path=OUTPUT_FILE,
        resume=True
    )
    
    # Step 3: 保存统计
    print("\n📊 步骤 3: 保存统计信息...")
    processor.save_statistics(STATS_FILE)
    
    print("\n✅ 数据处理完成！")


def demo_training():
    """演示训练流程"""
    print("\n" + "=" * 80)
    print("🎯 训练演示")
    print("=" * 80)
    
    from optimized_pretrain import Trainer, TrainingConfig
    
    # 创建训练配置
    config = TrainingConfig(
        # 数据
        data_path="./dataset/pretrain_data.bin",
        val_data_path=None,
        
        # 训练参数
        epochs=3,  # 演示用，只训练3个epoch
        batch_size=16,  # 较小的batch size
        learning_rate=3e-4,
        min_lr=3e-5,
        
        # 优化器
        optimizer_type="adamw",
        lr_scheduler="cosine",
        warmup_ratio=0.1,
        
        # 训练技巧
        use_ema=True,
        grad_clip=1.0,
        accumulation_steps=2,
        dtype="bfloat16",
        
        # 验证和保存
        eval_interval=500,
        save_interval=1000,
        log_interval=10,
        
        # 输出
        out_dir="out_demo",
        
        # Wandb
        use_wandb=False
    )
    
    print("\n配置:")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Optimizer: {config.optimizer_type}")
    print(f"  Use EMA: {config.use_ema}")
    
    # TODO: 加载实际的模型和数据
    # from model.model import Transformer
    # from model.LMConfig import LMConfig
    # from model.dataset import PretrainDataset
    
    print("\n⚠️  注意: 需要实际的模型代码才能运行训练")
    print("   请确保 model/ 目录中包含:")
    print("   - model.py (Transformer 模型)")
    print("   - LMConfig.py (模型配置)")
    print("   - dataset.py (数据集类)")


def demo_inference():
    """演示推理流程"""
    print("\n" + "=" * 80)
    print("💬 推理演示")
    print("=" * 80)
    
    from optimized_inference import TextGenerator, GenerationConfig
    
    # 配置
    TOKENIZER_PATH = '/root/autodl-tmp/MateGenConv/model/mateconv_tokenizer'
    MODEL_PATH = 'out/pretrain_512.pth'
    
    if not os.path.exists(MODEL_PATH):
        print(f"\n⚠️  模型文件不存在: {MODEL_PATH}")
        print("   请先训练模型或提供预训练模型路径")
        return
    
    print("\n📥 加载分词器...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, use_fast=False)
        print(f"✅ 分词器加载成功 (词表大小: {len(tokenizer)})")
    except Exception as e:
        print(f"❌ 分词器加载失败: {e}")
        return
    
    # TODO: 加载模型
    print("\n⚠️  注意: 需要实际的模型代码才能运行推理")
    print("   请确保 model/ 目录中包含:")
    print("   - model.py (Transformer 模型)")
    print("   - LMConfig.py (模型配置)")
    
    # 演示配置
    gen_config = GenerationConfig(
        do_sample=True,
        temperature=0.8,
        top_k=50,
        top_p=0.9,
        repetition_penalty=1.2,
        max_new_tokens=100
    )
    
    print("\n生成配置:")
    print(f"  Temperature: {gen_config.temperature}")
    print(f"  Top-K: {gen_config.top_k}")
    print(f"  Top-P: {gen_config.top_p}")
    print(f"  Repetition Penalty: {gen_config.repetition_penalty}")
    
    print("\n📝 示例提示:")
    prompts = [
        "中国",
        "长江、黄河",
        "你好，好久不见"
    ]
    for i, prompt in enumerate(prompts, 1):
        print(f"  {i}. {prompt}")


def show_command_examples():
    """显示命令行示例"""
    print("\n" + "=" * 80)
    print("📝 命令行使用示例")
    print("=" * 80)
    
    print("\n【数据处理】")
    print("python optimized_data_processing.py")
    
    print("\n【训练 - 使用配置文件】")
    print("python train.py --config config.yaml")
    
    print("\n【训练 - 使用命令行参数】")
    print("python train.py \\")
    print("    --epochs 15 \\")
    print("    --batch_size 32 \\")
    print("    --learning_rate 3e-4 \\")
    print("    --use_ema \\")
    print("    --early_stopping")
    
    print("\n【训练 - 分布式训练】")
    print("torchrun --nproc_per_node=2 train.py --config config.yaml --ddp")
    
    print("\n【推理 - 单次生成】")
    print("python inference.py \\")
    print("    --model_path out/pretrain_512.pth \\")
    print("    --prompt \"中国\" \\")
    print("    --temperature 0.8")
    
    print("\n【推理 - 交互式对话】")
    print("python inference.py \\")
    print("    --model_path out/pretrain_512.pth \\")
    print("    --interactive")
    
    print("\n【推理 - 批量生成】")
    print("python inference.py \\")
    print("    --model_path out/pretrain_512.pth \\")
    print("    --batch_file prompts.txt")


def show_optimization_summary():
    """显示优化总结"""
    print("\n" + "=" * 80)
    print("🎉 优化总结")
    print("=" * 80)
    
    print("\n【数据处理优化】")
    print("  ✅ 合并清洗和预处理步骤")
    print("  ✅ 添加断点续传功能")
    print("  ✅ 详细的数据统计和可视化")
    print("  ✅ 更好的错误处理")
    
    print("\n【训练优化】")
    print("  ✅ 多种优化器 (AdamW, Adam)")
    print("  ✅ 灵活的学习率调度 (Cosine, Linear, Polynomial)")
    print("  ✅ 模型 EMA 提升生成质量")
    print("  ✅ 验证集和早停机制")
    print("  ✅ 完善的检查点管理")
    print("  ✅ Wandb 集成")
    print("  ✅ 分布式训练支持")
    
    print("\n【推理优化】")
    print("  ✅ 多种采样策略 (Greedy, Top-K, Top-P)")
    print("  ✅ KV Cache 加速生成")
    print("  ✅ 批量生成支持")
    print("  ✅ 流式输出")
    print("  ✅ 交互式对话")
    print("  ✅ 重复惩罚和 ngram 阻止")
    
    print("\n【代码质量】")
    print("  ✅ 清晰的代码结构")
    print("  ✅ 详细的注释")
    print("  ✅ 配置文件支持")
    print("  ✅ 完善的错误处理")


def main():
    """主菜单"""
    print("\n" + "=" * 80)
    print("🚀 MateConv 预训练优化版 - 快速启动")
    print("=" * 80)
    
    while True:
        print("\n请选择要演示的功能:")
        print("  1. 数据处理演示")
        print("  2. 训练流程演示")
        print("  3. 推理流程演示")
        print("  4. 命令行示例")
        print("  5. 优化总结")
        print("  6. 全部演示")
        print("  0. 退出")
        
        choice = input("\n请输入选项 (0-6): ").strip()
        
        if choice == "1":
            # demo_data_processing()  # 实际运行会很慢
            print("\n⚠️  数据处理需要较长时间，这里只展示代码")
            print("    实际使用时请取消注释运行")
            
        elif choice == "2":
            demo_training()
            
        elif choice == "3":
            demo_inference()
            
        elif choice == "4":
            show_command_examples()
            
        elif choice == "5":
            show_optimization_summary()
            
        elif choice == "6":
            # demo_data_processing()
            demo_training()
            demo_inference()
            show_command_examples()
            show_optimization_summary()
            
        elif choice == "0":
            print("\n👋 再见！")
            break
            
        else:
            print("\n❌ 无效选项，请重新输入")


if __name__ == "__main__":
    main()

