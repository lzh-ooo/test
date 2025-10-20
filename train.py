"""
简化的训练启动脚本
用法:
    python train.py --config config.yaml
    或
    python train.py --epochs 15 --batch_size 32 --learning_rate 3e-4
"""

import os
import sys
import argparse
import torch
from pathlib import Path

# 导入自定义模块（需要确保这些模块存在）
try:
    from model.model import Transformer
    from model.LMConfig import LMConfig
    from model.dataset import PretrainDataset
except ImportError:
    print("⚠️  警告: 无法导入模型模块，请确保 model/ 目录存在")
    print("   这是一个示例脚本，需要您的实际模型代码")
    # 创建模拟类以便演示
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
    
    class PretrainDataset:
        pass

from utils import load_config, set_seed, get_device, print_model_structure
from optimized_pretrain import Trainer, TrainingConfig
from transformers import AutoTokenizer


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="MateConv 预训练")
    
    # 配置文件
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径"
    )
    
    # 数据相关
    parser.add_argument("--data_path", type=str, default="./dataset/pretrain_data.bin")
    parser.add_argument("--val_data_path", type=str, default=None)
    parser.add_argument("--tokenizer_path", type=str, 
                       default="/root/autodl-tmp/MateGenConv/model/mateconv_tokenizer")
    
    # 训练参数
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--min_lr", type=float, default=3e-5)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    
    # 优化器和调度器
    parser.add_argument("--optimizer_type", type=str, default="adamw",
                       choices=["adamw", "adam"])
    parser.add_argument("--lr_scheduler", type=str, default="cosine",
                       choices=["cosine", "linear", "polynomial"])
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    
    # 训练技巧
    parser.add_argument("--use_ema", action="store_true", help="使用 EMA")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                       choices=["float32", "float16", "bfloat16"])
    
    # 验证和保存
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--patience", type=int, default=5)
    
    # 输出和恢复
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--resume", action="store_true", help="恢复训练")
    parser.add_argument("--resume_from", type=str, default=None)
    
    # 分布式
    parser.add_argument("--ddp", action="store_true", help="使用分布式训练")
    parser.add_argument("--local_rank", type=int, default=-1)
    
    # Wandb
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--wandb_project", type=str, default="MateConv-Pretrain")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    
    # 其他
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--num_workers", type=int, default=4)
    
    return parser.parse_args()


def main():
    """主函数"""
    print("\n" + "=" * 80)
    print("🚀 MateConv 预训练脚本")
    print("=" * 80 + "\n")
    
    # 解析参数
    args = parse_args()
    
    # 加载配置（如果提供了配置文件）
    if args.config and os.path.exists(args.config):
        print(f"📄 加载配置文件: {args.config}")
        config_dict = load_config(args.config)
        
        # 从配置文件创建训练配置
        training_cfg = config_dict.get('training', {})
        data_cfg = config_dict.get('data', {})
        
        config = TrainingConfig(
            # 数据
            data_path=data_cfg.get('train_data_path', args.data_path),
            val_data_path=data_cfg.get('val_data_path', args.val_data_path),
            
            # 训练参数
            epochs=training_cfg.get('epochs', args.epochs),
            batch_size=training_cfg.get('batch_size', args.batch_size),
            learning_rate=training_cfg.get('learning_rate', args.learning_rate),
            min_lr=training_cfg.get('min_lr', args.min_lr),
            weight_decay=training_cfg.get('weight_decay', args.weight_decay),
            
            # 优化器
            optimizer_type=training_cfg.get('optimizer_type', args.optimizer_type),
            lr_scheduler=training_cfg.get('lr_scheduler', args.lr_scheduler),
            warmup_ratio=training_cfg.get('warmup_ratio', args.warmup_ratio),
            
            # 训练技巧
            use_ema=training_cfg.get('use_ema', args.use_ema),
            grad_clip=training_cfg.get('grad_clip', args.grad_clip),
            accumulation_steps=training_cfg.get('accumulation_steps', args.accumulation_steps),
            dtype=training_cfg.get('dtype', args.dtype),
            
            # 验证和保存
            eval_interval=training_cfg.get('eval_interval', args.eval_interval),
            save_interval=training_cfg.get('save_interval', args.save_interval),
            log_interval=training_cfg.get('log_interval', args.log_interval),
            early_stopping=training_cfg.get('early_stopping', args.early_stopping),
            patience=training_cfg.get('patience', args.patience),
            
            # 输出
            out_dir=training_cfg.get('out_dir', args.out_dir),
            
            # Wandb
            use_wandb=config_dict.get('wandb', {}).get('enable', args.use_wandb),
            wandb_project=config_dict.get('wandb', {}).get('project', args.wandb_project),
            wandb_run_name=config_dict.get('wandb', {}).get('run_name', args.wandb_run_name),
            
            # 恢复
            resume=args.resume,
            resume_from=args.resume_from,
            
            # 分布式
            ddp=args.ddp,
            local_rank=args.local_rank,
            
            # 其他
            num_workers=args.num_workers
        )
    else:
        # 从命令行参数创建配置
        config = TrainingConfig(
            data_path=args.data_path,
            val_data_path=args.val_data_path,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            min_lr=args.min_lr,
            weight_decay=args.weight_decay,
            optimizer_type=args.optimizer_type,
            lr_scheduler=args.lr_scheduler,
            warmup_ratio=args.warmup_ratio,
            use_ema=args.use_ema,
            grad_clip=args.grad_clip,
            accumulation_steps=args.accumulation_steps,
            dtype=args.dtype,
            eval_interval=args.eval_interval,
            save_interval=args.save_interval,
            log_interval=args.log_interval,
            early_stopping=args.early_stopping,
            patience=args.patience,
            out_dir=args.out_dir,
            use_wandb=args.use_wandb,
            wandb_project=args.wandb_project,
            wandb_run_name=args.wandb_run_name,
            resume=args.resume,
            resume_from=args.resume_from,
            ddp=args.ddp,
            local_rank=args.local_rank,
            num_workers=args.num_workers
        )
    
    # 设置随机种子
    set_seed(args.seed)
    print(f"🎲 随机种子: {args.seed}")
    
    # 打印配置
    print("\n📋 训练配置:")
    print(f"  Epochs:          {config.epochs}")
    print(f"  Batch Size:      {config.batch_size}")
    print(f"  Learning Rate:   {config.learning_rate}")
    print(f"  Optimizer:       {config.optimizer_type}")
    print(f"  LR Scheduler:    {config.lr_scheduler}")
    print(f"  Dtype:           {config.dtype}")
    print(f"  Use EMA:         {config.use_ema}")
    print(f"  Early Stopping:  {config.early_stopping}")
    print(f"  Output Dir:      {config.out_dir}")
    
    # 创建输出目录
    os.makedirs(config.out_dir, exist_ok=True)
    
    # 初始化模型配置
    print("\n🏗️  初始化模型...")
    lm_config = LMConfig()
    
    try:
        # 初始化模型
        model = Transformer(lm_config)
        print("✅ 模型初始化成功")
        
        # 打印模型结构
        print_model_structure(model)
        
    except Exception as e:
        print(f"❌ 模型初始化失败: {e}")
        print("   这是一个示例脚本，需要您提供实际的模型代码")
        return
    
    # 加载数据集
    print("📦 加载数据集...")
    try:
        # 这里需要您的实际数据集加载代码
        train_dataset = PretrainDataset(
            [config.data_path],
            max_length=lm_config.max_seq_len,
            memmap=True
        )
        print(f"✅ 训练集加载成功: {len(train_dataset):,} 样本")
        
        val_dataset = None
        if config.val_data_path:
            val_dataset = PretrainDataset(
                [config.val_data_path],
                max_length=lm_config.max_seq_len,
                memmap=True
            )
            print(f"✅ 验证集加载成功: {len(val_dataset):,} 样本")
            
    except Exception as e:
        print(f"❌ 数据集加载失败: {e}")
        print("   请确保数据路径正确且数据集类已正确实现")
        return
    
    # 创建训练器
    print("\n⚙️  创建训练器...")
    trainer = Trainer(config)
    
    # 设置模型
    trainer.setup_model(model)
    
    # 设置优化器
    trainer.setup_optimizer()
    
    # 设置数据加载器
    trainer.setup_dataloaders(train_dataset, val_dataset)
    
    # 开始训练
    print("\n" + "=" * 80)
    print("🎯 开始训练")
    print("=" * 80 + "\n")
    
    try:
        trainer.train()
        print("\n✅ 训练完成！")
        
    except KeyboardInterrupt:
        print("\n⚠️  训练被用户中断")
        print(f"💾 保存检查点...")
        trainer.save_checkpoint("checkpoint_interrupted.pth")
        
    except Exception as e:
        print(f"\n❌ 训练出错: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

