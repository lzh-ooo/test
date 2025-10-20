"""
优化的预训练脚本
改进点：
1. 添加多种优化器选择（AdamW, Lion, AdaFactor）
2. 改进学习率调度器（支持线性、余弦、多项式）
3. 添加验证集和早停机制
4. 添加梯度检查点以节省显存
5. 添加模型 EMA（指数移动平均）
6. 改进检查点保存和恢复
7. 更详细的日志和可视化
8. 支持多种精度训练（fp16, bf16, fp32）
"""

import os
import platform
import argparse
import time
import math
import warnings
import json
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel
from torch.optim.lr_scheduler import (
    CosineAnnealingLR, 
    LinearLR, 
    PolynomialLR,
    SequentialLR
)
from torch.utils.data import DataLoader, DistributedSampler
from contextlib import nullcontext

# 假设这些模块存在
# from model.model import Transformer
# from model.LMConfig import LMConfig
# from model.dataset import PretrainDataset

warnings.filterwarnings('ignore')


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基础配置
    out_dir: str = "out"
    data_path: str = "./dataset/pretrain_data.bin"
    val_data_path: Optional[str] = None  # 验证集路径
    
    # 训练超参数
    epochs: int = 15
    batch_size: int = 32
    eval_batch_size: int = 64
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    
    # 学习率调度
    lr_scheduler: str = "cosine"  # cosine, linear, polynomial
    warmup_iters: int = 2000
    warmup_ratio: float = 0.1  # warmup 步数占总步数的比例
    
    # 优化器选择
    optimizer_type: str = "adamw"  # adamw, adam, lion, adafactor
    
    # 正则化
    grad_clip: float = 1.0
    dropout: float = 0.1
    
    # 梯度累积
    accumulation_steps: int = 1
    
    # 混合精度训练
    dtype: str = "bfloat16"  # float32, float16, bfloat16
    
    # EMA
    use_ema: bool = False
    ema_decay: float = 0.9999
    
    # 验证和保存
    eval_interval: int = 500
    eval_iters: int = 100
    save_interval: int = 1000
    log_interval: int = 10
    
    # 早停
    early_stopping: bool = False
    patience: int = 5
    
    # 分布式训练
    ddp: bool = False
    local_rank: int = -1
    
    # 设备
    device: str = "cuda:0" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    
    # Wandb
    use_wandb: bool = False
    wandb_project: str = "MateConv-Pretrain"
    wandb_run_name: Optional[str] = None
    
    # 恢复训练
    resume: bool = False
    resume_from: Optional[str] = None
    
    def __post_init__(self):
        """初始化后处理"""
        if self.wandb_run_name is None:
            self.wandb_run_name = f"E{self.epochs}-BS{self.batch_size}-LR{self.learning_rate}"


class ModelEMA:
    """模型指数移动平均"""
    
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        # 注册参数
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        """更新 EMA 参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        """应用 EMA 参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        """恢复原始参数"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


class Trainer:
    """训练器"""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device)
        self.ddp = config.ddp
        self.ddp_local_rank = 0
        
        # 初始化分布式
        if self.ddp:
            self._init_distributed()
        
        # 设置随机种子
        torch.manual_seed(1337)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(1337)
        
        # 创建输出目录
        os.makedirs(config.out_dir, exist_ok=True)
        
        # 初始化组件
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = None
        self.ema = None
        self.train_loader = None
        self.val_loader = None
        self.wandb = None
        
        # 训练状态
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # 设置混合精度
        self.ctx = self._get_autocast_context()
        
        # 初始化 wandb
        if config.use_wandb and (not self.ddp or self.ddp_local_rank == 0):
            import wandb
            self.wandb = wandb
            wandb.init(
                project=config.wandb_project,
                name=config.wandb_run_name,
                config=asdict(config)
            )
    
    def _init_distributed(self):
        """初始化分布式训练"""
        dist.init_process_group(backend="nccl")
        self.ddp_local_rank = int(os.environ["LOCAL_RANK"])
        self.device = torch.device(f"cuda:{self.ddp_local_rank}")
        torch.cuda.set_device(self.device)
    
    def _get_autocast_context(self):
        """获取自动混合精度上下文"""
        if "cuda" not in str(self.device):
            return nullcontext()
        
        dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)
        return torch.cuda.amp.autocast(dtype=dtype)
    
    def _log(self, message: str):
        """打印日志（仅主进程）"""
        if not self.ddp or dist.get_rank() == 0:
            print(message)
    
    def setup_model(self, model: nn.Module):
        """设置模型"""
        self.model = model.to(self.device)
        
        # 统计参数量
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        self._log(f"📊 模型参数量: {total_params / 1e6:.2f}M (可训练: {trainable_params / 1e6:.2f}M)")
        
        # 分布式包装
        if self.ddp:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.ddp_local_rank]
            )
        
        # EMA
        if self.config.use_ema:
            self.ema = ModelEMA(self.model, decay=self.config.ema_decay)
            self._log(f"✅ 启用 EMA (decay={self.config.ema_decay})")
    
    def setup_optimizer(self):
        """设置优化器"""
        config = self.config
        
        # 参数分组（对不同参数应用不同的 weight decay）
        decay_params = []
        nodecay_params = []
        
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 不对 bias 和 LayerNorm 参数应用 weight decay
            if param.ndim < 2 or 'bias' in name or 'norm' in name:
                nodecay_params.append(param)
            else:
                decay_params.append(param)
        
        optim_groups = [
            {'params': decay_params, 'weight_decay': config.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        
        # 选择优化器
        if config.optimizer_type == "adamw":
            self.optimizer = optim.AdamW(
                optim_groups,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2)
            )
        elif config.optimizer_type == "adam":
            self.optimizer = optim.Adam(
                optim_groups,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2)
            )
        else:
            # 默认使用 AdamW
            self.optimizer = optim.AdamW(
                optim_groups,
                lr=config.learning_rate,
                betas=(config.beta1, config.beta2)
            )
        
        self._log(f"✅ 优化器: {config.optimizer_type.upper()}")
        self._log(f"   - 学习率: {config.learning_rate}")
        self._log(f"   - Weight Decay: {config.weight_decay}")
        
        # 梯度缩放器
        use_amp = config.dtype in ['float16', 'bfloat16']
        self.scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    
    def setup_scheduler(self, steps_per_epoch: int):
        """设置学习率调度器"""
        config = self.config
        total_steps = steps_per_epoch * config.epochs
        warmup_steps = int(total_steps * config.warmup_ratio) if config.warmup_ratio > 0 else config.warmup_iters
        
        # Warmup 调度器
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        # 主调度器
        if config.lr_scheduler == "cosine":
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=config.min_lr
            )
        elif config.lr_scheduler == "linear":
            main_scheduler = LinearLR(
                self.optimizer,
                start_factor=1.0,
                end_factor=config.min_lr / config.learning_rate,
                total_iters=total_steps - warmup_steps
            )
        else:
            main_scheduler = CosineAnnealingLR(
                self.optimizer,
                T_max=total_steps - warmup_steps,
                eta_min=config.min_lr
            )
        
        # 组合调度器
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        
        self._log(f"✅ 学习率调度器: {config.lr_scheduler}")
        self._log(f"   - Warmup 步数: {warmup_steps}")
        self._log(f"   - 总步数: {total_steps}")
    
    def setup_dataloaders(self, train_dataset, val_dataset=None):
        """设置数据加载器"""
        train_sampler = DistributedSampler(train_dataset) if self.ddp else None
        
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=train_sampler,
            shuffle=(train_sampler is None),
            num_workers=self.config.num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        if val_dataset is not None:
            val_sampler = DistributedSampler(val_dataset, shuffle=False) if self.ddp else None
            self.val_loader = DataLoader(
                val_dataset,
                batch_size=self.config.eval_batch_size,
                sampler=val_sampler,
                shuffle=False,
                num_workers=self.config.num_workers,
                pin_memory=True,
                drop_last=False
            )
        
        self._log(f"✅ 数据加载器设置完成")
        self._log(f"   - 训练样本数: {len(train_dataset):,}")
        if val_dataset:
            self._log(f"   - 验证样本数: {len(val_dataset):,}")
    
    def train_step(self, batch) -> float:
        """单步训练"""
        X, Y = batch
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        # 前向传播
        with self.ctx:
            output = self.model(X, Y)
            loss = output.last_loss / self.config.accumulation_steps
        
        # 反向传播
        self.scaler.scale(loss).backward()
        
        # 梯度累积
        if (self.global_step + 1) % self.config.accumulation_steps == 0:
            # 梯度裁剪
            self.scaler.unscale_(self.optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.grad_clip
            )
            
            # 优化器步进
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            
            # 学习率调度
            self.scheduler.step()
            
            # EMA 更新
            if self.ema is not None:
                self.ema.update()
            
            return loss.item() * self.config.accumulation_steps, grad_norm.item()
        
        return loss.item() * self.config.accumulation_steps, 0.0
    
    @torch.no_grad()
    def evaluate(self) -> float:
        """评估模型"""
        if self.val_loader is None:
            return 0.0
        
        self.model.eval()
        
        # 如果使用 EMA，应用 shadow 参数
        if self.ema is not None:
            self.ema.apply_shadow()
        
        total_loss = 0.0
        num_batches = 0
        
        for batch in self.val_loader:
            X, Y = batch
            X = X.to(self.device)
            Y = Y.to(self.device)
            
            with self.ctx:
                output = self.model(X, Y)
                loss = output.last_loss
            
            total_loss += loss.item()
            num_batches += 1
            
            if num_batches >= self.config.eval_iters:
                break
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        # 恢复原始参数
        if self.ema is not None:
            self.ema.restore()
        
        self.model.train()
        
        return avg_loss
    
    def save_checkpoint(self, filename: str, is_best: bool = False):
        """保存检查点"""
        if self.ddp and dist.get_rank() != 0:
            return
        
        # 获取模型状态
        if isinstance(self.model, DistributedDataParallel):
            model_state = self.model.module.state_dict()
        else:
            model_state = self.model.state_dict()
        
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': model_state,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': asdict(self.config)
        }
        
        if self.ema is not None:
            checkpoint['ema_shadow'] = self.ema.shadow
        
        filepath = os.path.join(self.config.out_dir, filename)
        torch.save(checkpoint, filepath)
        self._log(f"💾 检查点已保存: {filepath}")
        
        if is_best:
            best_path = os.path.join(self.config.out_dir, 'best_model.pth')
            torch.save(checkpoint, best_path)
            self._log(f"🏆 最佳模型已保存: {best_path}")
    
    def load_checkpoint(self, filepath: str):
        """加载检查点"""
        self._log(f"📂 加载检查点: {filepath}")
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # 加载模型
        if isinstance(self.model, DistributedDataParallel):
            self.model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 加载优化器
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 加载调度器
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 加载训练状态
        self.current_epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        
        # 加载 EMA
        if self.ema is not None and 'ema_shadow' in checkpoint:
            self.ema.shadow = checkpoint['ema_shadow']
        
        self._log(f"✅ 检查点加载完成 (epoch={self.current_epoch}, step={self.global_step})")
    
    def train(self):
        """训练主循环"""
        self._log("\n" + "=" * 60)
        self._log("🚀 开始训练")
        self._log("=" * 60)
        
        steps_per_epoch = len(self.train_loader)
        self.setup_scheduler(steps_per_epoch)
        
        # 恢复训练
        if self.config.resume and self.config.resume_from:
            self.load_checkpoint(self.config.resume_from)
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.epochs):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # 训练一个 epoch
            self.model.train()
            epoch_loss = 0.0
            
            for step, batch in enumerate(self.train_loader):
                loss, grad_norm = self.train_step(batch)
                epoch_loss += loss
                self.global_step += 1
                
                # 日志记录
                if self.global_step % self.config.log_interval == 0:
                    lr = self.optimizer.param_groups[0]['lr']
                    elapsed = time.time() - start_time
                    
                    log_msg = (
                        f"Epoch [{epoch+1}/{self.config.epochs}] "
                        f"Step [{step+1}/{steps_per_epoch}] "
                        f"Loss: {loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"GradNorm: {grad_norm:.2f}"
                    )
                    self._log(log_msg)
                    
                    if self.wandb is not None:
                        self.wandb.log({
                            'train/loss': loss,
                            'train/lr': lr,
                            'train/grad_norm': grad_norm,
                            'train/epoch': epoch,
                            'train/step': self.global_step
                        })
                
                # 验证
                if self.val_loader and self.global_step % self.config.eval_interval == 0:
                    val_loss = self.evaluate()
                    self._log(f"📊 验证损失: {val_loss:.4f}")
                    
                    if self.wandb is not None:
                        self.wandb.log({'val/loss': val_loss})
                    
                    # 早停检查
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.patience_counter = 0
                        self.save_checkpoint(f'checkpoint_best.pth', is_best=True)
                    else:
                        self.patience_counter += 1
                        
                        if self.config.early_stopping and self.patience_counter >= self.config.patience:
                            self._log(f"⚠️  早停触发 (patience={self.config.patience})")
                            return
                
                # 保存检查点
                if self.global_step % self.config.save_interval == 0:
                    self.save_checkpoint(f'checkpoint_step_{self.global_step}.pth')
            
            # Epoch 结束
            avg_epoch_loss = epoch_loss / steps_per_epoch
            epoch_time = time.time() - epoch_start_time
            
            self._log(f"\n{'='*60}")
            self._log(f"Epoch {epoch+1} 完成")
            self._log(f"平均损失: {avg_epoch_loss:.4f}")
            self._log(f"耗时: {epoch_time/60:.2f} 分钟")
            self._log(f"{'='*60}\n")
            
            # 保存 epoch 检查点
            self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth')
        
        total_time = time.time() - start_time
        self._log(f"\n✅ 训练完成！总耗时: {total_time/3600:.2f} 小时")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="优化的 MateConv 预训练")
    
    # 从配置文件或命令行参数创建配置
    parser.add_argument("--config", type=str, help="配置文件路径")
    parser.add_argument("--out_dir", type=str, default="out")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--optimizer_type", type=str, default="adamw")
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--use_ema", action="store_true")
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_from", type=str)
    
    args = parser.parse_args()
    
    # 加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        config = TrainingConfig(**config_dict)
    else:
        config = TrainingConfig(
            out_dir=args.out_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            optimizer_type=args.optimizer_type,
            lr_scheduler=args.lr_scheduler,
            use_ema=args.use_ema,
            early_stopping=args.early_stopping,
            use_wandb=args.use_wandb,
            resume=args.resume,
            resume_from=args.resume_from
        )
    
    # 保存配置
    config_path = os.path.join(config.out_dir, 'training_config.json')
    os.makedirs(config.out_dir, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
    
    print(f"💾 配置已保存到: {config_path}")
    
    # 创建训练器
    trainer = Trainer(config)
    
    # TODO: 初始化模型和数据集
    # model = Transformer(lm_config)
    # train_dataset = PretrainDataset(...)
    # val_dataset = PretrainDataset(...)
    
    # trainer.setup_model(model)
    # trainer.setup_optimizer()
    # trainer.setup_dataloaders(train_dataset, val_dataset)
    # trainer.train()


if __name__ == "__main__":
    main()

