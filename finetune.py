import os
from pathlib import Path
from typing import Optional

import torch
import tqdm
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm.prompting import PurePromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FinetuneConfig:
    vla_path: str = "/mnt/workspace/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

    # RLDS 数据集目录
    data_root_dir: Path = Path("/mnt/workspace/Libero_RLDS")
    dataset_name: str = "libero_spatial_no_noops"

    # 结果保存目录
    run_root_dir: Path = Path("runs")

    # Fine-tuning Parameters
    batch_size: int = 1
    learning_rate: float = 5e-4
    use_lora: bool = True                                           # Whether to use LoRA fine-tuning
    lora_rank: int = 32                                             # Rank of LoRA weight matrix
    lora_dropout: float = 0.05                                      # Dropout applied to LoRA weights
    use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
                                                                    #   => CAUTION: Reduces memory but hurts performance

    # 微调的最大时间步数（类似epoch）
    max_steps: int = 200_000

    # 保存检查点的间隔
    save_steps: int = 1000

    # 梯度累积
    """
    累积梯度： 当你设置 grad_accumulation_steps 大于 1 时，模型不会在每个 batch 之后立即更新参数，而是累计多个 batch 的梯度。只有累计了指定步数的梯度后，才会进行一次参数更新。
	模拟大批量训练： 如果显存有限，无法使用大 batch size，梯度累积可以在不增加显存占用的情况下，模拟大 batch size 的效果，从而稳定训练。
	实现方式： 例如，如果 grad_accumulation_steps=4，那么在每四个 batch 后调用一次 optimizer.step() 更新模型参数，并在每次更新后清空累积的梯度。
    这种方式可以在显存有限的情况下，通过梯度累积实现更稳定的训练效果。
    """
    grad_accumulation_steps: int = 4

    # 图像增强
    image_aug: bool = True                                          # Whether to train with image augmentations

    # 数据加载器用于打乱的缓冲区大小（如果内存不足可以减少）
    shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)

    # 加入运行 id 的额外注释
    run_id_note: Optional[str] = None


def finetune(cfg: FinetuneConfig) -> None:

    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")
    device = torch.device("cuda:0")

    # 构建本次实验的ID
    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )

    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    # Start =>> Build Directories
    run_dir = cfg.run_root_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)

    # 量化模型
    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )

    # 加载模型和 processor
    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16, # bf16
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # 将模型传送到GPU，BitsAndBytes量化时自动处理
    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device)

    # 构建 LoRA 模型
    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",    # 所有 linear 层都添加 LoRA
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()

    # 创建优化器
    trainable_params = [param for param in vla.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

    # Create Action Tokenizer
    # 将每个维度上的连续机器人动作，离散化为N个区间，并将其映射到最少使用的 token
    # processor.tokenizer 是预训练的分词器，用于将索引映射回词汇
    action_tokenizer = ActionTokenizer(processor.tokenizer)

    # 加载微调数据集
    # batch_transform 将 RLDS 格式数据集转换为模型输入形式
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
    )

    # 将本地的 RLDS 数据集文件，包装成一个 pytorch 的 Dataset 类
    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
    )

    # 创建 DataLoader
    # collator 对读取的每个 batch 数据做一些处理（填充、截断、mask 等）
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )

    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        collate_fn=collator,
    )

    # Train!
    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        vla.train()
        optimizer.zero_grad()

        # 梯度累积的轮数（反向传播次数）
        gradient_step_idx = 0

        for batch_idx, batch in enumerate(dataloader):

            # 自动混合精度
            with torch.autocast("cuda", dtype=torch.bfloat16):
                output: CausalLMOutputWithPast = vla(
                    input_ids=batch["input_ids"].to(device),
                    attention_mask=batch["attention_mask"].to(device),
                    pixel_values=batch["pixel_values"].to(torch.bfloat16).to(device),
                    labels=batch["labels"],
                )
                loss = output.loss

            # 梯度累积时，需要先积累梯度（并归一化），达到累积轮数后再更新参数
            normalized_loss = loss / cfg.grad_accumulation_steps
            normalized_loss.backward()

            # output.logits 是语言模型输出的 token 分布的分数（未经过softmax的）
            # 将图片特征对应的输出token排除
            action_logits = output.logits[:, vla.vision_backbone.featurizer.patch_embed.num_patches : -1]

            # 选择分数最高的 token 作为预测结果
            action_preds = action_logits.argmax(dim=2)

            # next token 预测任务，输出序列会右移一步，因此移除第一个 token，将目标序列与输出对齐
            action_gt = batch["labels"][:, 1:].to(action_preds.device)

            # action_token_begin_idx 是词汇表中第一个用作表示 action 的 token id，因此大于 action_token_begin_idx 的都是 action token
            # mask 表示哪些输出 token 属于 action，其他的 token 属于 language token，不用于计算准确率
            mask = action_gt > action_tokenizer.action_token_begin_idx

            # 计算 action 的准确率
            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()

            # 到达梯度累积轮次后，再更新参数
            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                progress.update()
                gradient_step_idx += 1

                # 按指定间隔保存检查点
                if gradient_step_idx % cfg.save_steps == 0:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")

                    # 创建检查点目录
                    checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                    os.makedirs(checkpoint_dir, exist_ok=True)

                    # 保存数据集统计，用于推理时的反量化操作
                    save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)

                    # 保存 lora 权重和 processor
                    processor.save_pretrained(checkpoint_dir)
                    vla.save_pretrained(checkpoint_dir)

                    print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")

            # 到达最大步数时，停止训练
            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    config = FinetuneConfig()
    finetune(config)
