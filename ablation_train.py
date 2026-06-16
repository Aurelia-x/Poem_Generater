"""
消融实验 - 训练脚本

独立训练各消融模型，保存 checkpoint 和训练清单。
训练完成后运行 ablation_evaluate.py 做横评。
"""
import os

import torch

from config import (
    ABLATION_DIR,
    ABLATION_EXPERIMENTS,
    ABLATION_MANIFEST_PATH,
    BATCH_SIZE,
    RANDOM_SEED,
)
from data_utils import build_dataloaders
from model import build_model
from trainer import fit
from utils import configure_torch_runtime, count_parameters, ensure_dir, get_device, save_json, set_seed


def main():
    print("=" * 50)
    print(" 消融实验 - 训练")
    print("=" * 50)

    ensure_dir(ABLATION_DIR)
    configure_torch_runtime()
    set_seed(RANDOM_SEED)

    device = get_device()
    if device == "cuda":
        print(f"[设备] CUDA | GPU: {torch.cuda.get_device_name(0)} | PyTorch: {torch.__version__}")
    else:
        print(f"[设备] CPU | PyTorch: {torch.__version__}")

    train_loader, val_loader, test_loader, stoi, itos = build_dataloaders()

    manifest = []

    for exp_idx, exp_config in enumerate(ABLATION_EXPERIMENTS, start=1):
        print(f"\n{'=' * 50}")
        print(f" [{exp_idx}/{len(ABLATION_EXPERIMENTS)}] {exp_config['experiment_name']}")
        print(f"{'=' * 50}")

        merged_config = dict(exp_config)
        merged_config["batch_size"] = BATCH_SIZE
        merged_config["random_seed"] = RANDOM_SEED
        merged_config["vocab_size"] = len(stoi)

        print(f"[配置] layers={merged_config['num_layers']} embed={merged_config['embed_dim']} "
              f"hidden={merged_config['hidden_dim']} dropout={merged_config['dropout']} "
              f"{'GRU' if merged_config['model_type'] == 'gru' else 'LSTM'}")

        model = build_model(
            vocab_size=len(stoi),
            model_type=merged_config["model_type"],
            embed_dim=merged_config["embed_dim"],
            hidden_dim=merged_config["hidden_dim"],
            num_layers=merged_config["num_layers"],
            dropout=merged_config["dropout"],
        ).to(device)

        total_params, _ = count_parameters(model)
        print(f"[模型] params={total_params:,}")

        history, best_epoch, best_val_ppl, _ = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            stoi=stoi,
            itos=itos,
            config_dict=merged_config,
        )

        manifest.append({
            "experiment_name": exp_config["experiment_name"],
            "best_checkpoint": exp_config["best_model_save_path"],
            "model_type": merged_config["model_type"],
            "embed_dim": merged_config["embed_dim"],
            "hidden_dim": merged_config["hidden_dim"],
            "num_layers": merged_config["num_layers"],
            "dropout": merged_config["dropout"],
            "weight_decay": merged_config["weight_decay"],
            "learning_rate": merged_config["learning_rate"],
            "num_epochs": merged_config["num_epochs"],
            "parameter_count": total_params,
            "best_epoch": best_epoch,
            "best_val_ppl": best_val_ppl,
        })

        print(f"[完成] best_epoch={best_epoch} best_val_ppl={best_val_ppl:.4f}")

    # 保存清单
    ensure_dir(ABLATION_DIR)
    save_json(manifest, ABLATION_MANIFEST_PATH)
    print(f"\n[清单] {ABLATION_MANIFEST_PATH}")

    print(f"\n{'=' * 50}")
    print(f" 训练完成 ({len(manifest)} 个模型)")
    print(f" 下一步: python ablation_evaluate.py")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
