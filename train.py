import argparse
import os
import time
import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from NextMove import MyModel
from dataloader import MyDataset
from tools import get_config, run_test, train_epoch, get_mapper, update_config, custom_collate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--dataset', type=str, default='TC')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--dim', type=int, default=16, help='must be a multiple of 4')
    parser.add_argument('--topic', type=int, default=0, help='LDA topic num')
    parser.add_argument('--at', type=str, default='none', help='arrival time module type')
    parser.add_argument('--encoder', type=str, default='trans', help='encoder type')
    parser.add_argument('--batch', type=int, default=256)
    parser.add_argument('--epoch', type=int, default=50)
    return parser.parse_args()


def setup_device(gpu_list):
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_list
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_config_info(config, device):
    print(f"Dataset: {config.Dataset.name} | Device: {device} | Model: {config.Encoder.encoder_type}")
    print(f"AT type: {config.Model.at_type} | topic_num: {config.Dataset.topic_num} | dim: {config.Embedding.base_dim}")


def main():
    args = parse_args()
    device = setup_device(args.gpu)
    dataset_path = f'./data/{args.dataset}'
    save_dir = f"./saved_models/{args.dataset}"
    config_path = os.path.join(save_dir, "settings.yml")
    os.makedirs(save_dir, exist_ok=True)

    torch.manual_seed(args.seed)
    get_mapper(dataset_path)

    update_config(config_path, ['Dataset', 'topic_num'], args.topic)
    update_config(config_path, ['Model', 'seed'], args.seed)
    update_config(config_path, ['Model', 'at_type'], args.at)
    update_config(config_path, ['Embedding', 'base_dim'], args.dim)
    update_config(config_path, ['Encoder', 'encoder_type'], args.encoder)
    update_config(config_path, ['Model', 'batch_size'], args.batch)
    update_config(config_path, ['Model', 'epoch'], args.epoch)

    config = get_config(config_path, easy=True)
    config.Dataset.name = args.dataset  

    dataset = MyDataset(config=config, dataset_path=dataset_path, device=device, load_mode='train')
    dataloader = DataLoader(
        dataset,
        batch_size=config.Model.batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=True,
        collate_fn=lambda batch: custom_collate(batch, device, config)
    )

    model = MyModel(config).to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.Adam_optimizer.initial_lr,
        weight_decay=config.Adam_optimizer.weight_decay
    )

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(dataloader),
        num_training_steps=len(dataloader) * config.Model.epoch
    )

    log_config_info(config, device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total training samples: {len(dataloader) * config.Model.batch_size} | Trainable parameters: {total_params}")

    if args.test:
        run_test(dataset_path, model_path=save_dir, model=model, device=device, epoch=config.Model.epoch - 1, test_only=True)
        return

    best_val_loss = float("inf")
    start_time = time.time()

    report_path = os.path.join(save_dir, "report.txt")
    with open(report_path, "w") as report_file:
        print(f'Train batches: {len(dataloader)}')
        for epoch in range(config.Model.epoch):
            epoch_start = time.time()
            avg_loss = train_epoch(model, dataloader, optimizer, loss_fn, scheduler)

            log_line = (
                f"Epoch [{epoch + 1}/{config.Model.epoch}] "
                f"{time.strftime('%Y-%m-%d %H:%M:%S')} | "
            )
            if avg_loss < best_val_loss:
                log_line += f"Best Loss: {best_val_loss:.6f} → {avg_loss:.6f} | Time: {time.time() - epoch_start:.2f}s"
                best_val_loss = avg_loss
            else:
                log_line += f"Best Loss: {best_val_loss:.6f} | Epoch Loss: {avg_loss:.6f} | Time: {time.time() - epoch_start:.2f}s"

            print(log_line)
            report_file.write(log_line + '\n\n')
            report_file.flush()

            if (epoch + 1) % config.Model.test_epoch == 0:
                run_test(dataset_path, model_path=save_dir, model=model, device=device, epoch=epoch)

    total_time = time.time() - start_time
    with open(report_path, "a") as report_file:
        report_file.write(f"Total Training Time: {total_time:.2f} seconds\n")

    print(f"\nTraining complete.\n")


if __name__ == '__main__':
    main()
