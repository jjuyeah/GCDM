import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import OneCycleLR
from ema_pytorch import EMA

import numpy as np
# import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import hydra
import logging
import os

from DSTD.metric import Metric
from DSTD.model import DSTD
from pooler import Pooler
from utils import to_directed_tgt
from dataloader import load_dataset, load_adj_from_numpy, get_dataset_info


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg):
    path = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    os.makedirs(f"{path}/checkpoints", exist_ok=True)

    cfg.years = str(cfg.years)
    data_path, adj_path, cfg.num_nodes = get_dataset_info(cfg.dataset)

    adj_mx = load_adj_from_numpy(adj_path)
    adj_mx = adj_mx - np.eye(cfg.num_nodes)
    adj_mx = torch.from_numpy(adj_mx)
    A = adj_mx.float().to(cfg.device)

    dataloader, scaler = load_dataset(data_path, cfg)

    pooler = Pooler(cfg.num_nodes, cfg.num_cnodes).to(cfg.device)
    pooler.load_state_dict(torch.load(cfg.pooler_ckpt, map_location=cfg.device))
    pooler.eval()

    net = DSTD(
        input_dim=cfg.in_channels,
        num_nodes=cfg.num_cnodes,
        periods=cfg.periods,
        num_cond_blocks=cfg.num_cond_blocks,
        num_noise_blocks=cfg.num_noise_blocks,
        diff_steps=cfg.diff_steps,
        loss_type=cfg.loss_type,
        beta_end=cfg.beta_end,
        beta_schedule=cfg.beta_schedule,
        hidden_dim=cfg.hid_channels,
        eta=cfg.eta,
    ).to(cfg.device)

    ema = EMA(
        net,
        beta=cfg.ema_decay,
        update_after_step=cfg.ema_update_after_step,
        update_every=cfg.ema_update_every,
    ).to(cfg.device)


    optimizer = Adam(net.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    lr_scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg.max_learning_rate,
        steps_per_epoch=dataloader["train_loader"].num_batch,
        epochs=cfg.epochs,
    )

    metric = Metric(criteria=cfg.criteria, patience=cfg.patience * cfg.ema_update_every)

    train_losses = []

    for epoch in range(cfg.epochs):
        cumm_epoch_loss = 0.0
        cumm_epoch_valid_loss = 0.0

        net.train()
        dataloader["train_loader"].shuffle()
        for x_h, x_f in tqdm(dataloader["train_loader"].get_iterator(), desc=f"Epoch {epoch + 1}/{cfg.epochs}, Train"):
            x_h = torch.from_numpy(x_h).to(cfg.device)
            x_f = torch.from_numpy(x_f).to(cfg.device)

            with torch.no_grad():
                x_hc, A_c = pooler.pooling(x_h, A)
                x_fc = pooler.pooling(x_f, A)[0].squeeze()

                edge_index = to_directed_tgt(A_c, cfg.topk)

            loss = net(x=x_fc, prev=x_hc, edge_index=edge_index)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            ema.update()
            lr_scheduler.step()

            cumm_epoch_loss += loss.item()
        
        train_loss = cumm_epoch_loss / dataloader["train_loader"].num_batch
        train_losses.append(train_loss)

        logging.info(f"Epoch {epoch + 1:04d} \t Train Loss {train_loss:.4f}")

        if epoch > cfg.ema_update_after_step and epoch % cfg.ema_update_every == cfg.ema_update_every - 1:
            y_pred = []
            y_true = []
            ema.eval()
            for x_h, x_f in tqdm(dataloader["val_loader"].get_iterator(), desc=f"Epoch {epoch + 1}/{cfg.epochs}, Validation", colour="green"):
                x_h = torch.from_numpy(x_h).to(cfg.device)
                x_f = torch.from_numpy(x_f).to(cfg.device)
            
                with torch.no_grad():
                    x_hc, A_c = pooler.pooling(x_h, A)
                    x_fc = pooler.pooling(x_f, A)[0].squeeze()

                    edge_index = to_directed_tgt(A_c, cfg.topk)

                    loss = ema.ema_model(x=x_fc, prev=x_hc, edge_index=edge_index)
                    pred = ema.ema_model.sample(prev=x_hc, num_samples=1, edge_index=edge_index)
                    pred = pooler.unpooling(pred.unsqueeze(-1), A_c)[0].squeeze()
            
                cumm_epoch_valid_loss += loss.item()

                y_pred.append(pred.squeeze().cpu())
                y_true.append(x_f.squeeze().cpu())
        
            y_pred = scaler.inverse_transform(torch.cat(y_pred, dim=0))
            y_true = scaler.inverse_transform(torch.cat(y_true, dim=0))

            valid_loss = cumm_epoch_valid_loss / dataloader["val_loader"].num_batch
            
            metrics = metric(y_true, y_pred, loss=valid_loss, epoch=epoch, test=False)

            logging.info(f"Epoch {metrics['epoch'] + 1:04d} \t Valid Loss {metrics['loss']:.4f} \t Valid MAE {metrics['mae']:.4f}")

            if metric.check_early_stop(epoch):
                torch.save(pooler.state_dict(), f"{path}/checkpoints/pooler_params.pth")
                torch.save(ema.ema_model.state_dict(), f"{path}/checkpoints/params.pth")

                logging.info("Early Stopped")
                break


    metric.clear()

    truth = []
    preds = []

    ema.eval()
    for x_h, x_f in tqdm(dataloader["test_loader"].get_iterator(), desc="Test", colour="blue"):
        x_h = torch.from_numpy(x_h).to(cfg.device)
        x_f = torch.from_numpy(x_f)
        
        with torch.no_grad():
            x_hc, A_c = pooler.pooling(x_h, A)

            edge_index = to_directed_tgt(A_c, cfg.topk)

            pred = ema.ema_model.sample(prev=x_hc, num_samples=cfg.num_samples, edge_index=edge_index)
            pred = pooler.unpooling(pred.unsqueeze(-1), A_c)[0].squeeze()
        
        truth.append(x_f)
        preds.append(pred.cpu())
        
    preds = scaler.inverse_transform(preds.squeeze())
    truth = scaler.inverse_transform(truth.squeeze())
    median = torch.median(preds, dim=1)[0]

    metrics = metric(truth, median, preds)

    logging.info(f"Periods {cfg.periods}")

    logging.info(f"Masked MAE {metrics['mae']:.4f}")
    logging.info(f"Masked RMSE {metrics['rmse']:.4f}")
    logging.info(f"Masked MAPE {metrics['mape']:.4f}")
    logging.info(f"Masked CRPS {metrics['crps']:.4f}")


if __name__ == "__main__":
    main()
