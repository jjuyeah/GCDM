import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.utils.sparse import dense_to_sparse
from torch_geometric.utils import is_undirected

import numpy as np
import hydra
import logging
import os

from pooler import Pooler, DeepPooler
from metric import get_metric
from dataloader import load_dataset, load_adj_from_numpy, get_dataset_info


@hydra.main(version_base=None, config_path=".", config_name="config_main")
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

    optimizer = Adam(pooler.parameters(), lr=cfg.learning_rate)

    best_mae = float("inf")
    cnt = 0

    for epoch in range(cfg.epochs):
        train_loss = 0
        pooler.train()
        dataloader["train_loader"].shuffle()
        for x, _ in dataloader["train_loader"].get_iterator():
            x = torch.from_numpy(x).to(cfg.device)

            x_c, A_c = pooler.pooling(x, A)
            x_r, A_r = pooler.unpooling(x_c, A_c)

            loss = F.mse_loss(x_r, x) + F.mse_loss(A_r, A) + cfg.lmda * pooler.entropy()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() / dataloader["train_loader"].num_batch
        
        x_pred = []
        x_true = []
        valid_loss = 0
        pooler.eval()
        for x, _ in dataloader["val_loader"].get_iterator():
            x = torch.from_numpy(x).to(cfg.device)

            with torch.no_grad():
                x_c, A_c = pooler.pooling(x, A)
                x_r, A_r = pooler.unpooling(x_c, A_c)
            
            loss = F.mse_loss(x_r, x) + F.mse_loss(A_r, A) + cfg.lmda * pooler.entropy()

            valid_loss += loss.item() / dataloader["val_loader"].num_batch

            x_pred.append(x_r.squeeze().cpu())
            x_true.append(x.squeeze().cpu())
        
        x_pred = scaler.inverse_transform(torch.cat(x_pred, dim=0))
        x_true = scaler.inverse_transform(torch.cat(x_true, dim=0))

        mae, rmse, mape = get_metric(x_true, x_pred)
        
        logging.info(f"Epoch {epoch} | Train Loss {train_loss:.4f} | Valid Loss {valid_loss:.4f} | Valid MAE {mae:.4f} | Valid RMSE {rmse:.4f} | Valid MAPE {mape:.4f}")

        if mae < best_mae:
            best_mae = mae
            cnt = 0
        else:
            cnt += 1
            if cnt == cfg.patience:
                torch.save(pooler.state_dict(), f"{path}/checkpoints/params.pth")
                logging.info("Early Stopped")
                break
        
    x_pred = []
    x_true = []
    pooler.eval()
    for x, _ in dataloader["test_loader"].get_iterator():
        x = torch.from_numpy(x).to(cfg.device)

        with torch.no_grad():
            x_c, A_c = pooler.pooling(x, A)
            x_r, A_r = pooler.unpooling(x_c, A_c)
        
        x_pred.append(x_r.squeeze().cpu())
        x_true.append(x.squeeze().cpu())
    
    x_pred = scaler.inverse_transform(torch.cat(x_pred, dim=0))
    x_true = scaler.inverse_transform(torch.cat(x_true, dim=0))

    mae, rmse, mape = get_metric(x_true, x_pred)

    logging.info(f"MAE {mae:.4f} | RMSE {rmse:.4f} | MAPE {mape:.4f}")


if __name__ == "__main__":
    main()
