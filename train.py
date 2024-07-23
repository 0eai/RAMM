import os

import numpy as np
import torch
import torch.optim as optim
from eval import evaluate
import copy

def train(model, train_loader, optimizer, loss_fn, use_gpu=False, device='cpu'):

    train_loss_list = []

    model.train()
    if use_gpu:
        model.to(device)

    for batch, batch_data in enumerate(train_loader, 1):
        features, feature_lens, labels, metas = batch_data
        
        if use_gpu:
            features = [feature.to(device) for feature in features]
            feature_lens = feature_lens.to(device)
            labels = labels.to(device)

        optimizer.zero_grad()

        preds,_ = model(features, feature_lens)

        loss = loss_fn(preds.squeeze(-1), labels.squeeze(-1))

        loss.backward()
        optimizer.step()

        train_loss_list.append(loss.item())

    train_loss = np.mean(train_loss_list)
    return train_loss


def save_model(model, model_folder, id):
    model_file_name = f'model_{id}.pth'
    model_file = os.path.join(model_folder, model_file_name)
    model_cpu = model.to('cpu')
    torch.save(model_cpu.state_dict(), model_file)
    return model_file


def train_model(task, model, data_loader, epochs, lr, model_path, identifier, use_gpu, loss_fn, eval_fn,
                eval_metric_str, early_stopping_patience, regularization=0.0, device='cpu'):
    train_loader, val_loader, test_loader = data_loader['train'], data_loader['devel'], data_loader['test']

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=regularization)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10)

    best_val_loss = float('inf')
    best_val_score = -1
    best_model_file = ''
    early_stop = 0

    metrics = {"best_val_loss": best_val_loss, # TODO
                 "best_val_score": best_val_score,
                 "epoch": -1,
                 "lr": lr,
                 "train_loss": float('inf'),
                 "val_loss": float('inf'),
                 "train_score": -1,
                 "val_score": -1,}
    metrics_log = []
    
    for epoch in range(1, epochs + 1):
        metrics.update({"epoch": epoch})        ## TODO update epoch
        print(f'Training for Epoch {epoch}...')
        train_loss = train(model, train_loader, optimizer, loss_fn, use_gpu, device=device)
        metrics.update({"train_loss": train_loss})  ## TODO update train_loss
        val_loss, val_score = evaluate(task, model, val_loader, loss_fn=loss_fn, eval_fn=eval_fn, use_gpu=use_gpu, device=device)
        metrics.update({"val_loss": val_loss, "val_score": val_score})      ## TODO update val_loss, val_score
        
        # Log the learning rate
        for param_group in optimizer.param_groups:
            lr = param_group['lr']
            break

        metrics.update({"lr": lr})
        
        print(f'Epoch:{epoch:>3} / {epochs} | [Train] | Loss: {train_loss:>.4f}')
        print(f'Epoch:{epoch:>3} / {epochs} | [Val] | Loss: {val_loss:>.4f} | [{eval_metric_str}]: {val_score:>7.4f}')
        print('-' * 50)

        if val_score > best_val_score:
            early_stop = 0
            best_val_score = val_score
            best_val_loss = val_loss
            metrics.update({"best_val_score": best_val_score, "best_val_loss": best_val_loss})      ## TODO update best_val_score, best_val_loss
            best_model_file = save_model(model, model_path, identifier)
            _train_loss, train_score = evaluate(task, model, train_loader, loss_fn=loss_fn, eval_fn=eval_fn, use_gpu=use_gpu, device=device)
            metrics.update({"train_score": train_score})      ## TODO update train_score

        else:
            early_stop += 1
            if early_stop >= early_stopping_patience:
                print(f'Note: target can not be optimized for {early_stopping_patience} consecutive epochs, '
                      f'early stop the training process!')
                print('-' * 50)
                break
        
        metrics_log.append(copy.deepcopy(metrics))
        # Step the scheduler based on validation loss
        scheduler.step(val_loss)
        
    print(f'ID/Seed {identifier} | '
          f'Best [Val {eval_metric_str}]:{best_val_score:>7.4f} | Loss: {best_val_loss:>.4f}')
    return best_val_loss, best_val_score, best_model_file, metrics_log