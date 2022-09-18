import torch
import os
import torch.nn as nn
from tqdm import tqdm
import numpy as np

from mlhelp import (
    model_utils,
    plotting,
    train,
    utils,
    models
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_statistics():
    return {
        "train_batch_losses": [],
        "train_batch_accuracies": [],
        "test_batch_losses": [],
        "test_batch_accuracies": [],
        "train_epoch_losses": [],
        "test_epoch_losses": [],
        "train_epoch_accuracies": [],
        "test_epoch_accuracies": [],
        "test_batch_losses_EN": [],
        "test_batch_accuracies_EN": [],
        "test_epoch_losses_EN": [],
        "test_epoch_accuracies_EN": [],
    }


def stats_saver(path):
    def stats_saver_hook(statistics, **kwargs):
        utils.save_file(os.path.join(path), statistics)

    return stats_saver_hook


def count_correct_predictions(prediction, target):
    """
    Args:
      prediction: tensor of one-hot encoded predicted labels; shape (N, C)
        where C is the number of classes
      target: tensor (of int) of class ids; shape (N)
    """
    return np.sum(np.where(np.sum(torch.abs(prediction - target).cpu().detach().numpy(), axis=1) < 0.1, 1, 0))


def train_loop(model, loader, criterion, optimizer, statistics, epoch=1, use_tqdm=True, writer=None):
    statistics["train_batch_losses"].append([])
    statistics["train_batch_accuracies"].append([])
    batch_correct_count = 0

    if use_tqdm:
        loader = tqdm(enumerate(loader), total=len(loader))
    else:
        loader = enumerate(loader)

    for i, (imgs, target_labels) in loader:
        imgs = imgs.to(DEVICE)
        target_labels = target_labels.to(DEVICE)

        batch_size = imgs.size(dim=0)

        pred_labels = model(imgs)

        loss = criterion(pred_labels, target_labels)
        statistics["train_batch_losses"][-1].append(loss.item())

        correct_pred = count_correct_predictions(pred_labels, target_labels)
        batch_correct_count += correct_pred

        acc = correct_pred / batch_size * 100
        statistics["train_batch_accuracies"][-1].append(acc)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if use_tqdm:
            loader.set_description(
                f"Epoch {epoch + 1:>3} Batch {i + 1:>4}/{len(loader)}: "
                f"Loss {loss.item():>8.5f} Accuracy {acc:>6.2f}%"
            )

        writer.add_scalar(f'TestLoss', loss.item(), global_step=i + batch_size * epoch)
        writer.add_scalar(f'TestAccuracy', acc, global_step=i + batch_size * epoch)

    return batch_correct_count


@torch.no_grad()
def test_loop(model, loader, criterion, statistics, epoch=1, use_tqdm=True, writer=None):
    statistics["test_batch_losses"].append([])
    statistics["test_batch_accuracies"].append([])

    batch_correct_count = 0

    if use_tqdm:
        loader = tqdm(enumerate(loader), total=len(loader))
    else:
        loader = enumerate(loader)

    for i, (imgs, target_labels) in loader:
        imgs = imgs.to(DEVICE)
        target_labels = target_labels.to(DEVICE)

        batch_size = imgs.size(dim=0)

        pred_labels = model(imgs)

        loss = criterion(pred_labels, target_labels)

        correct_pred = count_correct_predictions(pred_labels, target_labels)
        batch_correct_count += correct_pred

        acc = correct_pred / batch_size * 100

        statistics["test_batch_losses"][-1].append(loss.item())
        statistics["test_batch_accuracies"][-1].append(acc)

        if use_tqdm:
            loader.set_description(
                f"Epoch {epoch + 1:>3} Batch {i + 1:>4}/{len(loader)}: "
                f"Loss {loss.item():>8.5f} Accuracy {acc:>6.2f}%"
            )

        writer.add_scalar(f'TestLoss', loss.item(), global_step=i + batch_size * epoch)
        writer.add_scalar(f'TestAccuracy', acc, global_step=i + batch_size * epoch)

    return batch_correct_count


def train(model, train_loader, test_loader, criterion, optimizer, run='',
          nb_epochs=1, after_evaluation_hook=None, use_tqdm=True, save_best=True, stats=None, writer=None):
    if stats:
        statistics = stats
    else:
        statistics = create_statistics()

    for epoch in range(0, nb_epochs):
        model.train()
        counts = train_loop(model, train_loader, criterion, optimizer, statistics, epoch,
                            use_tqdm=use_tqdm, writer=writer)
        statistics["train_epoch_losses"].append(np.mean(statistics["train_batch_losses"][-1]))
        train_dataset_size = len(train_loader.dataset)
        epoch_acc = counts / train_dataset_size * 100
        statistics["train_epoch_accuracies"].append(epoch_acc)

        model.eval()

        # model test
        counts = test_loop(model, test_loader, criterion, statistics, epoch,
                           use_tqdm=use_tqdm, writer=writer)
        statistics["test_epoch_losses"].append(np.mean(statistics["test_batch_losses"][-1]))
        test_dataset_size = len(test_loader.dataset)
        epoch_acc = counts / test_dataset_size * 100
        statistics["test_epoch_accuracies"].append(epoch_acc)

        if save_best and statistics["test_epoch_losses"][-1] == min(statistics["test_epoch_losses"]):
            model_utils.save_model_simple(model, optimizer, epoch, statistics, prefix='best' + run)

        if epoch == nb_epochs - 1:
            model_utils.save_model_simple(model, optimizer, epoch, statistics, prefix='last' + run)
            model_utils.save_model_simple(model, optimizer, epoch, statistics, prefix='current')

        if after_evaluation_hook:
            after_evaluation_hook(epoch=epoch, statistics=statistics)

        writer.add_scalar(f'TestEpochLoss', statistics["test_epoch_losses"][-1], global_step=epoch)
        writer.add_scalar(f'TestEpochAccuracy', statistics["test_epoch_accuracies"][-1], global_step=epoch)

        writer.add_scalar(f'TrainEpochLoss', statistics["train_epoch_losses"][-1], global_step=epoch)
        writer.add_scalar(f'TrainEpochAccuracy', statistics["train_epoch_accuracies"][-1], global_step=epoch)

        print(statistics["test_epoch_accuracies"][-1])

    return statistics
