import data
import models
import utils
import train
import torch


def trainer_EN(run, writer, nb_epoch=5, model=None, train_loader=None, valid_loader=None):

    device = utils.set_device()
    if not train_loader:
        train_loader, valid_loader = data.get_dataloaders_reasoning('')
    if not model:
        model = models.EN().to(device)
    else:
        model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.9), eps=1e-08, weight_decay=0)
    criterion = torch.nn.BCELoss()

    stats = train.train(model, train_loader, valid_loader, criterion, optimizer, run, nb_epochs=nb_epoch,
                        after_evaluation_hook=train.stats_saver('models/EN_' + run + '.pkl'), writer=writer)
    return stats
