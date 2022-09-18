import utils


def trainer_combine(run, writer, nb_epoch=5, model=None, train_loader=None, valid_loader=None, EN=None):
    global DEVICE
    DEVICE = utils.set_device()
    if not train_loader:
        train_loader, valid_loader = data.get_dataloaders_reasoning('')
    if not model:
        model = models.Combine().to(DEVICE)
    else:
        model = model.to(DEVICE)
    if EN:
        EN.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0002, betas=(0.9, 0.9), eps=1e-08, weight_decay=0)
    criterion = torch.nn.BCELoss()

    stats = train.train(model, train_loader, valid_loader, criterion, optimizer, run, nb_epochs=nb_epoch,
                        after_evaluation_hook=train.stats_saver('models/combine_' + run + '.pkl'), writer=writer, EN=EN)
    return stats
