import os.path

import torch
# import torchinfo
PATH = '/home/gintzel/PycharmProjects/pytorchENOpen/'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_model_params(model):
    """ Counting the number of learnable parameters in a nn.Module """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return num_params


def save_model(model, optimizer, scheduler, epoch, stats, prefix=''):
    """ Saving model checkpoint """
    models_path = os.path.join(PATH, "models")
    if (not os.path.exists(models_path)):
        os.makedirs(models_path)

    cp_prefix = getattr(model, 'checkpoint_prefix', '')
    if prefix != '':
        savepath = os.path.join(models_path, f"{prefix}_{cp_prefix}_checkpoint.pth")
    else:
        savepath = os.path.join(models_path, f"{cp_prefix}_checkpoint_epoch_{epoch + 1}.pth")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'stats': stats
    }, savepath)


def save_model_simple(model, optimizer, epoch, stats, prefix=''):
    """ Saving model checkpoint """
    models_path = os.path.join(PATH, "models")
    if not os.path.exists(models_path):
        os.makedirs(models_path)

    cp_prefix = getattr(model, 'checkpoint_prefix', '')
    if prefix != '':
        savepath = os.path.join(models_path, f"{prefix}_{cp_prefix}_checkpoint.pth")
    else:
        savepath = os.path.join(models_path, f"{cp_prefix}_checkpoint_epoch_{epoch + 1}.pth")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'stats': stats
    }, savepath)


def load_model(model, optimizer, scheduler, savepath):
    """ Loading pretrained checkpoint """
    checkpoint = torch.load(savepath, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]

    return model, optimizer, scheduler, epoch, stats


def load_model_simple(model, optimizer, savepath):
    """ Loading pretrained checkpoint """
    checkpoint = torch.load(savepath, map_location=DEVICE)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint["epoch"]
    stats = checkpoint["stats"]

    return model, optimizer, epoch, stats