import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

import load_circle_infer as load_data


data_path = '/euler_circle_generator/generated_diagram'


def get_dataloaders_reasoning(path):
    data, labels = [], []
    if not os.path.exists(path + 'dataset'):
        data, labels = load_data.load(data_path, max_num_images)

        utils.save_file(path + 'dataset', data)
        utils.save_file(path + 'labels', labels)
    else:
        data = utils.load_file(path + 'dataset')
        labels = utils.load_file(path + 'labels')
    print(type(data))
    print(type(labels))
    # labels = make_one_hot(labels)
    labels = labels.astype(float)
    length = len(data)

    data = np.transpose(data, (0, 3, 1, 2, 4))

    train_data = data[:int(0.8 * length)]
    train_labels = labels[:int(0.8 * length)]

    valid_data = data[int(0.8 * length):]
    valid_labels = labels[int(0.8 * length):]

    train_data = torch.Tensor(train_data)
    train_labels = torch.Tensor(train_labels)  # .long()

    valid_data = torch.Tensor(valid_data)
    valid_labels = torch.Tensor(valid_labels)  # .long()

    train_dataset = TensorDataset(train_data, train_labels)
    valid_dataset = TensorDataset(valid_data, valid_labels)

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(dataset=valid_dataset, batch_size=batch_size, shuffle=True)
    return train_dataloader, valid_dataloader
