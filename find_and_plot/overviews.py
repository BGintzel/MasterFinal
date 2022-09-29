import matplotlib.pyplot as plt
import numpy as np

from find_and_plot.create_input_output_images import create_normal_output, \
    show_inputs_and_outputs_generator, show_inputs_and_outputs_actmax, data_exploration, show_inputs_and_outputs_arcs, \
    show_inputs_and_outputs_incomplete, show_inputs_and_outputs_color


def get_all_options(title="all possible combinations"):
    imgs1 = []
    imgs2 = []
    labels = [[0, 0, 0, 1], [0, 1, 1, 1], [1, 0, 1, 1], [0, 1, 0, 0], [1, 1, 1, 0], [0, 1, 1, 0], [1, 0, 1, 0],
              [1, 0, 0, 0], [1, 1, 1, 1]]

    for label in labels:
        img1, img2 = data_exploration(label, show=False)
        imgs1.append(img1)
        imgs2.append(img2)

    outputs = ['Red contains Blue', 'Blue contains Red', 'Red partially overlaps with Blue',
               'Red disconnects from Blue']
    empty = ["", "", "", ""]
    plots = []
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(title, fontsize=16)
    for i in range(0, 27):
        plots.append(fig.add_subplot(3, 9, i + 1))
        plots[i].set_xticklabels([])
        plots[i].set_yticklabels([])

    for i in range(0, 9):
        plots[i].imshow(imgs1[i])
        plots[i].set_title(str(labels[i]))
        plots[i + 9].imshow(imgs2[i])
        tabledata = list(zip(empty, labels[i]))
        table = plots[i + 18].table(cellText=tabledata, colLabels=["", "Label"], loc='center')
        cells = table.get_celld()
        for i in range(5):
            cells[(i, 0)].set_width(0)
        table.set_fontsize(40)
        table.scale(1.5, 1.5)

    # plots[0].set_title('normal input')
    #
    # plots[0].imshow(img1_n)
    # plots[3].imshow(img2_n)
    #
    # plots[1].set_title('best input')
    # plots[1].imshow(img1_c)
    # plots[4].imshow(img2_c)
    #
    # plots[2].set_title('second best input')
    # plots[2].imshow(img1_g)
    # plots[5].imshow(img2_g)
    # plots[6].bar(r1, output_n.cpu().detach().numpy(), label='normal', width=bar_width, edgecolor='black')
    # plots[6].bar(r2, output_g.cpu().detach().numpy(), label='second best input', width=bar_width, edgecolor='black')
    # plots[6].bar(r3, output_c.cpu().detach().numpy(), label='best input', width=bar_width, edgecolor='black')
    # plt.xticks([r + bar_width for r in range(len(outputs))], outputs)

    plt.legend()
    plt.show()
    fig.savefig(f'results/Test_.jpg')


def get_overview_arcs(model_path, label, run):
    imgs1 = []
    imgs2 = []
    outputs = []
    output_labels = []
    angles = []

    columns = 7

    cpu = False

    for c in range(0, columns):
        angle = int((360 / (columns - 1)) * (c))
        angles.append(angle)
        img1, img2, output_label, output = show_inputs_and_outputs_arcs(i=1, values=True, circle='', label=label,
                                                                        cpu=cpu, angle=angle, model_path=model_path)
        imgs1.append(img1)
        imgs2.append(img2)
        outputs.append(output)
        output_labels.append(output_label)

    plots = []
    plt.tight_layout()
    fig = plt.figure(figsize=(19, 10))
    fig.suptitle(f'Wrong inputs never seen by any net to create label {label} at 180°', fontsize=22)

    for i in range(0, columns * 3):
        plots.append(fig.add_subplot(3, columns, i + 1))
        plots[i].set_xticklabels([])
        plots[i].set_yticklabels([])
        if i > columns * 2 - 1:
            plots[i].axis('off')
        plots[i].set_xticks([])
        plots[i].set_yticks([])

    for i in range(0, columns):
        plots[i].imshow(imgs1[i])
        plots[i].set_title(f'EN {int(angles[i])}°', fontsize=20)
        plots[i + columns].imshow(imgs2[i])
        # tabledata = list(zip(label, output_labels[i].cpu().detach().numpy().astype(int), np.around(outputs[i].cpu().detach().numpy(), 4)))

        output_label_temp = output_label.cpu().detach().numpy()
        if angles[i] < 301:
            output_label_temp = [0, 0, 0, 0]
        else:
            output_label_temp = [0, 1, 0, 0]
        tabledata = list(zip(["", "", "", ""], np.around(outputs[i].cpu().detach().numpy(), 4)))
        # table = plots[i+ columns*2].table(cellText=tabledata,  colLabels=["Goal Wrong Ouput", "Correct Output", "Output"], loc='center', cellLoc='center')
        table = plots[i + columns * 2].table(cellText=tabledata, colLabels=["", "EN Output"], loc='center',
                                             cellLoc='center')
        cells = table.get_celld()
        for i in range(5):
            cells[(i, 0)].set_width(0)
        table.set_fontsize(40)
        table.scale(2, 3)

    # plt.legend()
    # plt.show()
    fig.savefig(f'results/{run}.png')


def get_overview_incomplete(model_path, label, run):
    imgs1 = []
    imgs2 = []
    outputs = []
    output_labels = []

    columns = 2

    cpu = False

    for c in range(0, columns):
        img1, img2, output_label, output = show_inputs_and_outputs_incomplete(i=c, label=label, cpu=cpu, model_path=model_path)
        imgs1.append(img1)
        imgs2.append(img2)
        outputs.append(output)
        output_labels.append(output_label)

    plots = []
    plt.tight_layout()
    fig = plt.figure(figsize=(19, 10))
    fig.suptitle(f'Incomplete inputs never seen by any net to create label {label}', fontsize=22)

    for i in range(0, columns * 3):
        plots.append(fig.add_subplot(3, columns, i + 1))
        plots[i].set_xticklabels([])
        plots[i].set_yticklabels([])
        if i > columns * 2 - 1:
            plots[i].axis('off')
        plots[i].set_xticks([])
        plots[i].set_yticks([])
    titles = ['Green missing', 'Only green']
    for i in range(0, columns):
        plots[i].imshow(imgs1[i])
        plots[i].set_title(titles[i], fontsize=20)
        plots[i + columns].imshow(imgs2[i])
        # tabledata = list(zip(label, output_labels[i].cpu().detach().numpy().astype(int), np.around(outputs[i].cpu().detach().numpy(), 4)))

        output_label_temp = output_label.cpu().detach().numpy()

        tabledata = list(zip(["", "", "", ""], np.around(outputs[i].cpu().detach().numpy(), 4)))
        # table = plots[i+ columns*2].table(cellText=tabledata,  colLabels=["Goal Wrong Ouput", "Correct Output", "Output"], loc='center', cellLoc='center')
        table = plots[i + columns * 2].table(cellText=tabledata, colLabels=["", "EN Output"], loc='center',
                                             cellLoc='center')
        cells = table.get_celld()
        for i in range(5):
            cells[(i, 0)].set_width(0)
        table.set_fontsize(40)
        table.scale(2, 3)

    # plt.legend()
    # plt.show()
    fig.savefig(f'results/{run}.png')



def get_overview_colors(model_path, label, run):
    imgs1 = []
    imgs2 = []
    outputs = []
    output_labels = []

    columns = 1

    cpu = False

    for c in range(0, columns):
        img1, img2, output_label, output = show_inputs_and_outputs_color(i=c, label=label, cpu=cpu,
                                                                              model_path=model_path)
        imgs1.append(img1)
        imgs2.append(img2)
        outputs.append(output)
        output_labels.append(output_label)

    plots = []
    plt.tight_layout()
    fig = plt.figure(figsize=(19, 10))
    fig.suptitle(f'Different color middle to create label {label}', fontsize=22)

    for i in range(0, columns * 3):
        plots.append(fig.add_subplot(3, columns, i + 1))
        plots[i].set_xticklabels([])
        plots[i].set_yticklabels([])
        if i > columns * 2 - 1:
            plots[i].axis('off')
        plots[i].set_xticks([])
        plots[i].set_yticks([])
    titles = ['Color changed']
    for i in range(0, columns):
        plots[i].imshow(imgs1[i])
        plots[i].set_title(titles[i], fontsize=20)
        plots[i + columns].imshow(imgs2[i])
        # tabledata = list(zip(label, output_labels[i].cpu().detach().numpy().astype(int), np.around(outputs[i].cpu().detach().numpy(), 4)))

        output_label_temp = output_label.cpu().detach().numpy()


        tabledata = list(zip(["", "", "", ""], np.around(outputs[i].cpu().detach().numpy(), 4)))
        # table = plots[i+ columns*2].table(cellText=tabledata,  colLabels=["Goal Wrong Ouput", "Correct Output", "Output"], loc='center', cellLoc='center')
        table = plots[i + columns * 2].table(cellText=tabledata, colLabels=["", "EN Output"], loc='center',
                                             cellLoc='center')
        cells = table.get_celld()
        for i in range(5):
            cells[(i, 0)].set_width(0)
        table.set_fontsize(40)
        table.scale(2, 3)

    # plt.legend()
    # plt.show()
    fig.savefig(f'results/{run}.png')