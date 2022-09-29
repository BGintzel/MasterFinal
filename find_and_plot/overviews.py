import matplotlib.pyplot as plt
import numpy as np

from find_and_plot.create_input_output_images import show_inputs_and_outputs_circles, create_normal_output, \
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


def get_overview_compare(label):
    titles = ['Iteration 0 EN',
              'Iteration 19 EN',
              'Iteration 0 EN',
              'Iteration 19 EN',
              'Iteration 0 EN',
              'Iteration 19 EN'
              ]

    imgs1 = []
    imgs2 = []
    outputs = []
    output_labels = []

    input = None
    flag = True
    cpu = False
    run = 4

    for net in range(0, 2):
        img1, img2, output_label, output = show_inputs_and_outputs_circles(run=run, label=label, net=net, input=input,
                                                                           flag=flag, cpu=cpu)
        imgs1.append(img1)
        imgs2.append(img2)
        outputs.append(output)
        output_labels.append(output_label)

    input = None
    flag = True
    cpu = False
    run = 5

    for net in range(0, 2):
        img1, img2, output_label, output = show_inputs_and_outputs_circles(run=run, label=label, net=net, input=input,
                                                                           flag=flag, cpu=cpu)
        imgs1.append(img1)
        imgs2.append(img2)
        outputs.append(output)
        output_labels.append(output_label)

    run = 0
    input = None
    flag = True
    cpu = False

    for net in range(0, 2):
        img1, img2, output_label, output = show_inputs_and_outputs_circles(run=run, label=label, net=net, input=input,
                                                                           flag=flag, cpu=cpu)
        imgs1.append(img1)
        imgs2.append(img2)
        outputs.append(output)
        output_labels.append(output_label)

    # outputs = ['Red contains Blue', 'Blue contains Red', 'Red partially overlaps with Blue', 'Red disconnects from Blue']

    empty = ["", "", "", ""]

    # bar_width = 0.25
    # r1 = np.arange(len(output_n))
    # r2 = [x + bar_width for x in r1]
    # r3 = [x + bar_width for x in r2]

    plots = []
    plt.tight_layout()
    fig = plt.figure(figsize=(19, 10))
    # fig.suptitle(f'Wrong inputs never seen by any Net for label {label}', fontsize=16)
    for i in range(0, 18):
        plots.append(fig.add_subplot(3, 6, i + 1))
        plots[i].set_xticklabels([])
        plots[i].set_yticklabels([])
        if i > 11:
            plots[i].axis('off')
        plots[i].set_xticks([])
        plots[i].set_yticks([])

    for i in range(0, 6):
        plots[i].imshow(imgs1[i])
        plots[i].set_title(titles[i], fontsize=20)
        plots[i + 6].imshow(imgs2[i])
        # tabledata = list(zip(label, output_labels[i].cpu().detach().numpy().astype(int), np.around(outputs[i].cpu().detach().numpy(), 4)))
        if i < 4:
            output_label = [0, 0, 0, 0]
        else:
            output_label = [0, 1, 1, 1]

        tabledata = list(zip(output_label, np.around(outputs[i].cpu().detach().numpy(), 4)))
        # print(tabledata)
        # table = plots[i+12].table(cellText=tabledata,  colLabels=["Fake Label", "Real Label", "Output"], loc='center', cellLoc='center')
        table = plots[i + 12].table(cellText=tabledata, colLabels=["Expected", "EN Output"], loc='center',
                                    cellLoc='center')
        cells = table.get_celld()
        # for i in range(5):
        #     cells[(i, 0)].set_width(0)
        table.set_fontsize(40)
        table.scale(1, 3)

    # plt.legend()
    # plt.show()
    fig.savefig(f'results/label_{label}_all.jpg')


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



def get_overview_new(label, img):
    imgs1 = []
    imgs2 = []
    outputs = []
    output_labels = []
    flag = False
    run = 0

    # images = ['normal_input', 'generator', 'act_max', 'random_search']
    if img == 'normal_input':
        input, img1, img2, output = create_normal_output(label)
        cpu = True

    elif img == 'generator':
        input, img1, img2, output = show_inputs_and_outputs_generator()
        cpu = False

    elif img == 'act_max':
        input, img1, img2, output = show_inputs_and_outputs_actmax()
        cpu = True

    elif img == 'random_search':
        input = None
        flag = True
        cpu = False

    elif img == 'incomplete1':
        input = None
        flag = True
        cpu = False
        run = 4

    elif img == 'incomplete2':
        input = None
        flag = True
        cpu = False
        run = 5

    img1, img2, output_label, output = show_inputs_and_outputs_circles(run=run, label=label, net=0, input=input,
                                                                       flag=flag, cpu=cpu)
    imgs1.append(img1)
    imgs2.append(img2)
    outputs.append(output)
    output_labels.append(output_label)

    plots = []

    plt.tight_layout()
    fig = plt.figure(figsize=(15, 10))
    # fig.suptitle(f'Wrong inputs never seen by any Net for label {label}')
    for i in range(0, 3):
        if i == 1:
            plots.append(fig.add_subplot(2, 2, (i + 1, i + 3)))
        else:
            plots.append(fig.add_subplot(2, 2, i + 1))

        plots[i].set_xticklabels([])
        plots[i].set_yticklabels([])
        if i == 1:
            plots[i].axis('off')
        plots[i].set_xticks([])
        plots[i].set_yticks([])

    plots[0].imshow(imgs1[0])
    plots[0].set_title('Intput image 1', fontsize=35)
    plots[2].imshow(imgs2[0])
    plots[2].set_title('Intput image 2', fontsize=35)

    if img == 'normal_input' or img == 'incomplete1' or img == 'incomplete2' or img == 'act_max' or img == 'generator':
        if img == 'normal_input':
            lbl = 'Label'
            tabledata = list(zip(label, np.around(outputs[0].cpu().detach().numpy(), 4)))
            table = plots[1].table(cellText=tabledata, colLabels=[lbl, "Output"], loc='center', cellLoc='center',
                                   fontsize=25)
        else:
            lbl = ''
            tabledata = list(zip(['', '', '', ''], np.around(outputs[0].cpu().detach().numpy(), 4)))
            table = plots[1].table(cellText=tabledata, colLabels=[lbl, "Output"], loc='center', cellLoc='center',
                                   fontsize=25)
            cells = table.get_celld()
            for i in range(5):
                cells[(i, 0)].set_width(0)
    else:
        tabledata = list(zip(output_labels[0].cpu().detach().numpy().astype(int),
                             np.around(outputs[0].cpu().detach().numpy(), 4)))
        table = plots[1].table(cellText=tabledata, colLabels=["Label", "Output"], loc='center',
                               cellLoc='center', fontsize=25)

    plots[1].set_title('Output of Euler net', fontsize=35)

    table.set_fontsize(100)
    table.scale(1, 8)

    plt.legend()
    # plt.show()
    fig.savefig(f'results/label_{label}_{img}_show.jpg')



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