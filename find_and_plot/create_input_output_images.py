import cv2

import euler_circle_generator.euler_circle_lib
import numpy as np
import torch

from find_and_plot import train_circles
import utils
import data

DEVICE = utils.set_device()


def data_exploration(label, show=True):
    train_loader, valid_loader = data.get_dataloaders_reasoning('')
    return data.data_exploration(train_loader, l=label, show=show)


def show_inputs_and_outputs_actmax():
    input = utils.load_file('outputimage')
    input = torch.Tensor(input)
    input = input.to(DEVICE)
    # input = input.permute(0, 2, 3, 1, 4)

    img1 = input[0, :, :, :, 0].permute(1, 2, 0).cpu().detach().numpy()
    img2 = input[0, :, :, :, 1].permute(1, 2, 0).cpu().detach().numpy()

    img1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    img2 = cv2.normalize(img2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

    # plt.imshow(img1)
    # plt.show()
    #
    # plt.imshow(img2)
    # plt.show()

    comp = torch.randn(2, 3, 64, 64, 2).to(DEVICE)
    comp[0, :, :, :, :] = input
    comp[1, :, :, :, :] = input

    model = revelio.load_model()
    model = model.to(DEVICE)
    model.eval()

    # print(model(comp)[0])

    return input, img1, img2, model(comp)[0]


def show_inputs_and_outputs_circles(run, label, net, input=None, flag=False, cpu=False, out=[[-100, -100, -100, -100]]):
    label = np.array(label, dtype='f')

    if flag:
        if not run == 0:
            input = utils.load_file(f'experiments/old/run{run}_imgs.pkl')[-1]
            out = utils.load_file(f'experiments//old/run{run}_out.pkl')[-1]
        else:
            input = utils.load_file(f'experiments/run{run}_label_{label}_imgs.pkl')[-1]
            # out = [utils.load_file(f'experiments/run{run}_label_{label}_imgs.pkl')[-1]]
    if cpu:
        input = torch.Tensor(input.cpu())
    else:
        input = torch.Tensor(input)

    input = input.to(DEVICE)
    # input = input.permute(0, 2, 3, 1, 4)

    img1 = input[0, :, :, :, 0].permute(1, 2, 0).cpu().detach().numpy()
    img2 = input[0, :, :, :, 1].permute(1, 2, 0).cpu().detach().numpy()

    img1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    img2 = cv2.normalize(img2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

    output_label = train_circles.create_label(out)

    comp = torch.randn(2, 3, 64, 64, 2).to(DEVICE)
    comp[0, :, :, :, :] = input[0]
    comp[1, :, :, :, :] = input[0]

    pre = f'/home/gintzel/PycharmProjects/ENCircularTraining/models/old'

    list_of_paths = [f'/home/gintzel/PycharmProjects/ENSelf-Training/models/EN.pth',
                     # f'{pre}/bestcircular_0__checkpoint.pth',
                     # f'{pre}/bestcircular_1__checkpoint.pth',
                     # f'{pre}/bestcircular_4__checkpoint.pth',
                     # f'{pre}/bestcircular_9__checkpoint.pth',
                     f'{pre}/bestcircular_18__checkpoint.pth']

    path = list_of_paths[net]

    model = utils.load_model(path)
    model = model.to(DEVICE)
    model.eval()

    return img1, img2, output_label, model(comp)[0]


def make_input(input1, input2):
    images = np.zeros((1, 3, 64, 64, 2), dtype='f')

    images[0, :, :, :, 0] = input1 / 256
    images[0, :, :, :, 1] = input2 / 256

    return torch.from_numpy(images)


def correct_out(out):
    order = [2, 3, 0, 1, 5, 4, 6]

    out[1] = [out[1][i] for i in order]
    return out


def show_inputs_and_outputs_arcs(i, circle, label, model_path,values=False, cpu=False, out=[[-100, -100, -100, -100]], angle=None):
    label = np.array(label, dtype='f')

    if values:
        out = utils.load_file(f'experiments/arc_{label}_out.pkl')[-i]
        # out = correct_out(out)
        if angle == 360:
            angle -= 1
        input1, out1 = euler_circle_generator.euler_circle_lib.make_image(experiments=[True, True, False, True, False],
                                                                          image=1, value=True, values=out[0], arc=True,
                                                                          start=out[0][-1], angle=angle)
        input2, out2 = euler_circle_generator.euler_circle_lib.make_image(experiments=[True, True, False, True, False],
                                                                          image=2, value=True, values=out[1], arc=True,
                                                                          start=out[1][-1], angle=angle)

        input = make_input(input1, input2)

    else:
        input = utils.load_file(f'experiments/run_arc_{circle}_{label}_full_green_circles_better_imgs.pkl')[-1]

    if cpu:
        input = torch.Tensor(input.cpu())
    else:
        input = torch.Tensor(input)

    input = input.to(DEVICE)
    # input = input.permute(0, 2, 3, 1, 4)

    img1 = input[0, :, :, :, 0].permute(1, 2, 0).cpu().detach().numpy()
    img2 = input[0, :, :, :, 1].permute(1, 2, 0).cpu().detach().numpy()

    img1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    img2 = cv2.normalize(img2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

    output_label = train_circles.create_label(out)

    comp = torch.randn(2, 3, 64, 64, 2).to(DEVICE)
    comp[0, :, :, :, :] = input[0]
    comp[1, :, :, :, :] = input[0]

    model = utils.load_model(model_path)
    model = model.to(DEVICE)
    model.eval()

    return img1, img2, output_label, model(comp)[0]


def show_inputs_and_outputs_incomplete(i, label, model_path, cpu=False, out=[[-100, -100, -100, -100]]):
    label = np.array(label, dtype='f')
    input = utils.load_file(f'experiments/incomplete_{i}_{label}_imgs.pkl')[-1]

    if cpu:
        input = torch.Tensor(input.cpu())
    else:
        input = torch.Tensor(input)

    input = input.to(DEVICE)
    # input = input.permute(0, 2, 3, 1, 4)

    img1 = input[0, :, :, :, 0].permute(1, 2, 0).cpu().detach().numpy()
    img2 = input[0, :, :, :, 1].permute(1, 2, 0).cpu().detach().numpy()

    img1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    img2 = cv2.normalize(img2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

    output_label = train_circles.create_label(out)

    comp = torch.randn(2, 3, 64, 64, 2).to(DEVICE)
    comp[0, :, :, :, :] = input[0]
    comp[1, :, :, :, :] = input[0]

    model = utils.load_model(model_path)
    model = model.to(DEVICE)
    model.eval()

    return img1, img2, output_label, model(comp)[0]


def show_inputs_and_outputs_color(i, label, model_path, cpu=False, out=[[-100, -100, -100, -100]]):
    label = np.array(label, dtype='f')
    input = utils.load_file(f'experiments/color_{label}_imgs.pkl')[-1]

    if cpu:
        input = torch.Tensor(input.cpu())
    else:
        input = torch.Tensor(input)

    input = input.to(DEVICE)
    # input = input.permute(0, 2, 3, 1, 4)

    img1 = input[0, :, :, :, 0].permute(1, 2, 0).cpu().detach().numpy()
    img2 = input[0, :, :, :, 1].permute(1, 2, 0).cpu().detach().numpy()

    img1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    img2 = cv2.normalize(img2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

    output_label = train_circles.create_label(out)

    comp = torch.randn(2, 3, 64, 64, 2).to(DEVICE)
    comp[0, :, :, :, :] = input[0]
    comp[1, :, :, :, :] = input[0]

    model = utils.load_model(model_path)
    model = model.to(DEVICE)
    model.eval()

    return img1, img2, output_label, model(comp)[0]



def show_inputs_and_outputs_generator():
    generator = revelio.load_model(type='generator').to(DEVICE)
    noise = torch.randn((1, 100, 1, 1), device=DEVICE)
    noise = noise.to(DEVICE)
    input = generator(noise)

    img1 = input[0, :, :, :, 0].permute(1, 2, 0).cpu().detach().numpy()
    img2 = input[0, :, :, :, 1].permute(1, 2, 0).cpu().detach().numpy()

    img1 = cv2.normalize(img1, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)
    img2 = cv2.normalize(img2, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F).astype(np.uint8)

    # plt.imshow(img1)
    # plt.show()
    #
    # plt.imshow(img2)
    # plt.show()

    comp = torch.randn(2, 3, 64, 64, 2).to(DEVICE)
    # print(input.size())
    comp[0, :, :, :, :] = input
    comp[1, :, :, :, :] = input

    model = revelio.load_model()
    model = model.to(DEVICE)
    model.eval()

    # print(model(comp)[0])

    return input, img1, img2, model(comp)[0]


def create_normal_output(label):
    img1, img2, input = data_exploration(show=False, label=label)

    model = revelio.load_model()
    model = model.to(DEVICE)
    model.eval()
    # print(input.size())

    return input, img1, img2, model(input.to(DEVICE))[0]


def act_max(steps=100000):
    unit = 1  # flamingo class of Imagenet
    alpha = torch.tensor(100)  # learning rate (step size)
    verbose = True  # print activation every step
    L2_Decay = False  # enable L2 decay regularizer
    Gaussian_Blur = True  # enable Gaussian regularizer
    Norm_Crop = False  # enable norm regularizer
    Contrib_Crop = True  # enable contribution regularizer

    activation_dictionary = {}
    layer_name = 'classifier_final'

    model = revelio.load_model()
    model = model.to(DEVICE)
    model.eval()

    model.fc3.register_forward_hook(amu.layer_hook(activation_dictionary, layer_name))

    input = torch.randn(3, 64, 64, 2)
    input = input.unsqueeze(0)

    input.requires_grad_(True)

    input = input.to(DEVICE)

    output = amu.act_max(network=model,
                         input=input,
                         layer_activation=activation_dictionary,
                         layer_name=layer_name,
                         unit=unit,
                         steps=steps,
                         alpha=alpha,
                         verbose=verbose,
                         L2_Decay=L2_Decay,
                         Gaussian_Blur=Gaussian_Blur,
                         Norm_Crop=Norm_Crop,
                         Contrib_Crop=Contrib_Crop,
                         device=DEVICE
                         )

    # final_image = amu.image_converter(output.squeeze(0))

    # final_image = final_image * 255

    # plt.imshow(final_image)
    # plt.show()
    # cv2.imshow('final image', final_image)
    # cv2.waitKey(0)

    # uncomment to save the image
    # cv2.imwrite('final image' + '.jpg', final_image)
    utils.save_file('outputimage', output)
    # print(model(final_image))


