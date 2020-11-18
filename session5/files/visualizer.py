import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch.nn as nn


# noinspection SpellCheckingInspection
class Visualizer(object):
    # noinspection SpellCheckingInspection
    def __init__(self, model, trans, size, device):
        self.model = model
        self.model_weights, self.conv_layers = self.configs()
        self.trans = trans
        self.size = size
        self.device = device

    def configs(self):

        model_weights = []
        conv_layers = []

        model_children = list(self.model.children())

        counter = 0

        for i in range(len(model_children)):
            if type(model_children[i]) == nn.Conv2d:
                counter += 1
                model_weights.append(model_children[i].weight)
                conv_layers.append(model_children[i])
            elif type(model_children[i]) == nn.Sequential:
                for j in range(len(model_children[i])):
                    for child in model_children[i][j].children():
                        if type(child) == nn.Conv2d:
                            counter += 1
                            model_weights.append(child.weight)
                            conv_layers.append(child)
        print(f"Total convolutional layers: {counter}")

        for weight, conv in zip(model_weights, conv_layers):
            print(f"CONV: {conv} ====> SHAPE: {weight.shape}")
        return model_weights, conv_layers

    def vis_feature_map(self, org_img, num_layer=0, num_subplot=9, path_save='./results'):
        img = cv2.resize(org_img, dsize=self.size)
        img_trans = self.trans(img).to(self.device)

        results = [self.conv_layers[0](img_trans)]
        for i in range(1, len(self.conv_layers)):
            results.append(self.conv_layers[i](results[-1]))

        outputs = results
        plt.figure()
        layer_viz = outputs[num_layer][0, :, :, :]
        layer_viz = layer_viz.data

        if len(layer_viz) > num_subplot:
            pass
        else:
            num_subplot = len(layer_viz)
        c = np.ceil(np.sqrt(num_subplot))
        r = np.ceil(num_subplot / (np.ceil(np.sqrt(num_subplot))))
        plt.suptitle(f"feature maps of layer:{num_layer}")
        for i, layer in enumerate(layer_viz):
            if i == num_subplot:
                break
            plt.subplot(c, r, i + 1)
            plt.imshow(layer, cmap='gray')
            plt.axis("off")
        print(f"Saving layer {num_layer} feature maps...")
        plt.savefig(f"./{path_save}/layer_{num_layer}_feature_maps.png")

    def vis_filters(self, num_layer=0, path_save='./results'):
        plt.figure()
        num_subplot = len(self.model_weights[num_layer])
        c = np.ceil(np.sqrt(num_subplot))
        r = np.ceil(num_subplot / (np.ceil(np.sqrt(num_subplot))))
        # num_weights = int(np.sqrt(len(self.model_weights[num_layer]))) + 1
        plt.suptitle(f"filters of layer:{num_layer}")
        for i, filter_ in enumerate(self.model_weights[num_layer]):
            plt.subplot(c, r, i + 1)
            plt.imshow(filter_[0, :, :].detach(), cmap='gray')
            plt.axis('off')
        print(f"Saving layer {num_layer} filters...")
        plt.savefig(f"./{path_save}/layer_{num_layer}_filters.png")

