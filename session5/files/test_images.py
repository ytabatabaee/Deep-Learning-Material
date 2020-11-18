import torch
import tensorflow as tf
import numpy as np
from train import normalize_dataset
from Networks import CNN_Net as Net
# from Networks import FC_Net as Net
from torchvision import transforms
from skimage import transform as trans
import matplotlib.pyplot as plt
import cv2
from visualizer import Visualizer

a = 3
max_output_size = 2
IOU = 0.5
image_name = "22143.png"


def expand_dim(patch):
    patch = torch.unsqueeze(patch, 0)
    return patch.type(torch.float32)


transform = transforms.Compose([
    transforms.ToTensor(),
    normalize_dataset,
    expand_dim
])


def sliding_window(img, patch_size=None,
                   istep=5, jstep=5, scale=1.0):
    Ni, Nj = (int(scale * s) for s in patch_size)
    for i in range(0, img.shape[0] - Ni, int(scale * istep)):
        for j in range(0, img.shape[1] - Nj, int(scale * jstep)):
            patch = img[i:i + Ni, j:j + Nj]
            if scale != 1:
                patch = trans.resize(patch, patch_size)
            # patch = np.transpose(patch, (2,0,1))

            yield (i, j), patch


def test(img, model, patch_size):
    # image = color.rgb2gray(img)
    image = img
    Ni, Nj = patch_size
    All_boxes = []
    All_scores = []
    LAbbels = []

    downscale = 1.28
    pyramid = trans.pyramid_gaussian(image, max_layer=7, downscale=downscale, multichannel=False)

    # pyramid = [image]
    for i, dawn_image in enumerate(pyramid):
        try:
            indices, patches = zip(*sliding_window(dawn_image, patch_size=patch_size))
        except:
            break

        outputs = np.array([model(transform(patch).to(device)) for patch in patches])
        labels = np.array([np.argmax(output.detach().numpy()) + 1 for output in outputs])
        scores = np.array([np.max(output.detach().numpy()) for output in outputs])

        indices = np.uint16(np.array(indices))

        if list(indices[labels != 0]) == []:
            continue

        num_boxes = len(indices[labels != 0])
        boxes = np.zeros([num_boxes, 4])

        scores_ = scores[labels != 0]
        boxes[:, 0] = indices[labels != 0][:, 0]
        boxes[:, 1] = indices[labels != 0][:, 0] + Ni
        boxes[:, 2] = indices[labels != 0][:, 1]
        boxes[:, 3] = indices[labels != 0][:, 1] + Nj

        boxes *= img.shape[0] / dawn_image.shape[0]
        boxes = np.uint32(boxes)

        All_boxes.extend(boxes)
        All_scores.extend(scores_)
        LAbbels.extend(labels)

    All_boxes_ = np.array(All_boxes)
    All_scores_ = np.array(All_scores)
    LAbbels_ = np.array(LAbbels)

    a = tf.image.non_max_suppression(
        All_boxes_,
        All_scores_,
        max_output_size,
        iou_threshold=IOU,
        name=None
    )
    sess = tf.Session()
    with sess.as_default():
        nms = a.eval()

    return All_boxes_[nms], LAbbels_[nms]


def show(image, Boxes=None, labels=None):
    fig, ax = plt.subplots()
    ax.imshow(image[:, :, ::-1])
    if Boxes is not None:
        for i, cor in enumerate(Boxes):
            ax.add_patch(plt.Rectangle((cor[2], cor[0]), int(cor[3] - cor[2]), int(cor[1] - cor[0]), edgecolor='green',
                                       alpha=0.9, lw=3,
                                       facecolor='none'))
            ax.text(cor[2], cor[0], "{0}".format(labels[i] % 10), color='red', fontsize=14)


if __name__ == "__main__":
    org_img = cv2.imread("./test_images/" + image_name)
    device = torch.device("cpu")
    model = Net().to(device)
    # myload = torch.load("./SVHN_FC.pt")
    myload = torch.load("./cnn_83.pt")
    try:
        model.load_state_dict(myload['state_dict'])
    except:
        model.load_state_dict(myload)

    model.eval()

    img = cv2.resize(org_img, dsize=(0, 0), fx=a, fy=a)
    boxes, labels = test(img, model, patch_size=(32, 32))
    show(img, boxes, labels)

    try:
        visualization = Visualizer(model, transform, size=(32, 32), device=device)
        visualization.vis_filters(num_layer=2, path_save='./results')
        visualization.vis_feature_map(org_img, num_layer=2, num_subplot=11, path_save='./results')
        plt.show()
        plt.close()
    except:
        plt.show()
        plt.close()