import scipy.io as sio
import os
import getopt
import sys
import torch
from fastai.vision import ConvLearner
from fastai import error_rate
from fastai.vision import models, ImageDataBunch, imagenet_stats, Image, open_image
from fastai.vision import get_transforms


class CarInference:
    def __init__(self, data_path: str):
        self.data_path = data_path
        path_list, label_list, bbox_list, self.class_names = self.get_paths_and_labels(is_test=0)

        print(f"path_list = {len(path_list)}")
        print(f"label_list = {len(label_list)}")
        print(f"path_list = {path_list[:5]}")

        data = ImageDataBunch.from_lists(self.data_path,
                                         path_list, label_list,
                                         ds_tfms=get_transforms(), size=224, bs=48)
        data.normalize(imagenet_stats)
        self.learn = ConvLearner(data, models.resnet50, metrics=error_rate)
        self.learn.load('golden_model')

    def predict(self, image):
        pred = Image.predict(image, self.learn)
        # pred = [(x, y.item()) for x, y in enumerate(pred)]
        # print(f" Predict result {pred}")
        values, indices = torch.max(pred, 0)
        return self.class_names[indices];

    def get_paths_and_labels(self, is_test=0):
        annos = sio.loadmat(os.path.join(self.data_path, 'cars_annos.mat'))
        class_names = [x[0] for x in annos["class_names"][0]]

        _, total_size = annos["annotations"].shape
        file_path_list = []
        label_list = []
        bbox_list = []
        print("total sample size is ", total_size)
        for i in range(total_size):
            file_name = annos["annotations"][:, i][0][0][0].split("/")[1]
            bbox_x1 = annos["annotations"][:, i][0][1][0]
            bbox_y1 = annos["annotations"][:, i][0][2][0]
            bbox_x2 = annos["annotations"][:, i][0][3][0]
            bbox_y2 = annos["annotations"][:, i][0][4][0]
            class_index = annos["annotations"][:, i][0][5][0][0]
            #         print(f"{class_index}")
            class_name = class_names[class_index - 1]
            _test = annos["annotations"][:, i][0][6][0]
            if is_test != int(_test):
                file_path_list.append(f"{self.data_path}//{file_name}")
                label_list.append(class_name)
                bbox_list.append((bbox_x1, bbox_y1, bbox_x2, bbox_y2))

        return file_path_list, label_list, bbox_list, class_names


def processImage(car_inferer, image_path):
    print(f"Processing image {image_path}")
    img = open_image(image_path)
    output = car_inferer.predict(img)
    print(f" {output}")


def main(argv):
    try:
        opts, args = getopt.getopt(argv, "h:i:q", ["input="])
    except getopt.GetoptError:
        print("QATrainer.py [-c <config_file>] [-i <input_file>] [-o <output_file>")
        sys.exit(2)

    input_file = None
    for opt, arg in opts:
        if opt in ("-i", "--input"):
            input_file = arg

    data_path = "/Users/sathya/work/data/cropped_car"

    car_infer = CarInference(data_path)

    if input_file:
        processImage(car_infer, input_file)

    while True:
        inpt = input('\> ')
        if inpt == "?":
            print(f"Provide full path of the image")
            continue
        if inpt == 'exit' or inpt == 'q':
            exit()
        processImage(car_infer, inpt)


if __name__ == '__main__':
    main(sys.argv[1:])
