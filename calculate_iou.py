import os
from PIL import Image
import torch
import torchvision
import torchmetrics
from prettytable import PrettyTable


def calculate_iou(pred_dir, target_dir, file_format='.bmp'):
    metric = torchmetrics.JaccardIndex(task='binary')

    # move the metric to device you want computations to take place
    device = "cuda" if torch.cuda.is_available() else "cpu"
    metric.to(device)

    preds = [f for f in os.listdir(pred_dir) if f.endswith(file_format)]
    targets = [f for f in os.listdir(target_dir) if f.endswith(file_format)]

    ones_iou = []
    zeros_iou = []

    for file in preds:
        pred_read_path = os.path.join(pred_dir, file)
        # print(pred_read_path)

        target_read_path = os.path.join(target_dir, file.replace('_', 'mask_'))
        # print(target_read_path)

        # read image
        pred = Image.open(pred_read_path)
        target = Image.open(target_read_path)

        # convert to tensor
        pred = torchvision.transforms.functional.pil_to_tensor(pred)
        target = torchvision.transforms.functional.pil_to_tensor(target)

        # convert to float
        pred = torchvision.transforms.functional.convert_image_dtype(pred)
        target = torchvision.transforms.functional.convert_image_dtype(target)

        pred_toggle = 1 - pred
        target_toggle = 1 - target
        # calculate IOU
        pred = pred.to(device)
        target = target.to(device)

        pred_toggle= pred_toggle.to(device)
        target_toggle = target_toggle.to(device)

        current_ones_iou = metric(pred, target)
        current_zeros_iou = metric(pred_toggle, target_toggle)

        ones_iou.append(current_ones_iou)
        zeros_iou.append(current_zeros_iou)

        print("{f} Ones IOU: {iou:.6f}".format(f=file, iou=current_ones_iou))
        print("{f} Zeros IOU: {iou:.6f}".format(f=file, iou=current_zeros_iou))

    ones_iou_clean = [x for x in ones_iou if not torch.isnan(x)]
    zeros_iou_clean = [x for x in zeros_iou if not torch.isnan(x)]
    return torch.mean(torch.stack(ones_iou_clean)), torch.mean(torch.stack(zeros_iou_clean))


if __name__ == "__main__":
    pred_dir = '/media/bmo/TOSHIBA_2T_HD/Data/test_iou/L1012res_256RGB/'
    target_dir = '/media/bmo/TOSHIBA_2T_HD/Data/test_iou/L1012_GFP/'
    ones_iou_mean, zeros_iou_mean = calculate_iou(pred_dir, target_dir, file_format='.bmp')
    table = PrettyTable()
    table.field_names = ['', 'Mean IOU']
    table.add_row(['Ones', ones_iou_mean.item()])
    table.add_row(['Zeros', zeros_iou_mean.item()])
    print(table)