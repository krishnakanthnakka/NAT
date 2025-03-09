import os
import torch
import json
import argparse

from collections import Counter
from tqdm import tqdm
from torchvision import models
from dataset import ImageDataset
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
from generator import StableGeneratorResnet

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--nat_attacked_neuron",
        type=int,
    )
    args = parser.parse_args()
    return args


def normalize_fn(
    t,
):
    t = t.clone()
    t[:, 0, :, :] = (t[:, 0, :, :] - mean[0]) / std[0]
    t[:, 1, :, :] = (t[:, 1, :, :] - mean[1]) / std[1]
    t[:, 2, :, :] = (t[:, 2, :, :] - mean[2]) / std[2]
    return t


def generate_adv(generator, inputs):
    inputs_1 = inputs.clone()
    assert torch.max(inputs_1) <= 1.0
    assert torch.min(inputs_1) >= 0.0
    eps = 10.0
    adv1 = generator(inputs_1)
    adv1 = torch.min(torch.max(adv1, inputs_1 - eps / 255.0), 
            inputs_1 + eps / 255.0)
    adv1 = torch.clamp(adv1, 0.0, 1.0)
    return adv1


def eval(args):

    model = models.resnet152(pretrained=True).cuda().eval()

    tranforms = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
    )

    dataset = ImageDataset(
        data_dir="/workspace/datasets/imagenet/",
        images_path=f"./data//imagenet_val5k.txt",
        split="",
        convolve_image=False,
        transform=tranforms,
        target_transform=None,
        keep_difficult=False,
        is_train=False,
        data_aug=None,
    )

    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    generator = StableGeneratorResnet(gen_dropout=0.0, data_dim="high").cuda().eval()
    weights = torch.load(f"./checkpoints/0_net_G_neuron={args.nat_attacked_neuron}.pth")
    generator.load_state_dict(weights["model_state_dict"])
    total = 0
    fooled = 0
    correct = 0

    print(f"Running experiment for neuron: {args.nat_attacked_neuron}")

    with torch.no_grad():
        for index, (inputs, labels, img_paths) in enumerate(tqdm(dataloader)):

            inputs = inputs.cuda()
            labels = labels.cuda()

            logits_clean = model(normalize_fn(inputs.clone()))
            pred_labels_clean = logits_clean.argmax(dim=-1)

            correct += (pred_labels_clean == labels).sum().item()

            adv = generate_adv(generator, inputs)

            logits_adv = model(normalize_fn(adv.clone()))
            pred_labels_adv = logits_adv.argmax(dim=-1)

            fooled += (pred_labels_adv != pred_labels_clean).sum().item()
            total += inputs.shape[0]

        print(
            f"Total: {total}, Fooled: {fooled}, Fooling rate: {100*fooled/total:.2f}%"
        )


if __name__ == "__main__":

    args = get_args()
    eval(
        args,
    )
