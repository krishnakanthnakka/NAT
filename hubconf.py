dependencies = ["torch"]

import torch
import argparse
from collections import OrderedDict
from generator import StableGeneratorResnet


def parse_args(antiburst=False, nv_pca=None, wpca=False, num_pcs=8192):
    parser = argparse.ArgumentParser(
        description="Eval VPR model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset_name", type=str, default="gsv_cities", help="Dataset"
    )

    args = parser.parse_args()

    # Parse image size
    if args.resize:
        if len(args.resize) == 1:
            args.resize = (args.resize[0], args.resize[0])
        elif len(args.resize) == 2:
            args.resize = tuple(args.resize)
        else:
            raise ValueError("Invalid image size, must be int, tuple or None")

        args.resize = tuple(map(int, args.resize))

    return args


def generator(neuron=250, layer=18, source_model="vgg16") -> torch.nn.Module:

    model = StableGeneratorResnet(gen_dropout=0.0, data_dim="high")

    checkpoint = torch.hub.load_state_dict_from_url(
        f"https://github.com/krishnakanthnakka/NAT/releases/download/checkpoint_neuron_250/0_net_G_neuron.{neuron}.pth"
    )

    model_state_dict = model.state_dict()
    model.load_state_dict(checkpoint["model_state_dict"])

    return model
