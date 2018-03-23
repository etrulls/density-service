import numpy as np
import torch
from argparse import ArgumentParser
from os.path import isfile
import os
from time import time
from datetime import datetime
import tifffile
import h5py
import unet
from collections import namedtuple


def stamp(string):
    return datetime.fromtimestamp(time()).strftime('%H:%M:%S ') + string


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--username", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--dataset-name", type=str, required=True)
    parser.add_argument("--weights", type=str, required=True)
    parser.add_argument("--threads", type=int, default=os.cpu_count())
    # parser.add_argument("--use-cpu", dest="use_cpu", action="store_true")
    parser.add_argument("--input", type=str, required=True)
    parser.set_defaults(use_cpu=False)
    params = parser.parse_args()

    # Number of threads
    torch.set_num_threads(params.threads)

    if not params.use_cpu:
        print(stamp("Running on GPU {}".format(os.environ["CUDA_VISIBLE_DEVICES"])))
    else:
        print(stamp("Running on CPU"))

    # Load data
    ext = params.input.split('.')[-1]
    if ext in ['tif', 'tiff']:
        stack = tifffile.imread(params.input)
    elif ext == 'h5':
        with h5py.File(params.input, 'r') as f:
            stack = f['data'].value.copy()
    else:
        raise RuntimeError('Bad input format')

    # Must be 3D for this
    if stack.ndim > 3:
        stack = np.squeeze(stack)
    if stack.ndim != 3:
        raise RuntimeError("Not a 3D stack?")

    stack = stack.astype(np.float32) / 255.0

    unet_config = unet.UNetConfig(steps=2,
                                  ndims=3,
                                  num_classes=1,
                                  first_layer_channels=64,
                                  num_input_channels=1,
                                  two_sublayers=True)

    # Bit weird but it works
    model = unet.UNetRegressor(unet_config)
    if not params.use_cpu:
        model = model.cuda()
    model.load_state_dict(torch.load(params.weights))

    # Testing mode
    model.eval()

    # Predict
    prediction = unet.predict_in_blocks(model,
                                        stack,
                                        (82, 82, 82),
                                        verbose=False)[0] / 1000.0

    # Save
    root = "{}/workspace/{}/runs/model-{}/data-{}".format(
        os.path.dirname(os.path.realpath(__file__)),
        params.username,
        params.model_name,
        params.dataset_name,
    )
    if not os.path.isdir(root):
        os.makedirs(root)

    print(stamp('Saving results...'))
    out = root + '/output.h5'
    with h5py.File(out, 'w') as f:
        f.create_dataset(name='data', data=np.array(prediction), chunks=True)
    print(stamp('Done!'))
