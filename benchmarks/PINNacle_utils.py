"""
Utility code taken (and slightly modified) from PINNacle to make our benchmark work
"""

import numpy as np


def trans_time_data_to_dataset(ref_data, datapath, input_dim, output_dim):
    data = ref_data
    slice = (data.shape[1] - input_dim + 1) // output_dim
    assert (
        slice * output_dim == data.shape[1] - input_dim + 1
    ), "Data shape is not multiple of pde.output_dim"

    with open(datapath, "r") as f:

        def extract_time(string):
            index = string.find("t=")
            if index == -1:
                return None
            return float(string[index + 2 :].split(" ")[0])

        t = None
        for line in f.readlines():
            if line.startswith("%") and line.count("@") == slice * output_dim:
                t = line.split("@")[1:]
                t = list(map(extract_time, t))
        if t is None or None in t:
            raise ValueError(
                "Reference Data not in Comsol format or does not contain time info"
            )
        t = np.array(t[::output_dim])

    t, x0 = np.meshgrid(t, data[:, 0])
    list_x = [x0.reshape(-1)]
    for i in range(1, input_dim - 1):
        list_x.append(np.stack([data[:, i] for _ in range(slice)]).T.reshape(-1))
    list_x.append(t.reshape(-1))
    for i in range(output_dim):
        list_x.append(data[:, input_dim - 1 + i :: output_dim].reshape(-1))
    return np.stack(list_x).T


def compute_relative_errors(test_y, pred_y):
    mse = ((test_y - pred_y) ** 2).mean()
    mae = np.abs(test_y - pred_y).mean()
    l1re = mae / np.abs(test_y).mean()
    l2re = np.sqrt(mse) / np.sqrt(test_y**2).mean()
    print(f"{l1re=}, {l2re=}")
