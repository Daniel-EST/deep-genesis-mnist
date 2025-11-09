import argparse
import json

from utils import flatten, get_shape, write_array


def generate(json_path: str, out_c_path: str) -> None:
    print(f"Loading weights from {json_path}...")
    with open(json_path, "r") as fp:
        state_dict = json.load(fp)

    with open(out_c_path, "w") as f:
        f.write("/* Auto-generated from torch_weights.json */\n")
        f.write("#ifndef _WEIGHTS_H_\n")
        f.write("#define _WEIGHTS_H_\n\n")
        f.write("#include <genesis.h>\n\n")
        f.write("#define FC1_IN 784\n")
        f.write("#define FC1_OUT 40\n")
        f.write("#define FC2_IN FC1_OUT\n")
        f.write("#define FC2_OUT 40\n")
        f.write("#define FC3_IN FC2_OUT\n")
        f.write("#define FC3_OUT 10\n\n")

        fc1_w = state_dict["fc1.weight"]
        shape_fc1_w = get_shape(fc1_w)
        fc1_w_flat = list(flatten(fc1_w))
        write_array(f, "fc1_weight_data", fc1_w_flat, shape_fc1_w)

        fc1_b = state_dict["fc1.bias"]
        shape_fc1_b = get_shape(fc1_b)
        fc1_b_flat = list(flatten(fc1_b))
        write_array(f, "fc1_bias_data", fc1_b_flat, shape_fc1_b)

        fc2_w = state_dict["fc2.weight"]
        shape_fc2_w = get_shape(fc2_w)
        fc2_w_flat = list(flatten(fc2_w))
        write_array(f, "fc2_weight_data", fc2_w_flat, shape_fc2_w)

        fc2_b = state_dict["fc2.bias"]
        shape_fc2_b = get_shape(fc2_b)
        fc2_b_flat = list(flatten(fc2_b))
        write_array(f, "fc2_bias_data", fc2_b_flat, shape_fc2_b)

        fc3_w = state_dict["fc3.weight"]
        shape_fc3_w = get_shape(fc3_w)
        fc3_w_flat = list(flatten(fc3_w))
        write_array(f, "fc3_weight_data", fc3_w_flat, shape_fc3_w)

        fc3_b = state_dict["fc3.bias"]
        shape_fc3_b = get_shape(fc3_b)
        fc3_b_flat = list(flatten(fc3_b))
        write_array(f, "fc3_bias_data", fc3_b_flat, shape_fc3_b)

        f.write("#endif")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="JSON weights converter",
        description="Converts Pytorch generated weights to C header file",
    )
    parser.add_argument(
        "-i",
        "--input",
        default="./data/models/torch_weights.json",
        help="Path to JSON weights file",
    )
    parser.add_argument(
        "-o", "--output", default="./inc/weights.h", help="Output C header file path"
    )

    args = parser.parse_args()
    generate(args.input, args.output)
