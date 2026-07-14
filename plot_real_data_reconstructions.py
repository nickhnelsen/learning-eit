import os
import yaml

import numpy as np
import torch

from models import FNO2d as my_model
from models.shared import resize_rfft2
from util import plt
from util.utilities_module import UnitGaussianNormalizer


torch.set_printoptions(precision=16)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device is", device)


################################################################
#
# USER INPUT
#
################################################################

FLAG_BEST = True
FLAG_SAVE_PDF = not True
FLAG_SHUFFLE_FROM_CHECKPOINT = True
REAL_NORMALIZER = True

N_train = 9500
Noise_train = 10
MODEL_SEED = 0

NORMALIZER_MODES = 16

REAL_KERNEL_PATH = "./data_real/kernel_KIT4.pt"
REAL_KERNEL_KEY = "kernel"

PHOTO_FOLDER = "./data_real/conductivity_KIT4"
SAVE_FOLDER = "./results/real_data_reconstructions"

# Number of realizations in experiments k=1,...,8.
EXP_NUM = [4, 6, 6, 4, 2, 7, 2, 6]

DATASETS = {
    "shape": {
        "exp_date": "2025-10-23",
        "load_prefix": "paper_sweep",
        "ds_name": "shape",
    },
    "three_phase": {
        "exp_date": "2025-11-05",
        "load_prefix": "paper_sweep_three_phase",
        "ds_name": "three_phase",
    },
    "lognormal": {
        "exp_date": "2025-11-10",
        "load_prefix": "paper_sweep_lognormal",
        "ds_name": "lognormal",
    },
}

os.makedirs(SAVE_FOLDER, exist_ok=True)


################################################################
#
# PLOTTING STYLE
#
################################################################

plt.close("all")
plt.rcParams["figure.dpi"] = 250
plt.rcParams["savefig.dpi"] = 250
plt.rcParams["font.size"] = 16


################################################################
#
# I/O AND MODEL UTILITIES
#
################################################################


def _load_tensor(path, key=None, weights_only=True):
    """Load either a tensor or a dictionary containing a tensor."""
    if path.endswith(".npy"):
        return torch.from_numpy(np.load(path))

    obj = torch.load(path, weights_only=weights_only)
    if isinstance(obj, dict):
        if key is None or key not in obj:
            raise KeyError(
                f"Key '{key}' not found in {path}. Available keys: {list(obj.keys())}"
            )
        obj = obj[key]

    return torch.as_tensor(obj)


def _as_batch(x):
    if x.ndim == 2:
        x = x.unsqueeze(0)
    return x.contiguous()


def checkpoint_folder(exp_date, load_prefix, noise, seed):
    return os.path.join(
        "./results",
        exp_date,
        f"{load_prefix}_N{N_train}_Noise{noise}_Seed{seed}",
    )


def read_config(load_path):
    with open(os.path.join(load_path, "config.yaml"), "r") as f:
        return yaml.safe_load(f)


def load_model_from_config(config, load_path):
    model = my_model(
        modes1=config["modes1"],
        modes2=config["modes2"],
        width=config["width"],
        width_final=config["width_final"],
        act=config["act"],
        n_layers=config["n_layers"],
    ).to(device)

    model_file = "model_best.pt" if FLAG_BEST else "model_last.pt"
    model.load_state_dict(
        torch.load(os.path.join(load_path, model_file), weights_only=True)
    )
    model.eval()
    return model


def training_data_folder(config, ds_cfg):
    if ds_cfg["ds_name"] == "shape":
        return os.path.join(config["data_folder"], ds_cfg["ds_name"])
    return config["data_folder"]


def load_train_kernel_for_normalizer(config, ds_cfg, load_path):
    data_folder = training_data_folder(config, ds_cfg)
    kernel_file = ds_cfg.get("train_kernel_file", "kernel.pt")
    kernel_key = ds_cfg.get("train_kernel_key", "kernel")

    sub_in_test = 1 if ds_cfg["ds_name"] == "shape" else config["sub_in_test"]
    N_val = config["N_val"]
    N_test = config["N_test"]

    x_train = _as_batch(
        _load_tensor(os.path.join(data_folder, kernel_file), kernel_key)
    )
    x_train = x_train[..., ::sub_in_test, ::sub_in_test]
    x_train = x_train[: -(N_val + N_test), ...]

    if config.get("FLAG_SHUFFLE", False) and FLAG_SHUFFLE_FROM_CHECKPOINT:
        shuffle_idx = torch.load(
            os.path.join(load_path, "idx_shuffle.pt"), weights_only=True
        )
        x_train = x_train[shuffle_idx]

    return x_train[:N_train].contiguous()


def load_output_mask(config, ds_cfg, output_shape):
    data_folder = training_data_folder(config, ds_cfg)
    mask_folder = ds_cfg.get("mask_data_folder", data_folder)
    mask_file = ds_cfg.get("mask_file", "mask.pt")
    mask_key = ds_cfg.get("mask_key", "mask")
    mask_path = os.path.join(mask_folder, mask_file)

    if not os.path.exists(mask_path):
        print(f"WARNING: {mask_path} not found; using the full square as mask.")
        return torch.ones(output_shape, dtype=torch.bool)

    mask = _load_tensor(mask_path, mask_key).to(torch.bool)
    sub_out_test = config["sub_out_test"]
    mask = mask[::sub_out_test, ::sub_out_test]

    if tuple(mask.shape) != tuple(output_shape):
        raise ValueError(
            f"Mask has shape {tuple(mask.shape)}, but model output has shape "
            f"{tuple(output_shape)}."
        )
    return mask.contiguous()


def load_real_kernels():
    x_real = _as_batch(_load_tensor(REAL_KERNEL_PATH, REAL_KERNEL_KEY)).float()

    if x_real.shape[0] != sum(EXP_NUM):
        raise ValueError(
            f"Expected {sum(EXP_NUM)} real kernels, but loaded {x_real.shape[0]} "
            f"from {REAL_KERNEL_PATH}."
        )
    if tuple(x_real.shape[-2:]) != (256, 256):
        raise ValueError(
            f"Expected real-kernel shape (37, 256, 256), got {tuple(x_real.shape)}."
        )

    return x_real.contiguous()


################################################################
#
# FOURIER NORMALIZATION
#
################################################################


def projection_fourier(x, num_modes):
    """
    Project a real batch x onto the central num_modes x num_modes 2D Fourier block, and pads back to physical space

    x: (..., J1, J2), real tensor
    """
    s = x.shape[-2], x.shape[-1]                            # (J1, J2)
    xhat = torch.fft.rfft2(x, norm="forward")               # (J1, J2//2+1)
    xhat_low = resize_rfft2(xhat, (num_modes, num_modes))   # (M, M//2+1)
    xhat_pad = resize_rfft2(xhat_low, s)                    # (J1, J2//2+1)
    
    return torch.fft.irfft2(xhat_pad, s=s, norm="forward")  # (J1, J2)


################################################################
#
# REAL-DATA INFERENCE AND PLOTTING
#
################################################################


def phantom_labels():
    """Return [(1,1),...,(1,4),(2,1),...,(8,6)] in kernel order."""
    labels = []
    for expidx, count in enumerate(EXP_NUM, start=1):
        for realidx in range(1, count + 1):
            labels.append((expidx, realidx))
    return labels


def predict_real_kernels(model, x_normalizer, x_real):
    x_in = x_normalizer.encode(x_real).unsqueeze(1).to(device)

    with torch.no_grad():
        pred = model(x_in)

    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0]
    if pred.ndim != 3:
        raise ValueError(f"Unexpected model output shape: {tuple(pred.shape)}")

    return pred.detach().cpu(), x_in.detach().cpu()


def save_real_data_plots(dataset_name, x_real, pred, mask):
    dataset_folder = os.path.join(SAVE_FOLDER, dataset_name)
    os.makedirs(dataset_folder, exist_ok=True)

    pred = torch.where(mask.unsqueeze(0), pred, torch.ones_like(pred))
   
    # Handle either (B, H, W) or (B, 1, H, W) outputs.
    if pred.ndim == 4 and pred.shape[1] == 1:
        pred = pred[:, 0, :, :]

    pred = pred * mask + (~mask)
    
    pred[..., ~mask.cpu()] = float('nan')

    for i, (expidx, realidx) in enumerate(phantom_labels()):
        photo_path = os.path.join(
            PHOTO_FOLDER, f"fantom_{expidx}_{realidx}.jpg"
        )
        if not os.path.exists(photo_path):
            raise FileNotFoundError(f"Target photo not found: {photo_path}")
        target_photo = plt.imread(photo_path)

        kernel_i = x_real[i].cpu()
        pred_i = pred[i].cpu()

        fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4), constrained_layout=True)

        im0 = axs[0].imshow(kernel_i, origin="lower")
        axs[0].set_title("Real NtD Kernel (Normalized)")
        axs[0].axis("off")
        fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)
        
        # Possible orientation adjustments:
        # pred_plot = pred_i                        # no array transform
        # pred_plot = np.flipud(pred_i)             # vertical mirror
        # pred_plot = np.fliplr(pred_i)             # horizontal mirror
        # pred_plot = pred_i.T                      # interchange x and y
        # pred_plot = np.rot90(pred_i, 1)           # 90 degrees counterclockwise
        # pred_plot = np.rot90(pred_i, -1)          # 90 degrees clockwise
        pred_plot = torch.flip(
            pred_i.transpose(-2, -1),
            dims=(-2, -1)
            )                                       # flip bottom right to top left

        im1 = axs[1].imshow(pred_plot, origin="upper")
        axs[1].set_title(f"{dataset_name.replace('_', ' ').title()}, FNO")
        axs[1].axis("off")
        fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

        axs[2].imshow(target_photo, origin="upper")
        axs[2].set_title("Target")
        axs[2].axis("off")

        fig.suptitle(f"KIT4 Phantom {expidx}-{realidx}")

        stem = (
            f"{dataset_name}_fantom_{expidx}_{realidx}"
            f"_Ntrain{N_train}_Noisetrain{Noise_train}_Seed{MODEL_SEED}"
        )
        png_path = os.path.join(dataset_folder, stem + ".png")
        fig.savefig(png_path, dpi=300)

        if FLAG_SAVE_PDF:
            fig.savefig(os.path.join(dataset_folder, stem + ".pdf"))

        plt.close(fig)
        print("Saved", png_path)


def compute_dataset(dataset_name, ds_cfg, x_real, real_norm=REAL_NORMALIZER):
    print("\n" + "=" * 70)
    print("Training dataset:", dataset_name)
    print("=" * 70)

    load_path = checkpoint_folder(
        ds_cfg["exp_date"], ds_cfg["load_prefix"], Noise_train, MODEL_SEED
    )
    config = read_config(load_path)

    x_train = load_train_kernel_for_normalizer(config, ds_cfg, load_path)
    if tuple(x_train.shape[-2:]) != tuple(x_real.shape[-2:]):
        raise ValueError(
            f"Training kernels for {dataset_name} have grid {tuple(x_train.shape[-2:])}, "
            f"but real kernels have grid {tuple(x_real.shape[-2:])}."
        )

    if real_norm:
        x_normalizer = UnitGaussianNormalizer(x_real)
    else:
        # Match the normalization used in the M=16 resolution-ablation experiment.
        x_train_low = projection_fourier(x_train, NORMALIZER_MODES)
        x_normalizer = UnitGaussianNormalizer(x_train_low)
        del x_train_low

    model = load_model_from_config(config, load_path)
    pred, x_in = predict_real_kernels(model, x_normalizer, x_real)
    mask = load_output_mask(config, ds_cfg, pred.shape[-2:])

    save_real_data_plots(dataset_name, x_in.squeeze(), pred, mask)

    del model, x_train, x_normalizer, pred
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


if __name__ == "__main__":
    real_kernels = load_real_kernels()

    for dataset_name, ds_cfg in DATASETS.items():
        compute_dataset(dataset_name, ds_cfg, real_kernels)
