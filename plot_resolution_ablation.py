"""
Resolution ablation for EIT FNO models.

This script loads the best N=9500 model for each dataset/noise/seed, projects clean test kernels onto a lower Fourier band, adds noise in the same Fourier band, recomputes average
relative L1 test error, and saves one PDF and PNG per dataset.
"""

import os
import yaml
from timeit import default_timer

import torch
import numpy as np
from models import FNO2d as my_model
from util import plt
from util.utilities_module import UnitGaussianNormalizer, set_seed, integrate, LpLoss, dataset_with_indices
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)
from util.sample_random_fields import RandomField
from models.shared import resize_rfft2


torch.set_printoptions(precision=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)


################################################################
#
# USER INPUT
#
################################################################

FLAG_SAVE_PLOTS = True
FLAG_SAVE_ERRORS = True
FLAG_BEST = True
FLAG_SHUFFLE_FROM_CHECKPOINT = True

# Best models are loaded from
#   ./results/{exp_date}/{load_prefix}_N{N_train}_Noise{noise}_Seed{seed}/model_best.pt
N_train = 9500
Noise_train = 30
Noise_list = [0, 3, 10, 30]
Seed_list = [0, 1, 2, 3, 4]
Fourier_modes_list = [256, 128, 64, 32, 16, 8, 4]

# Plot uncertainty: mean +/- n_std * std over Seed_list.
n_std = 2
plot_tol = 1e-8
INVERT_X_AXIS = not True

TEST_BATCH_SIZE = 64
my_test_distribution = 'uniform'
FLAG_NORMALIZER_ON_NOISY_TRAIN = True


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

save_folder = "./results/resolution_ablation_viz/"
os.makedirs(save_folder, exist_ok=True)

FLAG_SAVE_MODE_EXAMPLES = True
PLOT_EXAMPLE_SEED = 0
PLOT_EXAMPLE_INDEX = None           # None chooses one random test sample
PLOT_EXAMPLE_RANDOM_SEED = 1234

################################################################
#
# Plotting style copied from plot_sweep_data.py
#
################################################################

plt.close("all")
plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
plt.rcParams['font.size'] = 18
plt.rc('legend', fontsize=15)
plt.rcParams['lines.linewidth'] = 3.5
plt.rcParams['figure.figsize'] = [6.0, 6.0]
msz = 14
handlelength = 3.0
borderpad = 0.25

linestyle_tuples = {
     'solid':                 '-',
     'dashdot':               '-.',
     'loosely dotted':        (0, (1, 10)),
     'dotted':                (0, (1, 1)),
     'densely dotted':        (0, (1, 1)),
     'long dash with offset': (5, (10, 3)),
     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),
     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),
     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}

marker_list = ['o', 'd', 's', 'v', 'X', "*", "P", "^"]
style_list = ['-.', linestyle_tuples['dotted'], linestyle_tuples['densely dashdotted'],
              linestyle_tuples['densely dashed'], linestyle_tuples['densely dashdotdotted']]
color_list = ['k', 'C3', 'C5', 'C1', 'C2', 'C0', 'C4', 'C6', 'C7', 'C8', 'C9']
legs = [r"$0\%$", r"$3\%$", r"$10\%$", r"$30\%$"]


def get_stats(ar):
    """Return mean and std over axis 0, like plot_sweep_data.py."""
    out = np.zeros((*ar.shape[-(ar.ndim - 1):], 2))
    out[..., 0] = np.mean(ar, axis=0)
    out[..., 1] = np.std(ar, axis=0)
    return out


################################################################
#
# I/O and data utilities
#
################################################################

def _load_tensor(folder, filename, key, weights_only=True):
    path = os.path.join(folder, filename)
    obj = torch.load(path, weights_only=weights_only)
    if isinstance(obj, dict):
        if key not in obj:
            raise KeyError(f"Key '{key}' not found in {path}. Available keys: {list(obj.keys())}")
        obj = obj[key]
    return obj


def _as_batch(x):
    if x.ndim == 2:
        x = x.unsqueeze(0)
    return x.contiguous()


def checkpoint_folder(exp_date, load_prefix, noise, seed):
    return "./results/" + exp_date + "/" + load_prefix + "_N" + str(N_train) + \
           "_Noise" + str(noise) + "_Seed" + str(seed) + "/"


def read_config(load_path):
    config_path = os.path.join(load_path, "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_model_from_config(config, load_path):
    model = my_model(modes1=config['modes1'],
                     modes2=config['modes2'],
                     width=config['width'],
                     width_final=config['width_final'],
                     act=config['act'],
                     n_layers=config['n_layers']).to(device)
    model_file = 'model_best.pt' if FLAG_BEST else 'model_last.pt'
    model.load_state_dict(torch.load(os.path.join(load_path, model_file), weights_only=True))
    model.eval()
    return model


def get_noisy(dataset, my_noise, my_noise_distribution, device=device):
    if my_noise == 0:
        return dataset
    rf = RandomField(dataset.shape[-1], distribution=my_noise_distribution, device=device)
    dataset_noisy = rf.generate_noise_dataset(dataset.shape[0])
    dataset_noisy = (my_noise/100)*(integrate(dataset**2).sqrt()[:,None,None])*dataset_noisy
    dataset_noisy = dataset + dataset_noisy
    return dataset_noisy


def get_noise_only(dataset, my_noise, my_noise_distribution, device=device):
    if my_noise == 0:
        return torch.zeros_like(dataset)
    rf = RandomField(dataset.shape[-1], distribution=my_noise_distribution, device=device)
    dataset_noisy = rf.generate_noise_dataset(dataset.shape[0])
    dataset_noisy = (my_noise/100)*(integrate(dataset**2).sqrt()[:,None,None])*dataset_noisy
    return dataset_noisy


def load_train_kernel_for_normalizer(config, ds_cfg, load_path, noise_percent=Noise_train, device=device):           
    if ds_cfg['ds_name'] == "shape":
        data_folder = os.path.join(config['data_folder'], ds_cfg['ds_name'])
    else:
        data_folder = config['data_folder']
    
    kernel_file = ds_cfg.get("train_kernel_file", "kernel.pt")
    kernel_key = ds_cfg.get("train_kernel_key", "kernel")

    if ds_cfg['ds_name'] == "shape":
        sub_in_test = 1
    else:
        sub_in_test = config['sub_in_test']
    N_val = config['N_val']
    N_test = config['N_test']

    x_train = _load_tensor(data_folder, kernel_file, kernel_key)
    x_train = _as_batch(x_train)[..., ::sub_in_test, ::sub_in_test]

    if FLAG_NORMALIZER_ON_NOISY_TRAIN and noise_percent > 0:
        # This is full-grid training noise, not the test Fourier ablation noise.
        distribution = config.get('noise_distribution', 'gaussian')       
        x_train = get_noisy(x_train, noise_percent, distribution, device)
        
    x_train = x_train[:-(N_val + N_test), ...]
    
    if config.get('FLAG_SHUFFLE', False) and FLAG_SHUFFLE_FROM_CHECKPOINT:
        dataset_shuffle_idx = torch.load(os.path.join(load_path, 'idx_shuffle.pt'), weights_only=True)
        x_train = x_train[dataset_shuffle_idx, ...]

    x_train = x_train[:N_train, ...].contiguous()

    return x_train


def load_test_data(config, ds_cfg):
    if ds_cfg['ds_name'] == "shape":
        train_data_folder = os.path.join(config['data_folder'], ds_cfg['ds_name'])
    else:
        train_data_folder = config['data_folder']
    test_data_folder = ds_cfg.get("test_data_folder", train_data_folder)
    mask_data_folder = ds_cfg.get("mask_data_folder", train_data_folder)

    kernel_file = ds_cfg.get("test_kernel_file", ds_cfg.get("train_kernel_file", "kernel.pt"))
    kernel_key = ds_cfg.get("test_kernel_key", ds_cfg.get("train_kernel_key", "kernel"))
    cond_file = ds_cfg.get("test_conductivity_file", "conductivity.pt")
    cond_key = ds_cfg.get("test_conductivity_key", "conductivity")
    mask_file = ds_cfg.get("mask_file", "mask.pt")
    mask_key = ds_cfg.get("mask_key", "mask")

    if ds_cfg['ds_name'] == "shape":
        sub_in_test = 1
    else:
        sub_in_test = config['sub_in_test']
    sub_out_test = config['sub_out_test']
    N_val = config['N_val']
    N_test = config['N_test']

    x_test = _as_batch(_load_tensor(test_data_folder, kernel_file, kernel_key))
    y_test = _as_batch(_load_tensor(test_data_folder, cond_file, cond_key))

    if ds_cfg.get("apply_test_subsampling", True):
        x_test = x_test[..., ::sub_in_test, ::sub_in_test]
        y_test = y_test[..., ::sub_out_test, ::sub_out_test]

    if ds_cfg.get("use_last_test_split", True):
        x_test = x_test[-(N_val + N_test):, ...]
        x_test = x_test[-N_test:, ...]
        y_test = y_test[-(N_val + N_test):, ...]
        y_test = y_test[-N_test:, ...]

    mask_path = os.path.join(mask_data_folder, mask_file)
    if os.path.exists(mask_path):
        mask = _load_tensor(mask_data_folder, mask_file, mask_key)
        if ds_cfg.get("apply_test_subsampling", True):
            mask = mask[::sub_out_test, ::sub_out_test]
        mask = mask.to(torch.bool)
    else:
        print(f"WARNING: {mask_path} not found. Using the full square as the mask.")
        mask = torch.ones(y_test.shape[-2:], dtype=torch.bool)

    return x_test.contiguous(), y_test.contiguous(), mask.contiguous()


################################################################
#
# Fourier projection
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
# Error and evaluation
#
################################################################

def evaluate_my_loader(model, loader, N_test, mask, type="Test", device=device):
    criterion = LpLoss(p=1, size_average=False)
    err = 0.0
    mask = mask.to(device)
    with torch.no_grad():
        for x, y, idx in loader:
            x, y = x.to(device), y.to(device)
    
            out = model(x)*mask + ~mask # set model to one outside unit disk of radius 1
    
            err += criterion(out, y).item()
    
    err /= N_test

    return float(err)


def plot_random_mode_examples(dataset_name,
                              model,
                              x_normalizer,
                              x_test_clean,
                              y_test,
                              mask,
                              modes,
                              noise_percent=0,
                              distribution='uniform',
                              seed=0,
                              random_index=None,
                              plot_seed=1234,
                              N_train=9500,
                              Noise_train=0,
                              device=device):
    """
    For one random test conductivity, save one 2x2 figure for each Fourier mode M.

    Top row:
        [mode M NtD kernel used as FNO input] [full resolution NtD kernel]
    Bottom row:
        [FNO prediction from mode M kernel]   [true conductivity]

    If noise_percent > 0, the top-left NtD kernel includes noise projected to mode M,
    matching the ablation evaluation setup.
    """

    model.eval()

    N_test = x_test_clean.shape[0]
    if random_index is None:
        rng = np.random.default_rng(plot_seed)
        random_index = int(rng.integers(0, N_test))

    example_folder = os.path.join(save_folder, "mode_examples", dataset_name)
    os.makedirs(example_folder, exist_ok=True)

    x_full = x_test_clean[random_index:random_index+1].contiguous()
    y_true = y_test[random_index].detach().cpu()

    mask_device = mask.to(device)

    for M in modes:
        # Project full-resolution NtD kernel to mode M.
        x_low_clean = projection_fourier(x_full, M)

        # Add noise in the same lower Fourier band, if requested.
        if noise_percent > 0:
            noise = get_noise_only(x_low_clean, noise_percent, distribution, device=x_low_clean.device)
            noise = projection_fourier(noise, M)
            x_mode = x_low_clean + noise
        else:
            x_mode = x_low_clean

        # FNO prediction from the mode M input.
        x_in = x_normalizer.encode(x_mode).unsqueeze(1).to(device)

        with torch.no_grad():
            pred = model(x_in)

            # Handle either (B, H, W) or (B, 1, H, W) outputs.
            if pred.ndim == 4 and pred.shape[1] == 1:
                pred = pred[:, 0, :, :]

            pred = pred * mask_device + (~mask_device)
            pred = pred[0].detach().cpu()

        x_mode_plot = x_normalizer.encode(x_mode)[0].detach().cpu()
        x_full_plot = x_normalizer.encode(x_full)[0].detach().cpu()

        # Shared color scales for meaningful side-by-side comparisons.
        kernel_vmin = min(float(x_mode_plot.min()), float(x_full_plot.min()))
        kernel_vmax = max(float(x_mode_plot.max()), float(x_full_plot.max()))

        cond_vmin = min(float(pred.min()), float(y_true.min()))
        cond_vmax = max(float(pred.max()), float(y_true.max()))

        fig, axs = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)

        im00 = axs[0, 0].imshow(x_mode_plot, origin='lower',
                                vmin=kernel_vmin, vmax=kernel_vmax)
        axs[0, 0].set_title(rf"Mode $M={M}$ NtD kernel")
        axs[0, 0].axis("off")
        fig.colorbar(im00, ax=axs[0, 0], fraction=0.046, pad=0.04)

        im01 = axs[0, 1].imshow(x_full_plot, origin='lower',
                                vmin=kernel_vmin, vmax=kernel_vmax)
        axs[0, 1].set_title("Full resolution NtD kernel")
        axs[0, 1].axis("off")
        fig.colorbar(im01, ax=axs[0, 1], fraction=0.046, pad=0.04)

        im10 = axs[1, 0].imshow(pred, origin='lower',
                                vmin=cond_vmin, vmax=cond_vmax)
        axs[1, 0].set_title(rf"FNO prediction from $M={M}$")
        axs[1, 0].axis("off")
        fig.colorbar(im10, ax=axs[1, 0], fraction=0.046, pad=0.04)

        im11 = axs[1, 1].imshow(y_true, origin='lower',
                                vmin=cond_vmin, vmax=cond_vmax)
        axs[1, 1].set_title("True conductivity")
        axs[1, 1].axis("off")
        fig.colorbar(im11, ax=axs[1, 1], fraction=0.046, pad=0.04)

        fig.suptitle(
            rf"{dataset_name.replace('_', ' ').title()}, "
            rf"test index {random_index}, noise {noise_percent}\%, seed {seed}",
            y=1.02
        )

        if FLAG_SAVE_PLOTS:
            pdf_path = os.path.join(
                example_folder,
                f"{dataset_name}_example_idx{random_index}_M{M}_Noise{noise_percent}_Seed{seed}_Ntrain{N_train}_Noisetrain{Noise_train}.pdf"
            )
            png_path = os.path.join(
                example_folder,
                f"{dataset_name}_example_idx{random_index}_M{M}_Noise{noise_percent}_Seed{seed}_Ntrain{N_train}_Noisetrain{Noise_train}.png"
            )
            plt.savefig(pdf_path, format='pdf')
            plt.savefig(png_path, format='png', dpi=300)
            print("Saved", pdf_path)

        plt.close(fig)


def compute_dataset(dataset_name, ds_cfg, device=device):
    print("\n" + "="*70)
    print("Dataset:", dataset_name)
    print("="*70)

    exp_date = ds_cfg["exp_date"]
    load_prefix = ds_cfg["load_prefix"]

    errors = np.zeros((len(Seed_list), len(Noise_list), len(Fourier_modes_list)), dtype=np.float64)
    valid_modes = None
    distribution = my_test_distribution

    for k, seed in enumerate(Seed_list):
        load_path = checkpoint_folder(exp_date, load_prefix, Noise_train, seed)
        print(f"\nLoading {dataset_name}: Noise={Noise_train}, Seed={seed}")
        print(load_path)

        config = read_config(load_path)
        set_seed(seed)

        x_train = load_train_kernel_for_normalizer(config, ds_cfg, load_path) # Uses training data noise level
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_test_clean, y_test, mask = load_test_data(config, ds_cfg)
        
        J = x_test_clean.shape[-1]
        N_test = x_test_clean.shape[0]
        
        model = load_model_from_config(config, load_path)
        
        mode_list_here = [m for m in Fourier_modes_list if m <= J]
        if len(mode_list_here) < len(Fourier_modes_list):
            skipped = [m for m in Fourier_modes_list if m > J]
            print(f"WARNING: test grid is {J} x {J}; skipping modes {skipped}.")
        if valid_modes is None:
            valid_modes = mode_list_here
        elif valid_modes != mode_list_here:
            raise RuntimeError("Different seeds/noise levels gave different valid Fourier modes.")
            
        for j, noise_percent in enumerate(Noise_list):
            # Visualization
            if FLAG_SAVE_MODE_EXAMPLES and seed == PLOT_EXAMPLE_SEED:
                plot_random_mode_examples(dataset_name,
                                      model,
                                      x_normalizer,
                                      x_test_clean,
                                      y_test,
                                      mask,
                                      mode_list_here,
                                      noise_percent=noise_percent,
                                      distribution=distribution,
                                      seed=seed,
                                      random_index=PLOT_EXAMPLE_INDEX,
                                      plot_seed=PLOT_EXAMPLE_RANDOM_SEED,
                                      N_train=N_train,
                                      Noise_train=Noise_train,
                                      device=device)
                
            print(f"\nStarting test noise level {noise_percent}")
            start = default_timer()
            for i, M in enumerate(mode_list_here):
                x_low_clean = projection_fourier(x_test_clean, M)

                # different noise realization every loop
                x_eval = get_noise_only(x_low_clean, noise_percent, distribution, device)
                x_eval = projection_fourier(x_eval, M)
                x_eval = x_low_clean + x_eval
                
                x_eval = x_normalizer.encode(x_eval).unsqueeze(1)
                test_loader = DataLoader(TensorDatasetID(x_eval, y_test), batch_size=TEST_BATCH_SIZE, shuffle=False)
                errors[k, j, i] = evaluate_my_loader(model, test_loader, N_test, mask, type="Test", device=device)
                print(f"  modes={M:4d}, err={errors[k, j, i]:.6e}")

            print("Total time to loop modes:", (default_timer() - start), "sec.")

    if valid_modes is None:
        raise RuntimeError("No valid Fourier modes were evaluated.")

    errors = errors[..., :len(valid_modes)]
    stats = get_stats(errors)       # (Noise, Modes, MeanOrStd)

    if FLAG_SAVE_ERRORS:
        out = {
            'dataset_name': dataset_name,
            'errors_seed_noise_modes': torch.tensor(errors),
            'stats_noise_modes_mean_std': torch.tensor(stats),
            'Noise_list': Noise_list,
            'Seed_list': Seed_list,
            'Fourier_modes_list': valid_modes,
            'n_std': n_std,
            'N_train':  N_train,
            'Noise_train': Noise_train,
        }
        torch.save(out, os.path.join(save_folder, f"resolution_ablation_{dataset_name}_Ntrain{N_train}_Noisetrain{Noise_train}.pt"))
        np.save(os.path.join(save_folder, f"resolution_ablation_{dataset_name}_Ntrain{N_train}_Noisetrain{Noise_train}.npy"), errors)

    make_plot(dataset_name, valid_modes, stats)
    return errors, stats, valid_modes


def make_plot(dataset_name, modes, stats):
    """
    stats: (Noise, Modes, MeanOrStd), where last dim 0=mean and 1=std over seeds.
    """
    plt.figure()
    for j, noise_percent in enumerate(Noise_list):
        x = stats[j, :, 0]
        twosigma = n_std * stats[j, :, 1]
        lb = np.maximum(x - twosigma, plot_tol)
        ub = x + twosigma

        plt.loglog(modes, x,
                   ls=style_list[j],
                   color=color_list[j],
                   marker=marker_list[j],
                   markersize=msz,
                   label=legs[j] if j < len(legs) else str(noise_percent) + r"$\%$")
        plt.fill_between(modes, lb, ub, facecolor=color_list[j], alpha=0.125)

    plt.xlabel(r'Test \# of NtD Fourier Modes')
    plt.xticks(modes, [str(m) for m in modes])
    plt.grid(True, which="both")
    if INVERT_X_AXIS:
        plt.gca().invert_xaxis()
        
    mean_all = stats[..., 0]
    positive_vals = mean_all[np.isfinite(mean_all) & (mean_all > 0)]
    plt.ylim(0.5 * positive_vals.min(), 2.0 * positive_vals.max())
        
    if dataset_name == "shape":
        plt.ylabel(r'Average Relative $L^1$ Test Error')
        plt.legend(framealpha=1, loc='best', borderpad=borderpad, handlelength=handlelength).set_draggable(True)
    # plt.title(dataset_name.replace('_', ' ').title())
    plt.tight_layout()

    if FLAG_SAVE_PLOTS:
        pdf_path = os.path.join(save_folder, f"resolution_ablation_{dataset_name}_Ntrain{N_train}_Noisetrain{Noise_train}.pdf")
        png_path = os.path.join(save_folder, f"resolution_ablation_{dataset_name}_Ntrain{N_train}_Noisetrain{Noise_train}.png")
        plt.savefig(pdf_path, format='pdf')
        plt.savefig(png_path, format='png', dpi=300)
        print("Saved", pdf_path)

    # plt.show()


if __name__ == "__main__":
    for dataset_name, ds_cfg in DATASETS.items():
        compute_dataset(dataset_name, ds_cfg)
