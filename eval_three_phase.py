import os, yaml
import torch
import numpy as np
from models import FNO2d as my_model
from util import plt
from util.sample_random_fields import RandomField
from util.utilities_module import LpLoss, L0Loss, L0LossClip, DICE, RatioLoss, UnitGaussianNormalizer, count_params, dataset_with_indices, set_seed, integrate
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)
from timeit import default_timer

torch.set_printoptions(precision=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)

################################################################
#
# user configuration
#
################################################################
# Load training results
exp_date = "2025-11-05"
load_prefix = "paper_sweep_three_phase"
N_train = 9500
noise = 10
seed = 0

# New eval choices
subfolder = "figures_eval/"
eval_loss_str_list = ["L1"]
noise_new = 1
noise_distribution_new = "uniform"
FLAG_BEST = True
PLOT_CLEAN = True
FLAG_LOCAL = not True

# Get path
prefix_new = "three_phase_" + noise_distribution_new + str(noise_new) + "_Best" + str(int(FLAG_BEST)) + "_"
plot_folder_base = "./results/" + exp_date + "/" + load_prefix
save_path = plot_folder_base + "_N" + str(N_train) + "_Noise" + str(noise) + "_Seed" + str(seed) + "/"

# Get config
CONFIG_PATH = save_path + "config.yaml"     # hard coded path
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Set seed
if seed is not None:
    set_seed(seed)

# File I/O
if FLAG_LOCAL:
    data_folder = "/home/nnelsen/data/eit/three_phase/"
else:
    data_folder = config['data_folder']

# Sample size
N_val = config['N_val']
N_test = config['N_test']
N_max = config['N_max']

# Resolution subsampling
sub_in = config['sub_in']
sub_out = config['sub_out']
sub_in_test = config['sub_in_test']
sub_out_test = config['sub_out_test']

# FNO
modes1 = config['modes1']
modes2 = config['modes2']
width = config['width']
width_final = config['width_final']
act = config['act']
n_layers = config['n_layers']

# Training, evaluation, and testing
batch_size = config['batch_size']
FLAG_SHUFFLE = config['FLAG_SHUFFLE']
noise_distribution = config['noise_distribution']

# Checks
assert N_train + N_val + N_test <= N_max
assert sub_in_test <= sub_in and sub_out_test <= sub_out
    
valid_losses = {"L1", "L2", "L0", "Ratio", "L0Clip", "DICE"}
for loss in eval_loss_str_list:
    if loss not in valid_losses:
        raise ValueError(f"Invalid value for eval loss: {loss}. Must be one of {sorted(valid_losses)}.")

################################################################
#
# load and process data
#
################################################################
save_path_new = save_path + subfolder
os.makedirs(save_path_new, exist_ok=True)

start = default_timer()

x_test3 = torch.load(data_folder + 'kernel_3heart_rhop7.pt', weights_only=True)['kernel_3heart'][...,::sub_in_test,::sub_in_test]
y_test3 = torch.load(data_folder + 'conductivity_3heart_rhop7.pt', weights_only=True)['conductivity_3heart'][...,::sub_out_test,::sub_out_test]

sub_in_ratio = sub_in//sub_in_test
sub_out_ratio = sub_out//sub_out_test
x_train = torch.load(data_folder + 'kernel.pt', weights_only=True)['kernel'][...,::sub_in_test,::sub_in_test]
y_train = torch.load(data_folder + 'conductivity.pt', weights_only=True)['conductivity'][...,::sub_out_test,::sub_out_test]
mask = torch.load(data_folder + 'mask.pt', weights_only=True)['mask'][::sub_out_test,::sub_out_test]
mask_test = mask.to(device)
mask = mask_test[::sub_out_ratio,::sub_out_ratio].to(device)

# Fix same test data for all experiments
x_test_clean = x_train[-(N_val + N_test):,...]
x_test_clean = x_test_clean[-N_test:,...]
x_test3_clean = x_test3[...]

# Get noisy inputs
def get_noisy(dataset, my_noise=noise, my_noise_distribution=noise_distribution):
    rf = RandomField(dataset.shape[-1], distribution=my_noise_distribution, device=device)
    dataset_noisy = rf.generate_noise_dataset(dataset.shape[0])
    dataset_noisy = (my_noise/100)*(integrate(dataset**2).sqrt()[:,None,None])*dataset_noisy
    dataset_noisy = dataset + dataset_noisy
    return dataset_noisy

if noise > 0.0:
    x_train = get_noisy(x_train, noise, noise_distribution)

if noise_new > 0:
    x_test3 = get_noisy(x_test3, noise_new, noise_distribution_new)

x_test = get_noisy(x_test_clean, noise_new, noise_distribution_new)
x_train = x_train[:-(N_val + N_test),...]

y_test = y_train[-(N_val + N_test):,...]
y_test = y_test[-N_test:,...]
y_train = y_train[:-(N_val + N_test),...]

# Shuffle training set selection
if FLAG_SHUFFLE:
    dataset_shuffle_idx = torch.load(save_path + 'idx_shuffle.pt', weights_only=True)
    x_train = x_train[dataset_shuffle_idx,...]
    y_train = y_train[dataset_shuffle_idx,...]
else:
    dataset_shuffle_idx = torch.arange(x_train.shape[0])
    
x_train = x_train[:N_train,...]
y_train = y_train[:N_train,::sub_out_ratio,::sub_out_ratio]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)[:,::sub_in_ratio,::sub_in_ratio]
x_test = x_normalizer.encode(x_test)
x_test_clean = x_normalizer.encode(x_test_clean)
x_test3 = x_normalizer.encode(x_test3)
x_test3_clean = x_normalizer.encode(x_test3_clean)

# Make the singleton channel dimension match the FNO2D model input shape requirement
x_train = torch.unsqueeze(x_train, 1)
x_test = torch.unsqueeze(x_test, 1)
x_test_clean = torch.unsqueeze(x_test_clean, 1)
x_test3 = torch.unsqueeze(x_test3, 1)
x_test3_clean = torch.unsqueeze(x_test3_clean, 1)

print("Total time for data processing is", (default_timer()-start), "sec.")

################################################################
#
# load model
#
################################################################
model = my_model(modes1=modes1,
                 modes2=modes2,
                 width=width,
                 width_final=width_final,
                 act=act,
                 n_layers=n_layers
                 ).to(device)

if FLAG_BEST:
    print("Evaluating the best model.")
    model.load_state_dict(torch.load(save_path + 'model_best.pt', weights_only=True))
else:
    print("Evaluating the final epoch model.")
    model.load_state_dict(torch.load(save_path + 'model_last.pt', weights_only=True))
print(model)
model.eval()
print("FNO parameter count:", count_params(model))

# Set loss and minibatch reduction type
num_eval_losses = len(eval_loss_str_list)

################################################################
#
# evaluation on train and test sets
#
################################################################
test_loader = DataLoader(TensorDatasetID(x_test, y_test), batch_size=batch_size, shuffle=False)
test_clean_loader = DataLoader(TensorDatasetID(x_test_clean, y_test), batch_size=batch_size, shuffle=False)

# Assumes sum reduction for exact loss calculations
loss_dict = {"L1": LpLoss(p=1, size_average=False), 
             "L2": LpLoss(p=2, size_average=False),
             "L0": L0Loss(size_average=False),
             "Ratio": RatioLoss(size_average=False),
             "L0Clip": L0LossClip(size_average=False),
             "DICE": DICE(size_average=False)
}
loss_vec_dict = {"L1": LpLoss(p=1, size_average=False, reduction=False), 
             "L2": LpLoss(p=2, size_average=False, reduction=False),
             "L0": L0Loss(size_average=False, reduction=False),
             "Ratio": RatioLoss(size_average=False, reduction=False),
             "L0Clip": L0LossClip(size_average=False, reduction=False),
             "DICE": DICE(size_average=False, reduction=False)
}


def evaluate_my_loader(loader, y_data, mask, type="Test"):
    t1 = default_timer()
    errors = torch.zeros(num_eval_losses)
    out_array = torch.zeros(y_data.shape)
    errors_vec = torch.zeros(y_data.shape[0], num_eval_losses)
    with torch.no_grad():
        for x, y, idx in loader:
            x, y = x.to(device), y.to(device)
    
            out = model(x)*mask + ~mask # set model to one outside unit disk of radius 1
    
            for i, my_str in enumerate(eval_loss_str_list):
                errors[i] += loss_dict[my_str](out, y).item()
                errors_vec[idx, i] = loss_vec_dict[my_str](out, y).cpu()
            
            out_array[idx,...] = out.squeeze().cpu()
    
    errors /= y_data.shape[0]
    t2 = default_timer()
    combined_dict = dict(zip(eval_loss_str_list, errors))
    print(f'Eval Time (sec): {t2-t1}, ' + type + " " + ", ".join(f"{k}: {v:.4f}" for k,v in combined_dict.items()))
    
    return errors, out_array, errors_vec
    
    
errors_test, out_test, errors_test_vec = evaluate_my_loader(test_loader, y_test, mask_test, "Test")
errors_test_clean, out_test_clean, errors_test_clean_vec = evaluate_my_loader(test_clean_loader, y_test, mask_test, "Test (Clean)")

# Save final test errors
torch.save(errors_test, save_path_new + prefix_new + 'errors_test.pt')
torch.save(errors_test_clean, save_path_new + prefix_new + 'errors_test_clean.pt')
torch.save(errors_test_vec, save_path_new + prefix_new + 'errors_test_vec.pt')
torch.save(errors_test_clean_vec, save_path_new + prefix_new + 'errors_test_clean_vec.pt')

################################################################
#
# plotting
#
################################################################
plot_folder = save_path_new

plt.close("all")

plt.rcParams['figure.dpi'] = 250
plt.rcParams['savefig.dpi'] = 250
plt.rcParams['font.size'] = 18
plt.rc('legend', fontsize=15)
plt.rcParams['lines.linewidth'] = 3.5
msz = 14
fz = 14
handlelength = 3.0     # 2.75
borderpad = 0.25     # 0.15

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

color_list = ['k', 'C3', 'C5', 'C1', 'C2', 'C0', 'C4', 'C6', 'C7', 'C8', 'C9'] # black, red, brown, orange, green, blue, purple, pink, gray, olive, cyan

out_test[:, ~mask_test.cpu()] = float('nan')
out_test_clean[:, ~mask_test.cpu()] = float('nan')

# Phantom three evaluations
with torch.no_grad():
    out3 = model(x_test3.to(device))*mask_test + ~mask_test
    out3 = out3.squeeze().cpu()
    out3_clean = model(x_test3_clean.to(device))*mask_test + ~mask_test
    out3_clean = out3_clean.squeeze().cpu()
out3[:, ~mask_test.cpu()] = float('nan')
out3_clean[:, ~mask_test.cpu()] = float('nan')

# Tile plot
def tile_plot(errors_vec, x_data, y_data, mask, out_data, plotname="test"):
    names = ['Worst', 'Median', 'Best', 'Random']  # column titles
    K = len(names)
    
    # indices: worst, median, best, random
    idx_worst  = torch.argsort(errors_vec, dim=0)[-1, ...]
    idx_median = torch.argsort(errors_vec, dim=0)[errors_vec.shape[0] // 2, ...]
    idx_best   = torch.argsort(errors_vec, dim=0)[0, ...]
    idx_rand = torch.randint(N_test, [len(idx_best)])

    idxs = [idx_worst, idx_median, idx_best, idx_rand]
    for ii, iii in enumerate(idxs):
        print(names[ii] + ":", errors_vec[iii, torch.arange(num_eval_losses)])

    for loop in range(num_eval_losses):
        true_fields = []
        pred_fields = []
        err_fields  = []
        input_fields = []   
        
        loss = eval_loss_str_list[loop]
        for m, name in enumerate(names):
            idx = idxs[m][loop].item()
            
            input_trainsort = x_data[idx, ...].squeeze().clone()
        
            # true (masked)
            true_trainsort = y_data[idx, ...].squeeze().clone()
            true_trainsort[~mask.cpu()] = float('nan')
        
            # prediction
            plot_trainsort = out_data[idx, ...].squeeze().clone()
        
            # error
            er_trainsort = torch.abs(plot_trainsort - true_trainsort).squeeze()
        
            # convert to numpy
            input_np = input_trainsort.detach().cpu().numpy()
            true_np  = true_trainsort.detach().cpu().numpy()
            pred_np  = plot_trainsort.detach().cpu().numpy()
            err_np   = er_trainsort.detach().cpu().numpy()
        
            input_fields.append(input_np)
            true_fields.append(true_np)
            pred_fields.append(pred_np)
            err_fields.append(err_np)
        
        # ------------------------------------------------------------------
        # Global color limits
        # ------------------------------------------------------------------
        # all_stress = np.stack(true_fields + pred_fields, axis=0)
        all_stress = np.stack(true_fields[1:], axis=0)
        vmin_stress = float(np.nanmin(all_stress))
        vmax_stress = float(np.nanmax(all_stress))
        if not (vmax_stress > vmin_stress):
            vmin_stress, vmax_stress = vmin_stress - 1e-12, vmax_stress + 1e-12
        vmin_stress = max(0.0, vmin_stress)
            
        all_inputs = np.stack(input_fields[1:], axis=0)
        vmin_input = float(np.nanmin(all_inputs))
        vmax_input = float(np.nanmax(all_inputs))
        if not (vmax_input > vmin_input):
            vmin_input, vmax_input = vmin_input - 1e-12, vmax_input + 1e-12
        
        all_err = np.stack(err_fields[1:], axis=0)
        vmin_err = float(np.nanmin(all_err))
        vmax_err = float(np.nanmax(all_err))
        if not (vmax_err > vmin_err):
            vmin_err, vmax_err = vmin_err - 1e-12, vmax_err + 1e-12
        
        # ------------------------------------------------------------------
        # Create 4 x K grid: rows = {input, true, pred, error}
        # ------------------------------------------------------------------
        fig, axes = plt.subplots(
            4, K, figsize=(10, 8),
            sharex=False, sharey=False,
            constrained_layout=False
        )
        axes = np.atleast_2d(axes)
    
        # references for colorbars (use last column’s images)
        im_input_ref = im_true_ref = im_err_ref = None
        
        # ------------------------------------------------------------------
        # Fill the grid
        # ------------------------------------------------------------------
        for j, name in enumerate(names):
            ax_input = axes[0, j]
            ax_true  = axes[1, j]
            ax_pred  = axes[2, j]
            ax_err   = axes[3, j]
            
            ax_input.set_title(name, fontsize=fz, pad=4)
    
            # First row: NtD kernel
            im_input = ax_input.imshow(
                input_fields[j], origin='lower',
                vmin=vmin_input, vmax=vmax_input
            )
    
            # True / predicted conductivity (default viridis)
            im_true = ax_true.imshow(
                true_fields[j], origin='lower',
                interpolation='none',
                vmin=vmin_stress, vmax=vmax_stress
            )
            ax_true.set_frame_on(False)
    
            _ = ax_pred.imshow(
                pred_fields[j], origin='lower',
                interpolation='none',
                vmin=vmin_stress, vmax=vmax_stress
            )
            ax_pred.set_frame_on(False)
    
            im_err = ax_err.imshow(
                err_fields[j], origin='lower',
                vmin=vmin_err, vmax=vmax_err,
                cmap='inferno'
            )
            ax_err.set_frame_on(False)
    
            # Remove ticks
            for ax in (ax_input, ax_true, ax_pred, ax_err):
                ax.set_xticks([])
                ax.set_yticks([])
    
            # Row labels on the leftmost column (vertical)
            if j == 0:
                ax_input.set_ylabel('Noisy NtD kernel', rotation=90,
                                    ha='center', va='center', labelpad=10, fontsize=fz)
                ax_true.set_ylabel('True conductivity', rotation=90,
                                   ha='center', va='center', labelpad=10, fontsize=fz)
                ax_pred.set_ylabel('Predicted conductivity', rotation=90,
                                   ha='center', va='center', labelpad=10, fontsize=fz)
                ax_err.set_ylabel('Pointwise error', rotation=90,
                                  ha='center', va='center', labelpad=10, fontsize=fz)
    
            # Save references from last column for colorbars
            if j == K - 1:
                im_input_ref = im_input
                im_true_ref  = im_true
                im_err_ref   = im_err
    
        # Reserve some space on the right for the colorbars
        fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])
        
        # 1) NtD kernel colorbar (first row, inferno)
        cin_axes = axes[0, :].ravel()
        cbar_in = fig.colorbar(
            im_input_ref,                 # keep a ref to im_input from the last column
            ax=cin_axes,
            location='right',             # matplotlib >= 3.3
            fraction=0.03,
            pad=0.02
        )
        cbar_in.ax.tick_params(labelsize=fz)
        
        # 2) Conductivity colorbar (rows 2–3)
        stress_axes = axes[1:3, :].ravel()
        cbar_stress = fig.colorbar(
            im_true_ref,                  # ref to true conductivity imshow
            ax=stress_axes,
            location='right',
            fraction=0.03,
            pad=0.02
        )
        # --- conductivity (row 3) ---
        cbar_stress.ax.tick_params(labelsize=fz)
    
        # 3) Error colorbar (last row, cividis)
        err_axes = axes[3, :].ravel()
        cbar_err = fig.colorbar(
            im_err_ref,
            ax=err_axes,
            location='right',
            fraction=0.03,
            pad=0.02
        )
        cbar_err.ax.tick_params(labelsize=fz)
        
        # IMPORTANT: do NOT call plt.tight_layout() again after this
        plt.savefig(plot_folder + prefix_new + loss + "_" + plotname + ".png", format='png', dpi=300, bbox_inches='tight')

tile_plot(errors_test_vec, x_test, y_test, mask_test, out_test, plotname="test")
if PLOT_CLEAN:
    tile_plot(errors_test_clean_vec, x_test_clean, y_test, mask_test, out_test_clean, plotname="test_clean")


# %% Non-random phantoms of varying contrast
plt.close("all")
plt.rcParams['font.size'] = 11

def OOD_plot(out, x, name):
    for i in range(3):
        true_test3 = y_test3[i,...].squeeze()
        true_test3[~mask_test.cpu()] = float('nan')
        plot_test3 = out[i,...].squeeze()
        er_test3 = torch.abs(plot_test3 - true_test3).squeeze()
        true_test3 = true_test3.detach().cpu().numpy()
        
        vmin = float(np.nanmin(true_test3))
        vmax = float(np.nanmax(true_test3))
        if not (vmax > vmin):
            vmin, vmax = vmin - 1e-12, vmax + 1e-12
        vmin = max(0.0, vmin)
        
        plt.close(100)
        fig, axs = plt.subplots(1, 4, num=100, figsize=(8, 2))
        
        fz = 10  # or whatever you had
        
        # --- 1: Noisy NtD kernel ---
        ax = axs[0]
        ax.set_title('Noisy NtD kernel', fontsize=fz)
        ax.imshow(x[i, ...].squeeze(), origin='lower')
        
        # --- 2: True conductivity ---
        ax = axs[1]
        ax.set_title('True conductivity', fontsize=fz)
        ax.imshow(true_test3, interpolation='none', vmin=vmin, vmax=vmax)
        ax.set_frame_on(False)
        
        # --- 3: Predicted conductivity ---
        ax = axs[2]
        ax.set_title('Predicted conductivity', fontsize=fz)
        ax.imshow(plot_test3, interpolation='none', vmin=vmin, vmax=vmax)
        ax.set_frame_on(False)
        
        # --- 4: Pointwise error ---
        ax = axs[3]
        ax.set_title('Pointwise error', fontsize=fz)
        im = ax.imshow(er_test3, cmap='inferno')
        ax.set_frame_on(False)
        
        # Remove ticks from all subplots
        for ax in axs:
            ax.set_xticks([])
            ax.set_yticks([])
        
        # First tighten the layout but leave room on the right for the colorbar
        fig.tight_layout(rect=[0.0, 0.0, 0.86, 1.0])
        
        # One colorbar for the last image, aligned with all 4 axes
        cbar = fig.colorbar(
            im,
            ax=axs,              # <-- THIS is the key: attach to all 4
            location='right',
            fraction=0.028,
            pad=0.02
        )
        cond_ticks = [0, 1, 2, 3, 4,5]   # adjust to your range
        cbar.set_ticks(cond_ticks)
        cbar.set_ticklabels([str(t) for t in cond_ticks])
        cbar.ax.tick_params(labelsize=fz)
            
        plt.savefig(plot_folder + prefix_new + name + str(i) + ".png", format='png', dpi=300, bbox_inches='tight')

OOD_plot(out3, x_test3, "heartNlungs_")
if PLOT_CLEAN:
    OOD_plot(out3_clean, x_test3_clean, "heartNlungs_clean_")
