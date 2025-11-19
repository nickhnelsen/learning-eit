import os, yaml
import torch
import numpy as np
from models import FNO2d as my_model
from util import plt
from util.sample_random_fields import RandomField
from util.utilities_module import UnitGaussianNormalizer, count_params, dataset_with_indices, set_seed, integrate
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
exp_date = "2025-10-23"
load_prefix = "paper_sweep"
N_train_list = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 9500]
noise = 0
seed = 1

# New eval choices
subfolder = "figures_dbar/"
noise_new = 2
noise_distribution_new = "uniform"
FLAG_BEST = True
PLOT_CLEAN = False
FLAG_LOCAL = not True

save_path_new = "./results/" + subfolder
os.makedirs(save_path_new, exist_ok=True)

load_path = '/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/ntd_samples'
pp = ppoopoo

for N_train in N_train_list:
# Get path
prefix_new = noise_distribution_new + str(noise_new) + "_Best" + str(int(FLAG_BEST)) + "_"
plot_folder_base = "./results/" + exp_date + "/" + load_prefix
load_path = plot_folder_base + "_N" + str(N_train) + "_Noise" + str(noise) + "_Seed" + str(seed) + "/"

# Get config
CONFIG_PATH = save_path + "config.yaml"     # hard coded path
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Set seed
if seed is not None:
    set_seed(seed)

# File I/O
if FLAG_LOCAL:
    data_folder = "/home/nnelsen/data/eit/shape_detection/"
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
FLAG_SHUFFLE = config['FLAG_SHUFFLE']
noise_distribution = config['noise_distribution']

# Checks
assert N_train + N_val + N_test <= N_max
assert sub_in_test <= sub_in and sub_out_test <= sub_out

################################################################
#
# load and process data
#
################################################################

start = default_timer()

x_test3 = torch.load(data_folder + 'kernel_3heart_rhop7.pt', weights_only=True)['kernel_3heart'][...,::sub_in_test,::sub_in_test]
y_test3 = torch.load(data_folder + 'conductivity_3heart_rhop7.pt', weights_only=True)['conductivity_3heart'][...,::sub_out_test,::sub_out_test]
y_test3 = torch.flip(y_test3, [-2])

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

################################################################
#
# evaluate model
#
################################################################

# Phantom three evaluations
with torch.no_grad():
    out3 = model(x_test3.to(device))*mask_test + ~mask_test
    out3 = out3.squeeze().cpu()
    out3_clean = model(x_test3_clean.to(device))*mask_test + ~mask_test
    out3_clean = out3_clean.squeeze().cpu()
out3[:, ~mask_test.cpu()] = float('nan')
out3_clean[:, ~mask_test.cpu()] = float('nan')

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
        ax.imshow(true_test3, interpolation='none', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_frame_on(False)
        
        # --- 3: Predicted conductivity ---
        ax = axs[2]
        ax.set_title('Predicted conductivity', fontsize=fz)
        ax.imshow(plot_test3, interpolation='none', vmin=vmin, vmax=vmax, origin='lower')
        ax.set_frame_on(False)
        
        # --- 4: Pointwise error ---
        ax = axs[3]
        ax.set_title('Pointwise error', fontsize=fz)
        im = ax.imshow(er_test3, cmap='inferno', origin='lower')
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
        # cond_ticks = [0, 1, 2, 3, 4,5]   # adjust to your range
        # cbar.set_ticks(cond_ticks)
        # cbar.set_ticklabels([str(t) for t in cond_ticks])
        cbar.ax.tick_params(labelsize=fz)
            
        plt.savefig(plot_folder + prefix_new + name + str(i) + ".png", format='png', dpi=300, bbox_inches='tight')

OOD_plot(out3, x_test3, "heartNlungs_")
if PLOT_CLEAN:
    OOD_plot(out3_clean, x_test3_clean, "heartNlungs_clean_")