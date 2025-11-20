import os, yaml
import torch
import numpy as np
from models import FNO2d as my_model
from util import plt
from util.utilities_module import UnitGaussianNormalizer, set_seed, MatReader
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
N_train_list = [32, 128, 512, 2048, 9500]
# N_train_subset = [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 9500]
noise = 0
seed = 1
FLAG_BEST = True
FLAG_TITLE = True
FLAG_OTRUE = True

# New eval choices
subfolder = "figures_dbar/"
pname = "idx0_shape_clean"
kernel_name = "kernel_idx0_shape_clean.pt"
conductivity_name = "conductivity_idx0_shape.pt"
recon_name = "recon_idx0_shape_clean.mat"

# Load data
load_path = '/media/nnelsen/SharedHDD2TB/datasets/eit/dbar/ntd_samples/'
kernel = torch.load(load_path + kernel_name, weights_only=True)['kernel']
conductivity = torch.load(load_path + conductivity_name, weights_only=True)['conductivity']
recon = MatReader(load_path + recon_name, variable_names=['conductivity'])
recon = recon.read_field('conductivity').unsqueeze(0)

# File IO
save_path_new = "./results/" + subfolder
os.makedirs(save_path_new, exist_ok=True)
pred_conductivity = []
if FLAG_OTRUE:
    origin_true = "lower"
else:
    origin_true = None

for N_train in N_train_list:
    # Get path
    prefix_new = "Best" + str(int(FLAG_BEST)) + "_"
    plot_folder_base = "./results/" + exp_date + "/" + load_prefix
    load_path = plot_folder_base + "_N" + str(N_train) + "_Noise" + str(noise) + "_Seed" + str(seed) + "/"
    
    # Get config
    CONFIG_PATH = load_path + "config.yaml"     # hard coded path
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    
    # Set seed
    if seed is not None:
        set_seed(seed)
    
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
    
    # y_test3 = torch.flip(y_test3, [-2])

    sub_in_ratio = sub_in//sub_in_test
    sub_out_ratio = sub_out//sub_out_test
    x_train = torch.load(data_folder + 'kernel.pt', weights_only=True)['kernel'][...,::sub_in_test,::sub_in_test]
    mask = torch.load(data_folder + 'mask.pt', weights_only=True)['mask'][::sub_out_test,::sub_out_test]
    mask = mask.to(device)
    
    x_train = x_train[:-(N_val + N_test),...]
    
    # Shuffle training set selection
    if FLAG_SHUFFLE:
        dataset_shuffle_idx = torch.load(load_path + 'idx_shuffle.pt', weights_only=True)
        x_train = x_train[dataset_shuffle_idx,...]
    else:
        dataset_shuffle_idx = torch.arange(x_train.shape[0])
        
    x_train = x_train[:N_train,...]
    
    x_normalizer = UnitGaussianNormalizer(x_train)
    my_kernel = x_normalizer.encode(kernel)
    
    # Make the singleton channel dimension match the FNO2D model input shape requirement
    my_kernel = torch.unsqueeze(my_kernel, 1)
    
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
        model.load_state_dict(torch.load(load_path + 'model_best.pt', weights_only=True))
    else:
        print("Evaluating the final epoch model.")
        model.load_state_dict(torch.load(load_path + 'model_last.pt', weights_only=True))
    model.eval()
        
    ################################################################
    #
    # evaluate model
    #
    ################################################################
    
    with torch.no_grad():
        out = model(my_kernel.to(device))*mask + ~mask
        out = out.squeeze().cpu()
    out[..., ~mask.cpu()] = float('nan')
    pred_conductivity.append(out.detach().cpu().numpy())

# Store
all_predict = np.stack(pred_conductivity, axis=0)
num_predict = all_predict.shape[0]

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
from matplotlib.lines import Line2D

fz = 18

true_gamma = conductivity.squeeze()
true_gamma[~mask.cpu()] = float('nan')
dbar_gamma = recon.squeeze()
dbar_gamma[~mask.cpu()] = float('nan')

vmin = float(np.nanmin(true_gamma))
vmax = float(np.nanmax(true_gamma))
if not (vmax > vmin):
    vmin, vmax = vmin - 1e-12, vmax + 1e-12
vmin = max(0.0, vmin)

fig, axs = plt.subplots(1, num_predict + 2, num=100, figsize=(16, 2))

for i, out in enumerate(all_predict):
    N = N_train_list[i]
    ax = axs[i]
    if FLAG_TITLE:
        ax.set_title(rf'$N={N}$', fontsize=fz)
    ax.imshow(out, interpolation='none', vmin=vmin, vmax=vmax, origin="lower")
    ax.set_frame_on(False)
    
ax = axs[-2]
if FLAG_TITLE:
    ax.set_title('True', fontsize=fz)
ax.imshow(true_gamma, interpolation='none', vmin=vmin, vmax=vmax, origin=origin_true)
ax.set_frame_on(False)

ax = axs[-1]
if FLAG_TITLE:
    ax.set_title('D-bar', fontsize=fz)
ax.imshow(dbar_gamma, interpolation='none', vmin=vmin, vmax=vmax)
ax.set_frame_on(False)

# Remove ticks from all subplots
for ax in axs:
    ax.set_xticks([])
    ax.set_yticks([])

# Get positions (in figure coordinates)
pos_ltrue = axs[-3].get_position()
pos_true  = axs[-2].get_position()
pos_dbar  = axs[-1].get_position()

fig = axs[0].figure

# Common vertical span for the bars (use the first axes)
y0 = pos_ltrue.y0
y1 = pos_ltrue.y1

# Height
if FLAG_TITLE:
    margin = 0.09   # play with this (0.0–0.1)
else:
    margin = 0
y0_bar = max(0.0, y0)
y1_bar = min(1.0, y1 + margin)

# x-location between ltrue and true
xbar1 = 0.5 * (pos_ltrue.x1 + pos_true.x0)

# x-location between last N and D-bar
xbar2 = 0.5 * (pos_true.x1 + pos_dbar.x0)

# Add vertical lines in figure coordinates
for x in (xbar1, xbar2):
    line = Line2D([x, x], [y0_bar, y1_bar],
                  transform=fig.transFigure,
                  color='black', linewidth=2)
    fig.add_artist(line)

    
plt.savefig(plot_folder + "compare_" + pname + ".png", format='png', dpi=300, bbox_inches='tight')
plt.show()
