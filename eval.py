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
exp_date = "2025-10-23"
load_prefix = "paper_sweep"
N_train = 9500
noise = 0
seed = 1

# New eval choices
subfolder = "figures_eval/"
eval_loss_str_list = ["L1", "L0Clip", "DICE"]
noise_new = 1
noise_distribution_new = "uniform"
FLAG_BEST = True
prefix_new = noise_distribution_new + str(noise_new) + "_Best" + str(int(FLAG_BEST)) + "_"

# Get path
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

# %% Worst, median, best case inputs
def quartile_plot(errors_vec, x_data, y_data, mask, out_data, leg="Train", name="train"):
    plt.close("all")
    
    idx_worst = torch.argmax(errors_vec, dim=0)
    idx_median = torch.argsort(errors_vec)[errors_vec.shape[0]//2, ...]
    idx_best = torch.argmin(errors_vec, dim=0)
    
    idxs = [idx_worst, idx_median, idx_best]
    names = ["worst", "median", "best"]
    
    for loop in range(num_eval_losses):
        loss = eval_loss_str_list[loop]
        for i in range(3):
            idx = idxs[i][loop].item()
            true_trainsort = y_data[idx,...].squeeze()
            true_trainsort[~mask.cpu()] = float('nan')
            plot_trainsort = out_data[idx,...].squeeze()
            er_trainsort = torch.abs(plot_trainsort - true_trainsort).squeeze()
            
            true_trainsort = true_trainsort.detach().cpu().numpy()
            vmin = float(np.nanmin(true_trainsort))
            vmax = float(np.nanmax(true_trainsort))
            if not (vmax > vmin):
                vmin, vmax = vmin - 1e-12, vmax + 1e-12

            plt.close()
            plt.figure(3, figsize=(9, 9))
            plt.subplot(2,2,1)
            plt.title(leg + ' Output')
            plt.imshow(plot_trainsort, origin='lower', interpolation='none', vmin=vmin, vmax=vmax)
            plt.box(False)
            plt.subplot(2,2,2)
            plt.title(leg + ' Truth')
            plt.imshow(true_trainsort, origin='lower', interpolation='none', vmin=vmin, vmax=vmax)
            plt.box(False)
            plt.subplot(2,2,3)
            plt.title(leg + ' Input')
            plt.imshow(x_data[idx,...].squeeze(), origin='lower')
            plt.subplot(2,2,4)
            plt.title(leg + ' PW Error')
            plt.imshow(er_trainsort, origin='lower')
            plt.box(False)
            plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
            plt.tight_layout()
        
            plt.savefig(plot_folder + prefix_new + "eval_" + name + "_" + loss + "_" + names[i] + ".png", format='png', dpi=300, bbox_inches='tight')


quartile_plot(errors_test_vec, x_test, y_test, mask_test, out_test, leg="Test", name="test")
quartile_plot(errors_test_clean_vec, x_test_clean, y_test, mask_test, out_test_clean, leg="Test", name="test_clean")

# %% Non-random phantoms of varying contrast
plt.close("all")

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

        plt.close()
        plt.figure(22, figsize=(9, 9))
        plt.subplot(2,2,1)
        plt.title('Test Output')
        plt.imshow(plot_test3, origin='lower', interpolation='none', vmin=vmin, vmax=vmax)
        plt.box(False)
        plt.subplot(2,2,2)
        plt.title('Test Truth')
        plt.imshow(true_test3, origin='lower', interpolation='none', vmin=vmin, vmax=vmax)
        plt.box(False)
        plt.subplot(2,2,3)
        plt.title('Test Input')
        plt.imshow(x[i,...].squeeze(), origin='lower')
        plt.subplot(2,2,4)
        plt.title('Test PW Error')
        plt.imshow(er_test3, origin='lower')
        plt.box(False)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.tight_layout()
    
        plt.savefig(plot_folder + prefix_new + name + str(i) + ".png", format='png', dpi=300, bbox_inches='tight')

OOD_plot(out3, x_test3, "eval_phantom_rhop7_")
OOD_plot(out3_clean, x_test3_clean, "eval_phantom_rhop7_clean_")
