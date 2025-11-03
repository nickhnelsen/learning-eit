import os, sys, yaml
import torch
from models import FNO2d as my_model
from util import AdamW as my_optimizer
from util import plt
from util.sample_random_fields import RandomField
from util.utilities_module import LpLoss, L0Loss, RatioLoss, UnitGaussianNormalizer, count_params, dataset_with_indices, make_save_path, set_seed, integrate
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)
import copy
from tqdm import tqdm
from timeit import default_timer

torch.set_printoptions(precision=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)

################################################################
#
# user configuration
#
################################################################
# Config
CONFIG_PATH = "config.yaml"     # hard coded path
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

# Driver arguments
print(sys.argv)
N_train = int(sys.argv[1])
noise = int(sys.argv[2])        # "noise" percent noise; integer only for simplicity
seed = int(sys.argv[3])         # seed and Monte Carlo index for loops
if seed is not None:
    set_seed(seed)

# File I/O
data_folder = config['data_folder']
SAVE_STR = config['SAVE_STR']
SAVE_AFTER = config['SAVE_AFTER']
FLAG_save_model = config['FLAG_save_model']

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
epochs = config['epochs']
learning_rate = config['learning_rate']
weight_decay = config['weight_decay']
scheduler_step = config['scheduler_step']
scheduler_gamma = config['scheduler_gamma']
scheduler_iters = config['scheduler_iters']
scheduler_patience = config['scheduler_patience']
scheduler_name = config['scheduler_name']
FLAG_reduce = config['FLAG_reduce']
FLAG_BEST = config['FLAG_BEST']
FLAG_MEAN_REDUCTION = config['FLAG_MEAN_REDUCTION']
FLAG_SHUFFLE = config['FLAG_SHUFFLE']
train_loss_str = config['train_loss_str']
eval_loss_str_list = config['eval_loss_str_list']
noise_distribution = config['noise_distribution']

# Checks
assert N_train + N_val + N_test <= N_max
assert sub_in_test <= sub_in and sub_out_test <= sub_out

if scheduler_iters is None:
    scheduler_iters = epochs
    
valid_losses = {"L1", "L2", "L0", "Ratio"}
if train_loss_str not in valid_losses:
    raise ValueError(f"Invalid value for train_loss_str: {train_loss_str}. Must be one of {sorted(valid_losses)}.")
for loss in eval_loss_str_list:
    if loss not in valid_losses:
        raise ValueError(f"Invalid value for eval loss: {loss}. Must be one of {sorted(valid_losses)}.")
if train_loss_str != eval_loss_str_list[0]:
    raise ValueError("Train loss must be included in eval loss and in the first entry of the list")

################################################################
#
# load and process data
#
################################################################
SAVE_STR = SAVE_STR + "_N" + str(N_train) + "_Noise" + str(noise) + "_Seed" + str(seed)
save_path = make_save_path(SAVE_STR)
os.makedirs(save_path, exist_ok=True)

with open(save_path + CONFIG_PATH, "w") as f:
    yaml.safe_dump(
        config,
        f,
        sort_keys=False,            # keep key order
        default_flow_style=False,   # block style (human-readable)
        allow_unicode=True,
        indent=2
    )

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
def get_noisy(dataset):
    rf = RandomField(dataset.shape[-1], distribution=noise_distribution, device=device)
    dataset_noisy = rf.generate_noise_dataset(dataset.shape[0])
    dataset_noisy = (noise/100)*(integrate(dataset**2).sqrt()[:,None,None])*dataset_noisy
    dataset_noisy = dataset + dataset_noisy
    return dataset_noisy

if noise > 0.0:
    x_train = get_noisy(x_train)
    x_test3 = get_noisy(x_test3)

x_test = x_train[-(N_val + N_test):,...]
x_val = x_test[:N_val,...]
x_test = x_test[-N_test:,...]
x_train = x_train[:-(N_val + N_test),...]

y_test = y_train[-(N_val + N_test):,...]
y_val = y_test[:N_val,::sub_out_ratio,::sub_out_ratio]
y_test = y_test[-N_test:,...]
y_train = y_train[:-(N_val + N_test),...]

# Shuffle training set selection
if FLAG_SHUFFLE:
    dataset_shuffle_idx = torch.randperm(x_train.shape[0])
    x_train = x_train[dataset_shuffle_idx,...]
    y_train = y_train[dataset_shuffle_idx,...]
else:
    dataset_shuffle_idx = torch.arange(x_train.shape[0])
    
torch.save(dataset_shuffle_idx, save_path + 'idx_shuffle.pt')
x_train = x_train[:N_train,...]
y_train = y_train[:N_train,::sub_out_ratio,::sub_out_ratio]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)[:,::sub_in_ratio,::sub_in_ratio]
x_val = x_normalizer.encode(x_val)[:,::sub_in_ratio,::sub_in_ratio]
x_test = x_normalizer.encode(x_test)
x_test_clean = x_normalizer.encode(x_test_clean)
x_test3 = x_normalizer.encode(x_test3)
x_test3_clean = x_normalizer.encode(x_test3_clean)

# Make the singleton channel dimension match the FNO2D model input shape requirement
x_train = torch.unsqueeze(x_train, 1)
x_val = torch.unsqueeze(x_val, 1)
x_test = torch.unsqueeze(x_test, 1)
x_test_clean = torch.unsqueeze(x_test_clean, 1)
x_test3 = torch.unsqueeze(x_test3, 1)
x_test3_clean = torch.unsqueeze(x_test3_clean, 1)

# Data loaders for training
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

print("Total time for data processing is", (default_timer()-start), "sec.")

################################################################
#
# training
#
################################################################
model = my_model(modes1=modes1,
                 modes2=modes2,
                 width=width,
                 width_final=width_final,
                 act=act,
                 n_layers=n_layers
                 ).to(device)
print(model)
print("FNO parameter count:", count_params(model))

optimizer = my_optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
print(optimizer)

if scheduler_name == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=scheduler_iters)
elif scheduler_name == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=scheduler_step,
                                                gamma=scheduler_gamma)
else:
    raise ValueError(f'Got {scheduler_name=}')
print(scheduler)
if FLAG_reduce:
    scheduler_val = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=scheduler_gamma,
                                                               patience=scheduler_patience)

# Set loss and minibatch reduction type
num_eval_losses = len(eval_loss_str_list)
loss_dict = {"L1": LpLoss(p=1, size_average=FLAG_MEAN_REDUCTION), 
             "L2": LpLoss(p=2, size_average=FLAG_MEAN_REDUCTION),
             "L0": L0Loss(size_average=FLAG_MEAN_REDUCTION),
             "Ratio": RatioLoss(size_average=FLAG_MEAN_REDUCTION)
}
loss_f = loss_dict[train_loss_str]

errors_train_hist = torch.zeros((epochs, num_eval_losses))
errors_val_hist = torch.zeros((epochs, num_eval_losses))
lowest_val = 10.0   # initialize a test loss threshold
lowest_val_ep = epochs - 1
best_state = copy.deepcopy(model.state_dict())

start = default_timer()
for ep in tqdm(range(epochs)):
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)*mask + ~mask # set model to one outside unit disk of radius 1

        loss = loss_f(out, y)
        loss.backward()

        optimizer.step()
        
        with torch.no_grad():
            errors_train_hist[ep, 0] += loss.item()
            for i in range(num_eval_losses - 1):
                errors_train_hist[ep, i + 1] += loss_dict[eval_loss_str_list[i + 1]](out, y).item()

    model.eval()
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)*mask + ~mask # set model to one outside unit disk of radius 1

            for i, my_str in enumerate(eval_loss_str_list):
                errors_val_hist[ep, i] += loss_dict[my_str](out, y).item()

    if FLAG_MEAN_REDUCTION:
        errors_train_hist[ep, ...] /= len(train_loader)
        errors_val_hist[ep, ...] /= len(val_loader)
    else:
        errors_train_hist[ep, ...] /= N_train
        errors_val_hist[ep, ...] /= N_val
    
    scheduler.step()
    if FLAG_reduce:
        scheduler_val.step(errors_val_hist[ep, 0])
           
    if errors_val_hist[ep, 0] < lowest_val:
        lowest_val = errors_val_hist[ep, 0]
        lowest_val_ep = ep
        best_state = copy.deepcopy(model.state_dict())
        
    if FLAG_save_model:
        if not (ep % SAVE_AFTER):
            torch.save(model.state_dict(), save_path + 'model_last.pt')
            torch.save(best_state, save_path + 'model_best.pt')

    torch.save(errors_train_hist, save_path + 'errors_train_hist.pt')
    torch.save(errors_val_hist, save_path + 'errors_val_hist.pt')

    print(f'Train {train_loss_str}: {errors_train_hist[ep,0]}, Val {train_loss_str}: {errors_val_hist[ep,0]}')

# Final save
if FLAG_save_model:
    torch.save(model.state_dict(), save_path + 'model_last.pt')
    torch.save(best_state, save_path + 'model_best.pt')
    
end = default_timer()
print("Total time for", epochs, "epochs is", (end-start)/3600, "hours.")
print("Lowest validation error occurs in epoch", lowest_val_ep + 1)

################################################################
#
# evaluation on train and test sets
#
################################################################
train_loader = DataLoader(TensorDatasetID(x_train, y_train), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDatasetID(x_test, y_test), batch_size=batch_size, shuffle=False)
test_clean_loader = DataLoader(TensorDatasetID(x_test_clean, y_test), batch_size=batch_size, shuffle=False)

if FLAG_BEST:
    print("Evaluating the best model.")
    model.load_state_dict(best_state)
else:
    print("Evaluating the final epoch model.")
model.eval()

# Assumes sum reduction for exact loss calculations
loss_dict = {"L1": LpLoss(p=1, size_average=False), 
             "L2": LpLoss(p=2, size_average=False),
             "L0": L0Loss(size_average=False),
             "Ratio": RatioLoss(size_average=False)
}
loss_vec_dict = {"L1": LpLoss(p=1, size_average=False, reduction=False), 
             "L2": LpLoss(p=2, size_average=False, reduction=False),
             "L0": L0Loss(size_average=False, reduction=False),
             "Ratio": RatioLoss(size_average=False, reduction=False)
}


def evaluate_my_loader(loader, y_data, mask, type="Train"):
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
    
    
errors_train, out_train, errors_train_vec = evaluate_my_loader(train_loader, y_train, mask, "Train")
errors_test, out_test, errors_test_vec = evaluate_my_loader(test_loader, y_test, mask_test, "Test")
errors_test_clean, out_test_clean, errors_test_clean_vec = evaluate_my_loader(test_clean_loader, y_test, mask_test, "Test (Clean)")

# Save final test errors
torch.save(errors_train, save_path + 'errors_train.pt')
torch.save(errors_test, save_path + 'errors_test.pt')
torch.save(errors_test_clean, save_path + 'errors_test_clean.pt')

################################################################
#
# plotting
#
################################################################
plot_folder = save_path + "figures/"
os.makedirs(plot_folder, exist_ok=True)

handlelength = 3.0     # 2.75
borderpad = 0.25     # 0.15
color_list = ['k', 'C3', 'C5', 'C1', 'C2', 'C0', 'C4', 'C6', 'C7', 'C8', 'C9'] # black, red, brown, orange, green, blue, purple, pink, gray, olive, cyan

out_train[:, ~mask.cpu()] = float('nan')
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

# %% Error vs. epochs plot
plt.close("all")
plt.figure(0)
for i in range(num_eval_losses):
    plt.plot(errors_train_hist[..., i], ls="-.", color=color_list[i], label="Train " + eval_loss_str_list[i])
    plt.plot(errors_val_hist[..., i], ls="-", color=color_list[i], label="Val " + eval_loss_str_list[i])

plt.legend(framealpha=0.75, loc='best', borderpad=borderpad,handlelength=handlelength).set_draggable(True)
plt.xlabel(r'Epoch')
plt.tight_layout()
plt.savefig(plot_folder + "loss_epochs" + ".pdf", format='pdf', bbox_inches='tight')

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
            
            plt.close()
            plt.figure(3, figsize=(9, 9))
            plt.subplot(2,2,1)
            plt.title(leg + ' Output')
            plt.imshow(plot_trainsort, origin='lower', interpolation='none')
            plt.box(False)
            plt.subplot(2,2,2)
            plt.title(leg + ' Truth')
            plt.imshow(true_trainsort, origin='lower', interpolation='none')
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
        
            plt.savefig(plot_folder + "eval_" + name + "_" + loss + "_" + names[i] + ".png", format='png', dpi=300, bbox_inches='tight')


quartile_plot(errors_train_vec, x_train, y_train, mask, out_train, leg="Train", name="train")
quartile_plot(errors_test_vec, x_test, y_test, mask_test, out_test, leg="Test", name="test")
quartile_plot(errors_test_clean_vec, x_test_clean, y_test, mask_test, out_test_clean, leg="Test", name="test_clean")
        
# %% Train point
# =============================================================================
# plt.close("all")
# 
# pind_train = torch.randint(N_train, [1]).item()
# 
# true_train = y_train[pind_train,...].squeeze()
# true_train[~mask.cpu()] = float('nan')
# plot_train = out_train[pind_train,...].squeeze()
# er_train = torch.abs(plot_train - true_train).squeeze()
# 
# plt.figure(1, figsize=(9, 9))
# plt.subplot(2,2,1)
# plt.title('Train Output')
# plt.imshow(plot_train, origin='lower', interpolation='none')
# plt.box(False)
# plt.subplot(2,2,2)
# plt.title('Train Truth')
# plt.imshow(true_train, origin='lower', interpolation='none')
# plt.box(False)
# plt.subplot(2,2,3)
# plt.title('Train Input')
# plt.imshow(x_train[pind_train,FLAG_PLOT_IMAG,...].squeeze(), origin='lower')
# plt.subplot(2,2,4)
# plt.title('Train PW Error')
# plt.imshow(er_train, origin='lower')
# plt.box(False)
# plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
# plt.tight_layout()
# plt.show()
# =============================================================================

# %% Save train or not
# plt.savefig(plot_folder + "eval_train" + str(pind_train) + ".png", format='png', dpi=300, bbox_inches='tight')

# %% Test point
# =============================================================================
# plt.close("all")
# 
# pind_test = torch.randint(N_test, [1]).item()
# 
# true_test = y_test[pind_test,...].squeeze()
# true_test[~mask_test.cpu()] = float('nan')
# plot_test = out_test[pind_test,...].squeeze()
# er_test = torch.abs(plot_test - true_test).squeeze()
# 
# plt.figure(11, figsize=(9, 9))
# plt.subplot(2,2,1)
# plt.title('Test Output')
# plt.imshow(plot_test, origin='lower', interpolation='none')
# plt.box(False)
# plt.subplot(2,2,2)
# plt.title('Test Truth')
# plt.imshow(true_test, origin='lower', interpolation='none')
# plt.box(False)
# plt.subplot(2,2,3)
# plt.title('Test Input')
# plt.imshow(x_test[pind_test,FLAG_PLOT_IMAG,...].squeeze(), origin='lower')
# plt.subplot(2,2,4)
# plt.title('Test PW Error')
# plt.imshow(er_test, origin='lower')
# plt.box(False)
# plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
# plt.tight_layout()
# plt.show()
# =============================================================================

# %% Save test or not
# plt.savefig(plot_folder + "eval_test" + str(pind_test) + ".png", format='png', dpi=300, bbox_inches='tight')

# %% Non-random phantoms of varying contrast
plt.close("all")

def OOD_plot(out, x, name):
    for i in range(3):
        true_test3 = y_test3[i,...].squeeze()
        true_test3[~mask_test.cpu()] = float('nan')
        plot_test3 = out[i,...].squeeze()
        er_test3 = torch.abs(plot_test3 - true_test3).squeeze()
        
        plt.close()
        plt.figure(22, figsize=(9, 9))
        plt.subplot(2,2,1)
        plt.title('Test Output')
        plt.imshow(plot_test3, origin='lower', interpolation='none')
        plt.box(False)
        plt.subplot(2,2,2)
        plt.title('Test Truth')
        plt.imshow(true_test3, origin='lower', interpolation='none')
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
    
        plt.savefig(plot_folder + name + str(i) + ".png", format='png', dpi=300, bbox_inches='tight')

OOD_plot(out3, x_test3, "eval_phantom_rhop7_")
OOD_plot(out3_clean, x_test3_clean, "eval_phantom_rhop7_clean_")
