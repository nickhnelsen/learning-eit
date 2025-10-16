import os, sys 

import torch
from models import FNO2d as my_model
from util import AdamW as my_optimizer
from util import plt
from util.utilities_module import LpLoss, L0Loss, RatioLoss, UnitGaussianNormalizer, count_params, dataset_with_indices
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)
import copy

from tqdm import tqdm
from timeit import default_timer
from datetime import datetime


################################################################
#
# initialize
#
################################################################
# Output directory
def make_save_path(save_str, pth = "/"):
    save_path = "results/" + datetime.today().strftime('%Y-%m-%d') + pth + save_str +"/"
    return save_path

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

torch.set_printoptions(precision=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)


################################################################
#
# user configuration
#TODO: move all to config file, then save config to save_path
#
################################################################
# TODO: add args for noise level and seed/MC index
print(sys.argv)
N_train = int(sys.argv[1])
noise = int(sys.argv[2])  # TODO: 0 for zero 1 for X percent noise; todo sample noise
seed = int(sys.argv[3])
if seed is not None:
    set_seed(seed)

# File I/O
data_folder = '/media/nnelsen/SharedHDD2TB/datasets/eit/'
SAVE_STR = "debug_eval_losses"
SAVE_AFTER = 10     # save to disk after this many epochs
FLAG_save_model = True

# Sample size
N_val = 100
N_test = 400
assert N_train + N_val + N_test <= 10000    # N_train_max = 10000

# Resolution subsampling
sub_in = 2**2       # input subsample factor (power of two) from s_max_out = 512
sub_out = 2**1      # output subsample factor (power of two) from s_max_out = 256
sub_in_test = 2**1
sub_out_test = 2**0
assert sub_in_test <= sub_in and sub_out_test <= sub_out

# FNO
modes1 = 12         # default: 12
modes2 = 12         # default: 12
width = 48          # default: 48
width_final = 256   # default: 256
act = 'relu'        # default: 'relu' for rough outputs, 'gelu' for smooth outputs
n_layers = 2        # default: 2

# Training, evaluation, and testing
batch_size = 32
epochs = 250
learning_rate = 8e-3
weight_decay = 1e-4
scheduler_step = 50
scheduler_gamma = 0.5
scheduler_iters = epochs
scheduler_patience = 5
scheduler_name = 'CosineAnnealingLR'        # 'CosineAnnealingLR' or 'StepLR'
FLAG_reduce = False                         # use ReduceLROnPlateau
FLAG_BEST = True                            # evaluate on best model if true; else eval on last epoch
FLAG_MEAN_REDUCTION = True                  # more stable to choice of batch size
train_loss_str = "L1"
eval_loss_str_list = ["L1", "L2", "Ratio"]

# Check losses
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
# TODO: fix path with user variables (noise value)
SAVE_STR = SAVE_STR + "_N" + str(N_train) + "_Noise" + str(noise) + "_Seed" + str(seed)
save_path = make_save_path(SAVE_STR)
os.makedirs(save_path, exist_ok=True)

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

# TODO: fix same test data for all experiments; then do random index selection for train
x_test = x_train[-(N_val + N_test):,...]
x_val = x_test[:N_val,...]
x_test = x_test[-N_test:,...]
x_train = x_train[:N_train,...]

y_test = y_train[-(N_val + N_test):,...]
y_val = y_test[:N_val,::sub_out_ratio,::sub_out_ratio]
y_test = y_test[-N_test:,...]
y_train = y_train[:N_train,::sub_out_ratio,::sub_out_ratio]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)[:,::sub_in_ratio,::sub_in_ratio]
x_val = x_normalizer.encode(x_val)[:,::sub_in_ratio,::sub_in_ratio]
x_test = x_normalizer.encode(x_test)
x_test3 = x_normalizer.encode(x_test3)

# Make the singleton channel dimension match the FNO2D model input shape requirement
x_train = torch.unsqueeze(x_train, 1)
x_val = torch.unsqueeze(x_val, 1)
x_test = torch.unsqueeze(x_test, 1)
x_test3 = torch.unsqueeze(x_test3, 1)

# Data loaders
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

if FLAG_BEST:
    model.load_state_dict(best_state)

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

t1 = default_timer()
errors_train = torch.zeros(num_eval_losses)
out_train = torch.zeros(y_train.shape)
errors_train_vec = torch.zeros(y_train.shape[0], num_eval_losses)
with torch.no_grad():
    for x, y, idx_train in train_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)*mask + ~mask # set model to one outside unit disk of radius 1

        for i, my_str in enumerate(eval_loss_str_list):
            errors_train[i] += loss_dict[my_str](out, y).item()
            errors_train_vec[idx_train, i] = loss_vec_dict[my_str](out, y).cpu()
        
        out_train[idx_train,...] = out.squeeze().cpu()

errors_train /= N_train
t2 = default_timer()
combined_dict = dict(zip(eval_loss_str_list, errors_train))
print(f'Eval Time (sec): {t2-t1}, Train ' + ", ".join(f"{k}: {v:.4f}" for k,v in combined_dict.items()))

t1 = default_timer()
errors_test = torch.zeros(num_eval_losses)
out_test = torch.zeros(y_test.shape)
errors_test_vec = torch.zeros(y_test.shape[0], num_eval_losses)
with torch.no_grad():
    for x, y, idx_test in test_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)*mask_test + ~mask_test # set model to one outside unit disk of radius 1
        
        for i, my_str in enumerate(eval_loss_str_list):
            errors_test[i] += loss_dict[my_str](out, y).item()
            errors_test_vec[idx_test, i] = loss_vec_dict[my_str](out, y).cpu()
        
        out_test[idx_test,...] = out.squeeze().cpu()

errors_test /= N_test
t2 = default_timer()
combined_dict = dict(zip(eval_loss_str_list, errors_test))
print(f'Eval Time (sec): {t2-t1}, Test ' + ", ".join(f"{k}: {v:.4f}" for k,v in combined_dict.items()))

# Save final test errors
torch.save(errors_train, save_path + 'errors_train.pt')
torch.save(errors_test, save_path + 'errors_test.pt')

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

# Phantom three evaluations
with torch.no_grad():
    out3 = model(x_test3.to(device))*mask_test + ~mask_test
    out3 = out3.squeeze().cpu()
out3[:, ~mask_test.cpu()] = float('nan')

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

# %% Worst, median, best case inputs (train)
plt.close("all")

idx_worst = torch.argmax(errors_train_vec, dim=0)
idx_median = torch.argsort(errors_train_vec)[errors_train_vec.shape[0]//2, ...]
idx_best = torch.argmin(errors_train_vec, dim=0)

idxs = [idx_worst, idx_median, idx_best]
names = ["worst", "median", "best"]

for loop in range(num_eval_losses):
    loss = eval_loss_str_list[loop]
    for i in range(3):
        idx = idxs[i][loop].item()
        true_trainsort = y_train[idx,...].squeeze()
        true_trainsort[~mask.cpu()] = float('nan')
        plot_trainsort = out_train[idx,...].squeeze()
        er_trainsort = torch.abs(plot_trainsort - true_trainsort).squeeze()
        
        plt.close()
        plt.figure(3, figsize=(9, 9))
        plt.subplot(2,2,1)
        plt.title('Train Output')
        plt.imshow(plot_trainsort, origin='lower', interpolation='none')
        plt.box(False)
        plt.subplot(2,2,2)
        plt.title('Train Truth')
        plt.imshow(true_trainsort, origin='lower', interpolation='none')
        plt.box(False)
        plt.subplot(2,2,3)
        plt.title('Train Input')
        plt.imshow(x_train[idx,...].squeeze(), origin='lower')
        plt.subplot(2,2,4)
        plt.title('Train PW Error')
        plt.imshow(er_trainsort, origin='lower')
        plt.box(False)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.tight_layout()
    
        plt.savefig(plot_folder + "eval_train_" + loss + "_" + names[i] + ".png", format='png', dpi=300, bbox_inches='tight')
    
# %% Worst, median, best case inputs (test)
plt.close("all")

idx_worst = torch.argmax(errors_test_vec, dim=0)
idx_median = torch.argsort(errors_test_vec)[errors_test_vec.shape[0]//2, ...]
idx_best = torch.argmin(errors_test_vec, dim=0)

idxs = [idx_worst, idx_median, idx_best]
names = ["worst", "median", "best"]

for loop in range(num_eval_losses):
    loss = eval_loss_str_list[loop]
    for i in range(3):
        idx = idxs[i][loop].item()
        true_testsort = y_test[idx,...].squeeze()
        true_testsort[~mask_test.cpu()] = float('nan')
        plot_testsort = out_test[idx,...].squeeze()
        er_testsort = torch.abs(plot_testsort - true_testsort).squeeze()
        
        plt.close()
        plt.figure(3, figsize=(9, 9))
        plt.subplot(2,2,1)
        plt.title('Test Output')
        plt.imshow(plot_testsort, origin='lower', interpolation='none')
        plt.box(False)
        plt.subplot(2,2,2)
        plt.title('Test Truth')
        plt.imshow(true_testsort, origin='lower', interpolation='none')
        plt.box(False)
        plt.subplot(2,2,3)
        plt.title('Test Input')
        plt.imshow(x_test[idx,...].squeeze(), origin='lower')
        plt.subplot(2,2,4)
        plt.title('Test PW Error')
        plt.imshow(er_testsort, origin='lower')
        plt.box(False)
        plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
        plt.tight_layout()
    
        plt.savefig(plot_folder + "eval_test_" + loss + "_" + names[i] + ".png", format='png', dpi=300, bbox_inches='tight')
        
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

for i in range(3):
    true_test3 = y_test3[i,...].squeeze()
    true_test3[~mask_test.cpu()] = float('nan')
    plot_test3 = out3[i,...].squeeze()
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
    plt.imshow(x_test3[i,...].squeeze(), origin='lower')
    plt.subplot(2,2,4)
    plt.title('Test PW Error')
    plt.imshow(er_test3, origin='lower')
    plt.box(False)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()

    plt.savefig(plot_folder + "eval_phantom_rhop7_" + str(i) + ".png", format='png', dpi=300, bbox_inches='tight')
