import os, sys 

import torch
from models import FNO2d as my_model
from util import AdamW as my_optimizer
from util import plt
from util.utilities_module import LpLoss, UnitGaussianNormalizer, count_params, dataset_with_indices
from torch.utils.data import TensorDataset, DataLoader
TensorDatasetID = dataset_with_indices(TensorDataset)

from tqdm import tqdm
from timeit import default_timer
from datetime import datetime


################################################################
#
# initialize
#
################################################################
# TODO: fix path with variables
# Output directory
def make_save_path(save_str, pth = "/fno_exp_"):
    save_path = "results/" + datetime.today().strftime('%Y-%m-%d') + pth + save_str +"/"
    return save_path

def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

seed = 42
set_seed(seed)

torch.set_printoptions(precision=16)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device is", device)


################################################################
#
# user configuration
#
################################################################
# TODO: add args for N_train and noise level
print(sys.argv)


# File I/O
data_folder = '/media/nnelsen/SharedHDD2TB/datasets/eit/'
SAVE_STR = "debug_longer"
SAVE_AFTER = 10                                              # save to disk after this many epochs
FLAG_save_model = True
FLAG_L1 = True
FLAG_BEST = True    # evaluate on best model if true; else eval on last epoch

# Sample size
N_train = 9500      # N_train_max = 10000
N_val = 100
N_test = 400
assert N_train + N_val + N_test <= 10000

# TODO: diff res for train, val, test
# Resolution subsampling
sub_in = 2**2       # input subsample factor (power of two) from s_max_out = 512
sub_out = 2**1      # output subsample factor (power of two) from s_max_out = 256

# FNO
modes1 = 12
modes2 = 12
width = 48
width_final = 256
act = 'relu'
n_layers = 2

# Training
batch_size = 32
epochs = 250
learning_rate = 8e-3
weight_decay = 1e-4
scheduler_step = 50
scheduler_gamma = 0.5
scheduler_iters = epochs #*(N_train//batch_size)
scheduler_patience = 5
scheduler_name = 'CosineAnnealingLR'        # 'CosineAnnealingLR' or 'StepLR'
FLAG_reduce = False                         # use ReduceLROnPlateau

################################################################
#
# load and process data
#
################################################################
save_path = make_save_path(SAVE_STR)
os.makedirs(save_path, exist_ok=True)

start = default_timer()

x_test3 = torch.load(data_folder + 'kernel_3heart_rhop7.pt', weights_only=True)['kernel_3heart'][...,::sub_in,::sub_in]
y_test3 = torch.load(data_folder + 'conductivity_3heart_rhop7.pt', weights_only=True)['conductivity_3heart'][...,::sub_out,::sub_out]

x_test33 = torch.load(data_folder + 'kernel_3heart.pt', weights_only=True)['kernel_3heart'][...,::sub_in,::sub_in]
y_test33 = torch.load(data_folder + 'conductivity_3heart.pt', weights_only=True)['conductivity_3heart'][...,::sub_out,::sub_out]

y_test3 = torch.flip(y_test3, [-2])
y_test33 = torch.flip(y_test33, [-2])

x_train = torch.load(data_folder + 'kernel.pt', weights_only=True)['kernel'][...,::sub_in,::sub_in]
y_train = torch.load(data_folder + 'conductivity.pt', weights_only=True)['conductivity'][...,::sub_out,::sub_out]
mask = torch.load(data_folder + 'mask.pt', weights_only=True)['mask'][::sub_out,::sub_out]
mask = mask.to(device)

# TODO: fix same test data for all experiments; then do random index selection for train
x_test = x_train[-(N_val + N_test):,...]
x_val = x_test[:N_val,...]
x_test = x_test[-N_test:,...]
x_train = x_train[:N_train,...]

y_test = y_train[-(N_val + N_test):,...]
y_val = y_test[:N_val,...]
y_test = y_test[-N_test:,...]
y_train = y_train[:N_train,...]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_val = x_normalizer.encode(x_val)
x_test = x_normalizer.encode(x_test)

# Make the singleton channel dimension match the FNO2D model input shape requirement
x_train = torch.unsqueeze(x_train, 1)
x_val = torch.unsqueeze(x_val, 1)
x_test = torch.unsqueeze(x_test, 1)
x_test3 = x_normalizer.encode(torch.unsqueeze(x_test3, 1))
x_test33 = x_normalizer.encode(torch.unsqueeze(x_test33, 1))

# Data loaders
train_loader = DataLoader(TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)

print("Total time for data processing is", (default_timer()-start), "sec.")

################################################################
#
# training
#
################################################################
s_outputspace = tuple(y_test.shape[-2:])   # same output shape as the output dataset

model = my_model(modes1=modes1,
                 modes2=modes2,
                 width=width,
                 s_outputspace=s_outputspace, 
                 width_final=width_final,
                 act=act,
                 n_layers=n_layers
                 ).to(device)
print("FNO parameter count:", count_params(model))

optimizer = my_optimizer(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
if scheduler_name == 'CosineAnnealingLR':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                           T_max=scheduler_iters)
elif scheduler_name == 'StepLR':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 
                                                step_size=scheduler_step,
                                                gamma=scheduler_gamma)
else:
    raise ValueError(f'Got {scheduler_name=}')
if FLAG_reduce:
    scheduler_val = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=scheduler_gamma,
                                                               patience=scheduler_patience)

if FLAG_L1: # use L1 norm to train
    loss_f = LpLoss(p=1, size_average=False)
    loss_ff = LpLoss(size_average=False)
else: # use L2 norm to train
    loss_f = LpLoss(size_average=False)
    loss_ff = LpLoss(p=1, size_average=False)


errors = torch.zeros((epochs,4))
lowest_val = 10.0  # initialize a test loss threshold
lowest_val_ep = epochs - 1
start = default_timer()
for ep in tqdm(range(epochs)):
    t1 = default_timer()

    train_loss = 0.0
    train_other = 0.0
    model.train()
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        out = model(x)*mask + ~mask # set model to one outside unit disk of radius 1

        loss = loss_f(out, y)
        loss.backward()

        optimizer.step()
        
        with torch.no_grad():
            train_loss += loss.item()
            train_other += loss_ff(out, y).item()

    model.eval()
    val_loss = 0.0
    val_other = 0.0
    with torch.no_grad():
        for x, y in val_loader:
            x, y = x.to(device), y.to(device)

            out = model(x)*mask + ~mask # set model to one outside unit disk of radius 1

            val_loss += loss_f(out, y).item()
            val_other += loss_ff(out, y).item()

    train_loss /= N_train
    val_loss /= N_val
    train_other /= N_train
    val_other /= N_val
    
    scheduler.step()
    if FLAG_reduce:
        scheduler_val.step(val_loss)

    errors[ep,0] = train_loss
    errors[ep,1] = val_loss
    errors[ep,2] = train_other
    errors[ep,3] = val_other

    if FLAG_save_model:
        if not (ep % SAVE_AFTER):
            torch.save(model.state_dict(), save_path + 'model_last.pt')
            
        if val_loss < lowest_val:
            torch.save(model.state_dict(), save_path + 'model.pt')
            lowest_val = val_loss
            lowest_val_ep = ep

    t2 = default_timer()
    ep_time = t2 - t1

    torch.save({'errors': errors}, save_path + 'errors.pt')
    if FLAG_L1:
        print(f'Epoch [{ep+1}/{epochs}], Train L1: {train_loss}, Val L1: {val_loss}, Train L2: {train_other}, Val L2: {val_other}, Time (sec): {ep_time}')
    else:
        print(f'Epoch [{ep+1}/{epochs}], Train L2: {train_loss}, Val L2: {val_loss}, Train L1: {train_other}, Val L1: {val_other}, Time (sec): {ep_time}')

# Final save
if FLAG_save_model:
    torch.save(model.state_dict(), save_path + 'model_last.pt')
if lowest_val_ep >= epochs - 1:
    torch.save(model.state_dict(), save_path + 'model.pt')

end = default_timer()
print("Total time for", epochs, "epochs is", (end-start)/3600, "hours.")
print("Lowest validation error occurs in epoch", lowest_val_ep + 1)


################################################################
#
# evaluation on train and test sets
#
################################################################
# TODO: change model output res for visualization
# s_outputspace = tuple(y_test.shape[-2:])   # same output shape as the output dataset


train_loader = DataLoader(TensorDatasetID(x_train, y_train), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TensorDatasetID(x_test, y_test), batch_size=batch_size, shuffle=False)

if FLAG_save_model:
    if FLAG_BEST:
        model.load_state_dict(torch.load(save_path + 'model.pt', weights_only=True))
    else:
        model.load_state_dict(torch.load(save_path + 'model_last.pt', weights_only=True))
model.eval()

if FLAG_L1:
    loss_f = LpLoss(p=1, size_average=False)
    loss_vec = LpLoss(p=1, size_average=False, reduction=False)
    loss_ff = LpLoss(size_average=False)
    loss_vecf = LpLoss(size_average=False, reduction=False)
else:
    loss_f = LpLoss(size_average=False)
    loss_vec = LpLoss(size_average=False, reduction=False)
    loss_ff = LpLoss(p=1, size_average=False)
    loss_vecf = LpLoss(p=1, size_average=False, reduction=False)

t1 = default_timer()
train_loss = 0.0
train_other = 0.0
out_train = torch.zeros(y_train.shape)
errors_train = torch.zeros(y_train.shape[0], 2)
with torch.no_grad():
    for x, y, idx_train in train_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)*mask + ~mask # set model to one outside unit disk of radius 1

        train_loss += loss_f(out, y).item()
        train_other += loss_ff(out, y).item()
        
        errors_train[idx_train,0] = loss_vec(out, y).cpu()
        errors_train[idx_train,1] = loss_vecf(out, y).cpu()
        
        out_train[idx_train,...] = out.squeeze().cpu()

train_loss /= N_train
train_other /= N_train
t2 = default_timer()
if FLAG_L1:
    print(f'Train L1: {train_loss}, Train L2: {train_other}, Time (sec): {t2-t1}')
else:
    print(f'Train L2: {train_loss}, Train L1: {train_other}, Time (sec): {t2-t1}')

t1 = default_timer()
test_loss = 0.0
test_other = 0.0
out_test = torch.zeros(y_test.shape)
errors_test = torch.zeros(y_test.shape[0], 2)
with torch.no_grad():
    for x, y, idx_test in test_loader:
        x, y = x.to(device), y.to(device)

        out = model(x)*mask + ~mask # set model to one outside unit disk of radius 1

        test_loss += loss_f(out, y).item()
        test_other += loss_ff(out, y).item()
        
        errors_test[idx_test,0] = loss_vec(out,y).cpu()
        errors_test[idx_test,1] = loss_vecf(out,y).cpu()
        
        out_test[idx_test,...] = out.squeeze().cpu()

test_loss /= N_test
test_other /= N_test
t2 = default_timer()
if FLAG_L1:
    print(f'Test L1: {test_loss}, Test L2: {test_other}, Time (sec): {t2-t1}')
else:
    print(f'Test L2: {test_loss}, Test L1: {test_other}, Time (sec): {t2-t1}')
    
################################################################
#
# plotting
#
################################################################
plot_folder = save_path + "figures/"
os.makedirs(plot_folder, exist_ok=True)
mask_plot = torch.clone(mask).cpu()
out_train[:, ~mask_plot] = float('nan')
out_test[:, ~mask_plot] = float('nan')

# Phantom three evaluations
with torch.no_grad():
    out3 = model(x_test3.to(device))*mask + ~mask
    out3 = out3.squeeze().cpu()
    out33 = model(x_test33.to(device))*mask + ~mask
    out33 = out33.squeeze().cpu()
out3[:, ~mask_plot] = float('nan')
out33[:, ~mask_plot] = float('nan')

# %% Errors
plt.close()
errors = torch.load(save_path + 'errors.pt', weights_only=True)['errors']
plt.plot(errors)
plt.legend(["Train L1" if FLAG_L1 else "Train L2", "Test L1" if FLAG_L1 else "Test L2",\
            "Train L2" if FLAG_L1 else "Train L1",\
            "Test L2" if FLAG_L1 else "Test L1"])
plt.tight_layout()
plt.savefig(plot_folder + "loss_epochs" + ".pdf", format='pdf', bbox_inches='tight')

# %% Worst, median, best case inputs (train)
plt.close("all")

idx_worst = torch.argmax(errors_train, dim=0)
idx_median = torch.argsort(errors_train)[errors_train.shape[0]//2, ...]
idx_best = torch.argmin(errors_train, dim=0)

idxs = [idx_worst, idx_median, idx_best]
names = ["worst", "median", "best"]
losses = ["L1" if FLAG_L1 else "L2", "L2" if FLAG_L1 else "L1"]

for loop in range(2):
    loss = losses[loop]
    for i in range(3):
        idx = idxs[i][loop].item()
        true_trainsort = y_train[idx,...].squeeze()
        true_trainsort[~mask_plot] = float('nan')
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

idx_worst = torch.argmax(errors_test, dim=0)
idx_median = torch.argsort(errors_test)[errors_test.shape[0]//2, ...]
idx_best = torch.argmin(errors_test, dim=0)

idxs = [idx_worst, idx_median, idx_best]
names = ["worst", "median", "best"]
losses = ["L1" if FLAG_L1 else "L2", "L2" if FLAG_L1 else "L1"]

for loop in range(2):
    loss = losses[loop]
    for i in range(3):
        idx = idxs[i][loop].item()
        true_testsort = y_test[idx,...].squeeze()
        true_testsort[~mask_plot] = float('nan')
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
# true_train[~mask_plot] = float('nan')
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
# true_test[~mask_plot] = float('nan')
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
    true_test3[~mask_plot] = float('nan')
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
    
for i in range(3):
    true_test3 = y_test33[i,...].squeeze()
    true_test3[~mask_plot] = float('nan')
    plot_test3 = out33[i,...].squeeze()
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
    plt.imshow(x_test33[i,...].squeeze(), origin='lower')
    plt.subplot(2,2,4)
    plt.title('Test PW Error')
    plt.imshow(er_test3, origin='lower')
    plt.box(False)
    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    plt.tight_layout()

    plt.savefig(plot_folder + "eval_phantom" + str(i) + ".png", format='png', dpi=300, bbox_inches='tight')
