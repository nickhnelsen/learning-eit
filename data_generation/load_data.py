import torch
from utilities_module import UnitGaussianNormalizer, process_raw_ntd


def my_flip(x):
    x = torch.cat((x[...,0:1], torch.flip(x[...,1:], [-1])), -1)
    return x

def my_trunc(x):
    x = x[..., :(x.shape[-2] + 1)//2 + 1, :]
    return x
    
def load_data_ker(data_folder,
                  sub_in,
                  sub_out,
                  device,
                  N_train,
                  N_test,
                  batch_size,
                  NORMALIZE_IN=True
                  ):
    
    # Load kernel form of NtD input data
    x_test3 = torch.load(data_folder + 'kernel_3heart_rhop7.pt')['kernel_3heart'][...,::sub_in,::sub_in]
    y_test3 = torch.load(data_folder + 'conductivity_3heart_rhop7.pt')['conductivity_3heart'][...,::sub_out,::sub_out]

    x_test33 = torch.load(data_folder + 'kernel_3heart.pt')['kernel_3heart'][...,::sub_in,::sub_in]
    y_test33 = torch.load(data_folder + 'conductivity_3heart.pt')['conductivity_3heart'][...,::sub_out,::sub_out]

    y_test3 = torch.flip(y_test3, [-2])
    y_test33 = torch.flip(y_test33, [-2])

    x_train = torch.load(data_folder + 'kernel.pt')['kernel'][...,::sub_in,::sub_in]
    y_train = torch.load(data_folder + 'conductivity.pt')['conductivity'][...,::sub_out,::sub_out]
    mask = torch.load(data_folder + 'mask.pt')['mask'][::sub_out,::sub_out]
    mask = mask.to(device)

    x_test = x_train[-N_test:,...]
    x_train = x_train[:N_train,...]

    y_test = y_train[-N_test:,...]
    y_train = y_train[:N_train,...]

    # Normalize inputs in physical space
    if NORMALIZE_IN:
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_test3 = x_normalizer.encode(x_test3)
        x_test33 = x_normalizer.encode(x_test33)

    # Keep physical space inputs for plotting
    x_train_plot = torch.clone(x_train)
    x_test_plot = torch.clone(x_test)
    x_test3_plot = torch.clone(x_test3)
    x_test33_plot = torch.clone(x_test33)

    # Map back to NtD fft2 form
    tmp = torch.zeros(x_train.shape, dtype=torch.cfloat)
    for batch_idx, x in zip(torch.split(torch.arange(x_train.shape[0]), batch_size),
                            torch.split(x_train, batch_size)):
        tmp[batch_idx,...] = my_flip(torch.fft.fft2(x, norm="forward"))
    x_train = my_trunc(tmp)

    tmp = torch.zeros(x_test.shape, dtype=torch.cfloat)
    for batch_idx, x in zip(torch.split(torch.arange(x_test.shape[0]), batch_size),
                            torch.split(x_test, batch_size)):
        tmp[batch_idx,...] = my_flip(torch.fft.fft2(x, norm="forward"))
    x_test = my_trunc(tmp)

    tmp = my_flip(torch.fft.fft2(x_test3, norm="forward"))
    x_test3 = my_trunc(tmp)

    tmp = my_flip(torch.fft.fft2(x_test33, norm="forward"))
    x_test33 = my_trunc(tmp)
    del tmp

    # Make the singleton channel dimension match the OR-FNM model input shape requirement
    x_train = x_train[:,None,None,...]
    x_test = x_test[:,None,None,...]
    x_test3 = x_test3[:,None,None,...]
    x_test33 = x_test33[:,None,None,...]
    
    # Group
    X_all = x_train, x_test, x_test3, x_test33
    Y_all = y_train, y_test, y_test3, y_test33
    X_plot = x_train_plot, x_test_plot, x_test3_plot, x_test33_plot
    
    return mask, X_all, Y_all, X_plot


def load_data_ntd(data_folder,
                  sub_in,
                  sub_out,
                  device,
                  N_train,
                  N_test,
                  batch_size,
                  NORMALIZE_IN=True
                  ):
    
    # Output and mask
    y_test3 = torch.load(data_folder + 'conductivity_3heart_rhop7.pt')['conductivity_3heart'][...,::sub_out,::sub_out]
    y_test33 = torch.load(data_folder + 'conductivity_3heart.pt')['conductivity_3heart'][...,::sub_out,::sub_out]
    y_test3 = torch.flip(y_test3, [-2])
    y_test33 = torch.flip(y_test33, [-2])
    y_train = torch.load(data_folder + 'conductivity.pt')['conductivity'][...,::sub_out,::sub_out]
    mask = torch.load(data_folder + 'mask.pt')['mask'][::sub_out,::sub_out]
    mask = mask.to(device)
    y_test = y_train[-N_test:,...]
    y_train = y_train[:N_train,...]

    # Load ntd form of NtD input data
    x_train = torch.load(data_folder + 'ntd.pt')['ntd'].unsqueeze(1)
    x_test3 = torch.load(data_folder + 'ntd_3heart_rhop7.pt')['ntd_3heart'].unsqueeze(1)
    x_test33 = torch.load(data_folder + 'ntd_3heart.pt')['ntd_3heart'].unsqueeze(1)

    x_test = x_train[-N_test:,...]
    x_train = x_train[:N_train,...]
    s_in = (x_test.shape[-1] + 1)//sub_in

    # Process raw inputs
    tmp = torch.zeros(x_train.shape[0], 1, s_in//2 + 1, s_in, dtype=torch.cfloat)
    for batch_idx, x in zip(torch.split(torch.arange(x_train.shape[0]), batch_size),
                            torch.split(x_train, batch_size)):
        tmp[batch_idx,...] = process_raw_ntd(x, (s_in, s_in))
    x_train = tmp
    tmp = torch.zeros(x_test.shape[0], 1, s_in//2 + 1, s_in, dtype=torch.cfloat)
    for batch_idx, x in zip(torch.split(torch.arange(x_test.shape[0]), batch_size),
                            torch.split(x_test, batch_size)):
        tmp[batch_idx,...] = process_raw_ntd(x, (s_in, s_in))
    x_test = tmp
    x_test3 = process_raw_ntd(x_test3, (s_in, s_in))
    x_test33 = process_raw_ntd(x_test33, (s_in, s_in))
    del tmp

    # Normalize inputs in physical space
    if NORMALIZE_IN:
        x_normalizer = UnitGaussianNormalizer(x_train)
        x_train = x_normalizer.encode(x_train)
        x_test = x_normalizer.encode(x_test)
        x_test3 = x_normalizer.encode(x_test3)
        x_test33 = x_normalizer.encode(x_test33)

    # Make the singleton channel dimension match the OR-FNM model input shape requirement
    x_train = x_train[:,None,...]
    x_test = x_test[:,None,...]
    x_test3 = x_test3[:,None,...]
    x_test33 = x_test33[:,None,...]
    
    # Group
    X_all = x_train, x_test, x_test3, x_test33
    Y_all = y_train, y_test, y_test3, y_test33
    
    return mask, X_all, Y_all