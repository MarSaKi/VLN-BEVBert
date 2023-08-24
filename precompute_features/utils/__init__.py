def convert_weights_cuda_cpu(weights, device):
    names = list(weights.keys())
    is_module = names[0].split('.')[0] == 'module'
    if device == 'cuda' and not is_module:
        new_weights = {'module.'+k:v for k,v in weights.items()}
    elif device == 'cpu' and is_module:
        new_weights = {'.'.join(k.split('.')[1:]):v for k,v in weights.items()}
    else:
        new_weights = weights
    return new_weights
