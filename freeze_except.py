
def freeze_except(model, layer_names):
    """
    Freeze all parameters in the model except those whose names contain any of the specified substrings in layer_names.

    Args:
        model (torch.nn.Module): The model to modify.
        layer_names (list of str): A list of substrings. Parameters with names containing any of these substrings will not be frozen.

    Returns:
        torch.nn.Module: The modified model with specific parameters unfrozen.
    """
    if not isinstance(layer_names, list):
        layer_names = [layer_names]  # 如果是单个字符串，将其转换为列表

    for name, param in model.named_parameters():
        if any(layer_name in name for layer_name in layer_names):  # 检查是否包含任意子字符串
            param.requires_grad = True
            print(f'The name "{name}" has requires_grad set to {param.requires_grad}.')
        else:
            param.requires_grad = False

    print('\nOther layers are frozen.')
    return model