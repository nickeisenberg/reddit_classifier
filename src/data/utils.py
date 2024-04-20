def default_unpacker(loader_item: tuple, device):
    inputs, class_id = loader_item
    return inputs.to(device), class_id.to(device)


def transformer_unpacker(loader_item: tuple, device):
    inputs, masks, class_id = loader_item
    return [inputs.to(device), masks.to(device)], class_id.to(device)
