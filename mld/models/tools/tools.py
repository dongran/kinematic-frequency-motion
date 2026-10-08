import torch.nn as nn


def remove_padding(tensors, lengths):
    return [tensor[:tensor_length] for tensor, tensor_length in zip(tensors, lengths)]


class AutoParams(nn.Module):
    def __init__(self, **kargs):
        try:
            for param in self.needed_params:
                if param in kargs:
                    setattr(self, param, kargs[param])
                else:
                    raise ValueError(f"{param} is needed.")
        except Exception:
            pass

        try:
            for param, default in self.optional_params.items():
                if param in kargs and kargs[param] is not None:
                    setattr(self, param, kargs[param])
                else:
                    setattr(self, param, default)
        except Exception:
            pass
        super().__init__()


def freeze_params(module: nn.Module) -> None:
    for _, param in module.named_parameters():
        param.requires_grad = False

