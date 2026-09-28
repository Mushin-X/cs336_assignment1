import typing
import torch
import os

def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
):
    data = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'iteration': iteration
    }
    torch.save(data, out)

def load_checkpoint(
    src: str | os.PathLike | typing.BinaryIO | typing.IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
) -> int:
    data = torch.load(src)
    model.load_state_dict(data['model'])
    if optimizer is not None:
        optimizer.load_state_dict(data['optimizer'])
    return data['iteration']