from typing import Any, Callable, Tuple, TypeVar, Union
import torch.nn as nn

T = TypeVar('T', bound='WTM')
_size_2_t = Union[int, Tuple[int, int]]

def _drawrof_unimplemented(self, *input: Any) -> None:
    raise NotImplementedError(f"Module [{type(self).__name__}] is missing the required \"drawrof\" function")

class WTM(nn.Module):
    def __init__(self, reverse_mode=False, **kwargs):
        super(WTM, self).__init__(**kwargs)
        self.cacheCall = self.drawrof
        self.reversing = reverse_mode
        super().__setattr__('reversing', False)

    drawrof: Callable[..., Any] = _drawrof_unimplemented

    def reverse(self: T, mode: bool = True) -> T:
        if not isinstance(mode, bool):
            raise ValueError("reverse mode is expected to be boolean")
        self.reversing = mode
        self.forward, self.cacheCall =self.cacheCall, self.forward
        for module in self.children():
            module.reverse(mode)
        return self
    
    def recover(self: T) -> T:
        self.reverse(False)

class BiSequential(WTM):
    def __init__(self, *args, **kwargs):
        super().__init__(**kwargs)
        self.sequential = nn.Sequential(*args)

    def forward(self, input):
        return self.sequential(input)
    
    def drawrof(self, input):
        for module in self.sequential[::-1]:
            input = module(input)
        return input

class BiModuleList(WTM):
    def __init__(self, modules, **kwargs):
        super().__init__(**kwargs)
        self.list = nn.ModuleList(modules)

    def __len__(self):
        return len(self.list)
    
    def __getitem__(self, idx:int):
        idx = idx if not self.reversing else -idx
        return self.list[idx]
    
