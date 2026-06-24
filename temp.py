import torch, sys
print("torch    ", torch.__version__)
print("cuda     ", torch.version.cuda)
print("cxx11abi ", torch._C._GLIBCXX_USE_CXX11_ABI)   # pip torch -> usually False
print("python   ", f"cp{sys.version_info.major}{sys.version_info.minor}")