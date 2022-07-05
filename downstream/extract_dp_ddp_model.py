import torch
from train_dp_moco_ddp import DepthPointMOCO


class Wrapper(torch.nn.Module):
    def __init__(self, net):
        super().__init__()
        self.module = net

path = ""   # path of a checkpoint
name = ""   # name of extracted checkpoint

net = DepthPointMOCO(128, 512, 4096*8, 0.07)
wrapper = Wrapper(net)
wrapper.load_state_dict(torch.load(path, map_location=torch.device('cpu'))["state_dict"])

torch.save({"model_state_dict": wrapper.module.net2d.encoder.state_dict()}, "{:s}_depth.pth".format(name))
torch.save({"model_state_dict": wrapper.module.net3d.encoder.state_dict()}, "{:s}_pcd.pth".format(name))
print("ok")
