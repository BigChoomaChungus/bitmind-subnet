import os
import torch
import torch.nn as nn

class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)
        self.device = torch.device(f"cuda:{opt.gpu_ids[0]}" if torch.cuda.is_available() else "cpu")

    def save_networks(self, epoch):
        save_path = f"{self.save_dir}/model_epoch_{epoch}.pth"
        os.makedirs(self.save_dir, exist_ok=True)
        torch.save({'model': self.model.state_dict()}, save_path)
        print(f"Model saved to {save_path}")

    def load_networks(self, epoch):
        load_path = f"{self.save_dir}/model_epoch_{epoch}.pth"
        print(f"loading the model from {load_path}")

        state_dict = torch.load(load_path, map_location=self.device)

        # Check if 'model' key exists, else assume the file contains the state_dict directly
        if 'model' in state_dict:
            self.model.load_state_dict(state_dict['model'])
        else:
            print("[WARNING] 'model' key not found. Attempting to load state_dict directly.")
            self.model.load_state_dict(state_dict)

    def print_networks(self):
        num_params = sum(p.numel() for p in self.model.parameters())
        print(f"[INFO] Model contains {num_params:,} parameters.")


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError(f'Initialization method [{init_type}] is not implemented')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, gain)
            nn.init.constant_(m.bias.data, 0.0)

    print(f'Initializing network with {init_type}')
    net.apply(init_func)
