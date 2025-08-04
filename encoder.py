import torch
import torch.nn as nn
import clip
from torchvision.models import resnet152, densenet161, vgg19
from transformers import ViTModel
import contextlib 

class Encoder(nn.Module):
    def __init__(self, network='clip_rn50', is_finetune='False'):
        super(Encoder, self).__init__()
        self.network = network
        self.is_finetune = is_finetune
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        if network == 'resnet152':
            self.net = resnet152(pretrained=True)
            self.net = nn.Sequential(*list(self.net.children())[:-2])  
            self.dim = 2048


        elif network == 'clip_rn50':
            model, _ = clip.load("RN50", device=device, jit=False)
            self.clip_model = model.visual.float().to(device)
            self.dim = 2048
        
        elif network == 'clip_vit':
  
            model, _ = clip.load("ViT-B/32", device=device, jit=False)
            self.clip_model = model.visual.float().to(device)
            self.dim = 768

        elif network == 'vit_base':
            vit_local_path = './vit_models/vit_base/models--google--vit-base-patch16-224-in21k'  
            self.vit = ViTModel.from_pretrained(vit_local_path)
            self.vit.to(device)
            self.dim = 768

        self.set_finetune()

    def forward(self, x):
        
        grad_context = torch.no_grad() if self.is_finetune == 'True' else contextlib.nullcontext()
        
        if self.network == 'clip_rn50':
            with grad_context:
                x = self.clip_model.conv1(x)
                x = self.clip_model.bn1(x)
                x = self.clip_model.relu1(x)
                x = self.clip_model.conv2(x)
                x = self.clip_model.bn2(x)
                x = self.clip_model.relu2(x)
                x = self.clip_model.conv3(x)
                x = self.clip_model.bn3(x)
                x = self.clip_model.relu3(x)
                x = self.clip_model.avgpool(x)

                x = self.clip_model.layer1(x)
                x = self.clip_model.layer2(x)
                x = self.clip_model.layer3(x)
                x = self.clip_model.layer4(x)
                x = x.permute(0, 2, 3, 1)    
                x = x.view(x.size(0), -1, x.size(-1)) 

        elif self.network == 'clip_vit':
            with grad_context:
    
                x = self.clip_model.conv1(x)  
                x = x.reshape(x.shape[0], x.shape[1], -1)  
                x = x.permute(0, 2, 1) 
            
                class_embedding = self.clip_model.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device)
                x = torch.cat([class_embedding, x], dim=1) 
                
                x = x + self.clip_model.positional_embedding.to(x.dtype)
                
                x = self.clip_model.ln_pre(x)

                x = x.permute(1, 0, 2)  
                x = self.clip_model.transformer(x)
                x = x.permute(1, 0, 2) 
                
                x = self.clip_model.ln_post(x)
                
                x = x[:, 1:, :]

        elif self.network == 'vit_base':
            with grad_context:
                vit_outputs = self.vit(x, output_hidden_states=False, return_dict=True)
                x = vit_outputs.last_hidden_state  # (B, 197, 768)
                x = x[:, 1:, :]  # 去掉 cls token

        else:
            x = self.net(x)
            x = x.to(self.device)
            x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
            x = x.view(x.size(0), -1, x.size(-1))  # (B, seq_len, C)
        return x



    def set_finetune(self):
        is_finetune = self.is_finetune
        if  is_finetune=='False':
            if self.network == 'resnet152':
                for param in self.net.parameters():
                    param.requires_grad = False
            elif self.network == 'clip_rn50':
                for param in self.clip_model.parameters():
                    param.requires_grad = False
         
            elif self.network == 'clip_vit':
                for param in self.clip_model.parameters():
                    param.requires_grad = False
            elif self.network == 'vit_base':
                for param in self.vit.parameters():
                    param.requires_grad = False

        elif is_finetune=='True':
            if self.network == 'resnet152':
                for name, param in self.net.named_parameters():
                    param.requires_grad = 'layer4' in name

            elif self.network == 'clip_rn50':
                for name, param in self.clip_model.named_parameters():
                    param.requires_grad = 'layer4' in name 
            
          
            elif self.network == 'clip_vit':
                for name, param in self.clip_model.named_parameters():
                    param.requires_grad = any(key in name for key in [
                        'transformer.resblocks.11', 'ln_post'
                    ])

            elif self.network == 'vit_base':
                for name, param in self.vit.named_parameters():
                    param.requires_grad = any(key in name for key in [
                        'encoder.layer.11', 'layernorm', 'ln_f', 'ln_post'
                    ])

    
    def get_finetune_state(self):

        if  self.is_finetune== 'False':
            return False
        elif  self.is_finetune=='True':
            return True
