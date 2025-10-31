
import os
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "3"  # or "1" if you want blocking behavior for debugging
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = "max_split_size_mb:128"
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch
import tarfile
import os
# cwd = os.getcwd()
import pandas as pd
from pathlib import Path  # Add this import statement

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the new CSV file
df = pd.read_csv("/home/ali/storage1/Athar/Colab_new_files/Biobert_Dec/new_dataset.csv")
df.head()

# Rename columns to match the expected format
dataFrame = df[["image_id", "caption"]]
dataFrame.rename(columns={'image_id': 'image', 'caption': 'caption'}, inplace=True)
dataFrame.head()

# Construct full image paths
base_path = Path("/home/ali/storage1/Athar/XrayGPT/dataset/openi/image/")
dataFrame['image'] = dataFrame['image'].apply(
    lambda x: str(base_path / f"{x}.png")  # Append .png to the image_id
)

dataFrame.head()


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from timm import create_model, list_models
from types import SimpleNamespace
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForMaskedLM
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.cuda.amp import GradScaler, autocast
from tqdm.auto import tqdm
import gc
import json


import copy
import torch
import torchvision
from torch import nn
from lightly.loss import DINOLoss
from lightly.models.modules import DINOProjectionHead
from lightly.models.utils import deactivate_requires_grad, update_momentum
from lightly.transforms.dino_transform import DINOTransform
from lightly.utils.scheduler import cosine_schedule
from tqdm import tqdm
from ultralytics import YOLO
from ultralytics.nn.modules import Conv
from torchvision import datasets, transforms

sample_tfms = [
    A.HorizontalFlip(),
    A.RandomBrightnessContrast(),
    A.ColorJitter(),
    A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.3, rotate_limit=45, p=0.5),
    A.HueSaturationValue(p=0.3),
]
train_tfms = A.Compose([
    *sample_tfms,
    A.Resize(224,224),
    A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],always_apply=True),
    ToTensorV2()
])
valid_tfms = A.Compose([
    A.Resize(224,224),
    A.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5],always_apply=True),
    ToTensorV2()
])

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # Set pad token if not already defined
tokenizer.pad_token

train_df, val_df = train_test_split(dataFrame,test_size=0.1)
train_df.reset_index(drop=True,inplace=True)
val_df.reset_index(drop=True,inplace=True)
print(len(train_df),len(val_df))

class Dataset:
    def __init__(self, df, tfms):
        self.df = df
        self.tfms = tfms

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        sample = self.df.iloc[idx, :]
        image = sample['image']
        caption = sample['caption']
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        augs = self.tfms(image=image)
        image = augs['image']
        caption = f"{caption}<|endoftext|>"  # Add end-of-text token

        # Tokenize the caption using ClinicalBERT tokenizer
        inputs = tokenizer(
            caption,
            truncation=True,
            return_tensors='pt'
        )
        input_ids = inputs['input_ids'].squeeze(0)  # Remove batch dimension

        # Shift labels for language modeling
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:]

        return image, input_ids, labels
    
train_ds = Dataset(train_df,train_tfms)
val_ds = Dataset(val_df,valid_tfms)

def collate_fn(batch):
    images = [i[0] for i in batch]
    input_ids = [i[1] for i in batch]
    labels = [i[2] for i in batch]

    # Stack images
    images = torch.stack(images, dim=0)

    # Pad input_ids and labels
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids, batch_first=True, padding_value=tokenizer.pad_token_id
    )
    labels = torch.nn.utils.rnn.pad_sequence(
        labels, batch_first=True, padding_value=tokenizer.pad_token_id
    )

    # Create attention mask
    mask = (input_ids != tokenizer.pad_token_id).long()
    labels[mask == 0] = -100  # Ignore padding tokens in the loss calculation

    return images, input_ids, labels

dl = torch.utils.data.DataLoader(train_ds,shuffle=True,batch_size=2,collate_fn=collate_fn)
_,c,l = next(iter(dl))
print(c[0])
print(l[0])

class GPT2Attention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        self.c_attn = nn.Linear(self.embed_dim, self.head_size * self.n_heads * 3,bias=True)
        self.scale = self.head_size ** -0.5

        self.register_buffer('mask',torch.tril(torch.ones(1,1,self.seq_len,self.seq_len)))

        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)

        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)


    def forward(self, x):
        b,t,c = x.shape
        # q,k,v shape individually: batch_size x seq_len x embed_dim
        # we know that qk_t = q x k_t, where q=bxtxhead_dim, k_t=bxhead_timxt
        q,k,v = self.c_attn(x).chunk(3,dim=-1)
        q = q.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3) # batch x n_heads x seq_len x head_dim
        k = k.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3)
        v = v.view(b,t,self.n_heads,self.head_size).permute(0,2,1,3)

        qk_t = (q@k.transpose(-2,-1)) * self.scale
        qk_t = qk_t.masked_fill(self.mask[:,:,:t,:t]==0,float('-inf'))
        qk_t = F.softmax(qk_t,dim=-1)
        weights = self.attn_dropout(qk_t)

        attention = weights @ v # batch x n_heads x t x head_size
        attention = attention.permute(0,2,1,3).contiguous().view(b,t,c) # batch x t x embed_dim

        out = self.c_proj(attention)
        out = self.resid_dropout(out)

        return out
    
class GPT2CrossAttention(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.n_heads = config.num_heads
        assert self.embed_dim % self.n_heads == 0, 'embedding dimension by be divisible by number of heads'
        self.head_size = self.embed_dim // self.n_heads
        self.seq_len = config.seq_len

        # Add input projection layer
        self.input_proj = nn.Linear(2048, self.embed_dim)  # Project from DINO output dim to embed_dim
        
        self.q = nn.Linear(self.embed_dim, self.embed_dim)
        self.k = nn.Linear(self.embed_dim, self.embed_dim)  
        self.v = nn.Linear(self.embed_dim, self.embed_dim)
        self.scale = self.head_size ** -0.5

        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.attn_dropout = nn.Dropout(config.attention_dropout)
        self.resid_dropout = nn.Dropout(config.residual_dropout)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
                
        # Apply weight initialization after defining the method
        self.apply(self._init_weights)

    def forward(self, q, k, v):
        b,t,c = q.shape
        
        # Project inputs if needed
        if k.size(-1) != self.embed_dim:
            k = self.input_proj(k)
        if v.size(-1) != self.embed_dim:
            v = self.input_proj(v)

        q = self.q(q)
        k = self.k(k)
        v = self.v(v)
        
        q = q.view(b,q.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
        k = k.view(b,k.size(1),self.n_heads,self.head_size).permute(0,2,1,3)
        v = v.view(b,v.size(1),self.n_heads,self.head_size).permute(0,2,1,3)

        qk_t = (q@k.transpose(-2,-1)) * self.scale
        qk_t = F.softmax(qk_t,dim=-1)
        weights = self.attn_dropout(qk_t)

        attention = weights @ v
        attention = attention.permute(0,2,1,3).contiguous().view(b,t,c)

        out = self.c_proj(attention)
        out = self.resid_dropout(out)
        return out
    
class GPT2MLP(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.mlp_ratio = config.mlp_ratio
        self.mlp_dropout = config.mlp_dropout

        self.c_fc = nn.Linear(self.embed_dim,self.embed_dim*self.mlp_ratio)
        self.c_proj = nn.Linear(self.embed_dim*self.mlp_ratio,self.embed_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(self.mlp_dropout)

    def forward(self,x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x
class GPT2Block(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.embed_dim = config.embed_dim
        self.ln_1 = nn.LayerNorm(self.embed_dim)
        self.attn = GPT2Attention(config)
        self.ln_2 = nn.LayerNorm(self.embed_dim)
        self.mlp = GPT2MLP(config)
        self.ln_3 = nn.LayerNorm(self.embed_dim)
        self.cross_attn = GPT2CrossAttention(config)

    def forward(self,x,enc_out):
        x = x+self.attn(self.ln_1(x)) # Attention
        x = x+self.cross_attn(self.ln_2(x),enc_out,enc_out) # Cross Attention
        x = x+self.mlp(self.ln_3(x))
        return x
    
class PoolHead(nn.Module):
    def __init__(self, f, i, c1):
        super().__init__()
        self.f = f
        self.i = i
        self.conv = Conv(c1, 1280, 1, 1, None, 1)  # Keep output as 1280
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # Add a projection layer to match embedding dimension
        self.proj = nn.Linear(1280, 1280)  # Project to match embed_dim
    
    def forward(self, x):
        x = self.avgpool(self.conv(x))
        x = x.flatten(1)
        return self.proj(x)

class DINOViTBackbone(nn.Module):
    def __init__(self, embed_dim=768):  # ViT output dimension
        super().__init__()
        self.backbone = create_model('deit3_base_patch16_224', pretrained=True)
        self.backbone.head = nn.Identity()  # Remove classifier head

        # Add a projection layer to map ViT output to embed_dim
        self.proj = nn.Linear(768, embed_dim)

        self.student_head = DINOProjectionHead(
            embed_dim, 512, 64, 2048, freeze_last_layer=1
        )
        self.teacher_head = DINOProjectionHead(embed_dim, 512, 64, 2048)
        deactivate_requires_grad(self.teacher_head)

    def forward(self, x):
        y = self.backbone(x)  # Shape: [batch_size, 768]
        y = self.proj(y)  # Project to embed_dim
        y = y.unsqueeze(1)  # Reshape to [batch_size, 1, embed_dim]
        z = self.student_head(y)
        return z

    def forward_teacher(self, x):
        y = self.backbone(x)
        y = self.proj(y)  # Project to embed_dim
        y = y.unsqueeze(1)  # Reshape to [batch_size, 1, embed_dim]
        z = self.teacher_head(y)
        return z
    


class BEiTGPT2Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Initialize DINO ViT backbone
        self.dino = DINOViTBackbone(embed_dim=config.embed_dim).to(device)
        
        # Initialize GPT-2 components
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.embed_dim),
            wpe = nn.Embedding(config.seq_len, config.embed_dim),
            drop = nn.Dropout(config.emb_dropout),
            h = nn.ModuleList([GPT2Block(config) for _ in range(config.depth)]),
            ln_f = nn.LayerNorm(config.embed_dim)
        ))

        self.lm_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight

    def forward_backbone(self, x):
        # Extract image features using DINO
        return self.dino(x)

    def forward_teacher(self, x):
        return self.dino.forward_teacher(x)
    
    def unfreeze_gpt_layers(self):
        """Unfreeze GPT-2 layers for fine-tuning"""
        for i in range(self.config.depth):
            for layer in [
                self.transformer.h[i].ln_1,
                self.transformer.h[i].ln_2,
                self.transformer.h[i].ln_3,
                self.transformer.h[i].attn,
                self.transformer.h[i].mlp,
                self.transformer.h[i].cross_attn
            ]:
                if not isinstance(layer, nn.Parameter):
                    for param in layer.parameters():
                        param.requires_grad = True
                else:
                    layer.requires_grad = True
                    
        # Also unfreeze final layer norm and head
        for param in self.transformer.ln_f.parameters():
            param.requires_grad = True
        for param in self.lm_head.parameters():
            param.requires_grad = True

    def pretrained_layers_trainable(self, trainable=False):
        # Freeze or unfreeze the DINO backbone and its heads
        for layer in [
            self.dino.backbone, self.dino.student_head, self.dino.teacher_head,
            self.transformer.wte, self.transformer.wpe,
            self.transformer.ln_f, self.lm_head
        ]:
            if not isinstance(layer, nn.Parameter):
                for param in layer.parameters():
                    param.requires_grad = trainable
            else:
                layer.requires_grad = trainable

        # Freeze or unfreeze GPT-2 layers
        for i in range(self.config.depth):
            for layer in [
                self.transformer.h[i].ln_1, self.transformer.h[i].ln_2,
                self.transformer.h[i].attn, self.transformer.h[i].mlp
            ]:
                if not isinstance(layer, nn.Parameter):
                    for param in layer.parameters():
                        param.requires_grad = trainable
                else:
                    layer.requires_grad = trainable

    def forward(self, image, input_ids, labels=None):
        # Extract image features using DINO
        image_features = self.forward_backbone(image)
        
        # Process text tokens
        token_embeddings = self.transformer.wte(input_ids)
        pos_embs = torch.arange(0, input_ids.size(1)).to(input_ids.device)
        positional_embeddings = self.transformer.wpe(pos_embs)
        hidden_states = self.transformer.drop(token_embeddings + positional_embeddings)

        # Cross-attention between image and text features
        for i in range(self.config.depth):
            hidden_states = self.transformer.h[i](hidden_states, image_features)

        hidden_states = self.transformer.ln_f(hidden_states)

        if labels is not None:
            lm_logits = self.lm_head(hidden_states)
            loss = F.cross_entropy(lm_logits.view(-1, lm_logits.shape[-1]), labels.view(-1))
            return loss

        lm_logits = self.lm_head(hidden_states[:, [-1], :])
        return lm_logits

    def generate(self, image, sequence, max_tokens=50, temperature=1.0, deterministic=False):
        for _ in range(max_tokens):
            out = self(image, sequence)
            out = out[:, -1, :] / temperature
            probs = F.softmax(out, dim=-1)
            if deterministic:
                next_token = torch.argmax(probs, dim=-1, keepdim=True)
            else:
                next_token = torch.multinomial(probs, num_samples=1)
            sequence = torch.cat([sequence, next_token], dim=1)
            if next_token.item() == tokenizer.eos_token_id:
                break
        return sequence.cpu().flatten()
    
class Trainer:
    def __init__(self, model_config, train_config, dls):
        self.train_config = train_config
        self.model_config = model_config
        self.device = self.train_config.device

        # Initialize the model directly
        self.model = BEiTGPT2Model(model_config).to(self.device)
        self.model.pretrained_layers_trainable(trainable=False)

        print(f'trainable parameters: {sum([p.numel() for p in self.model.parameters() if p.requires_grad])}')

        # Use ClinicalBERT tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.scaler = GradScaler()

        self.train_dl, self.val_dl = dls

        total_steps = len(self.train_dl)

        self.optim = torch.optim.Adam(self.model.parameters(), lr=self.train_config.lr / 25.)
        self.sched = torch.optim.lr_scheduler.OneCycleLR(
            self.optim,
            max_lr=self.train_config.lr,
            epochs=self.train_config.epochs,
            steps_per_epoch=total_steps
        )

        self.metrics = pd.DataFrame()
        self.metrics[['train_loss', 'train_perplexity', 'val_loss', 'val_perplexity']] = None

        self.gen_tfms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
            ToTensorV2()
        ])


    def save_model(self,):
        self.train_config.model_path.mkdir(exist_ok=True)
        sd = self.model.state_dict()
        torch.save(self.model,self.train_config.model_path/'Deit_Dino.pt')


    def load_best_model(self,):
        sd = torch.load(self.train_config.model_path/'Deit_Dino.pt')
        self.model.load_state_dict(sd)


    def train_one_epoch(self,epoch):

        prog = tqdm(self.train_dl,total=len(self.train_dl))

        running_loss = 0.

        for image, input_ids, labels in prog:

            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image,input_ids,labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optim)
                self.scaler.update()
                self.sched.step()
                self.optim.zero_grad(set_to_none=True)

                running_loss += loss.item()

                prog.set_description(f'train loss: {loss.item():.3f}')

            del image, input_ids, labels, loss

        train_loss = running_loss / len(self.train_dl)
        train_pxp = np.exp(train_loss)

        self.metrics.loc[epoch,['train_loss','train_perplexity']] = (train_loss,train_pxp)


    @torch.no_grad()
    def valid_one_epoch(self,epoch):

        prog = tqdm(self.val_dl,total=len(self.val_dl))

        running_loss = 0.

        for image, input_ids, labels in prog:

            with autocast():
                image = image.to(self.device)
                input_ids = input_ids.to(self.device)
                labels = labels.to(self.device)

                loss = self.model(image,input_ids,labels)
                running_loss += loss.item()

                prog.set_description(f'valid loss: {loss.item():.3f}')

            del image, input_ids, labels, loss

        val_loss = running_loss / len(self.val_dl)
        val_pxp = np.exp(val_loss)

        self.metrics.loc[epoch,['val_loss','val_perplexity']] = (val_loss,val_pxp)

        return val_pxp


    def clean(self):
        gc.collect()
        torch.cuda.empty_cache()


    def fit(self,):

        best_pxp = 1e9
        best_epoch = -1
        prog = tqdm(range(self.train_config.epochs))

        for epoch in prog:

            if epoch == self.train_config.freeze_epochs_gpt:
                self.model.unfreeze_gpt_layers()
                print('unfreezing GPT2 entirely...')

            if epoch == self.train_config.freeze_epochs_all:
                self.model.pretrained_layers_trainable(trainable=True)

            self.model.train()
            prog.set_description('training')
            self.train_one_epoch(epoch)
            self.clean()

            self.model.eval()
            prog.set_description('validating')
            pxp = self.valid_one_epoch(epoch)
            self.clean()

            print(self.metrics.tail(1))

            if pxp < best_pxp:
                best_pxp = pxp
                best_epoch = epoch
                print('saving best model...')
                self.save_model()

        return {
            'best_perplexity': best_pxp,
            'best_epoch': best_epoch
        }


    @torch.no_grad()
    def generate_caption(image, max_tokens=200, temperature=1.0, deterministic=False):
        gen_tfms = A.Compose([
            A.Resize(224, 224),
            A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
            ToTensorV2()
        ])
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        image = gen_tfms(image=image)['image']
        image = image.unsqueeze(0).to(device)  # Move the input image tensor to the same device as the model
        
        # Check if bos_token_id is None and handle it
        if tokenizer.bos_token_id is None:
            # Use cls_token_id or pad_token_id as a fallback
            sequence = torch.ones(1, 1).long().to(device) * (tokenizer.cls_token_id or tokenizer.pad_token_id)
        else:
            sequence = torch.ones(1, 1).long().to(device) * tokenizer.bos_token_id

        # Generate the caption using your model's specific method
        caption = model.generate(
            image,
            sequence,
            max_tokens=max_tokens,
            temperature=temperature,
            deterministic=deterministic
        )
        caption = tokenizer.decode(caption.cpu().numpy(), skip_special_tokens=True)  # Move the generated caption back to CPU for decoding

        return caption
    
model_config = SimpleNamespace(
    vocab_size = tokenizer.vocab_size,  # Use ClinicalBERT's vocabulary size
    embed_dim = 768,  # ViT base model's embedding dimension
    num_heads = 12,  # ViT base uses 12 heads
    seq_len = 1024,
    depth = 12,
    attention_dropout = 0.1,
    residual_dropout = 0.1,
    mlp_ratio = 4,
    mlp_dropout = 0.1,
    emb_dropout = 0.1,
)


train_config = SimpleNamespace(
    epochs = 25,
    freeze_epochs_gpt = 1,
    freeze_epochs_all = 2,
    lr = 4e-4,
    device = 'cuda',
    model_path = Path("/home/ali/storage1/Athar/Colab_new_files/Biobert_Dec/ClinicalBERT/Trained_Model/"),
    batch_size = 32
)
train_dl = torch.utils.data.DataLoader(train_ds,batch_size=train_config.batch_size,shuffle=True,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)
val_dl = torch.utils.data.DataLoader(val_ds,batch_size=train_config.batch_size,shuffle=True,pin_memory=True,num_workers=2,persistent_workers=True,collate_fn=collate_fn)
trainer = Trainer(model_config,train_config,(train_dl,val_dl))

trainer.fit()
trainer.metrics

import torch
# Load the model and move it to the appropriate device
model = torch.load("/home/ali/storage1/Athar/Colab_new_files/Biobert_Dec/ClinicalBERT/Trained_Model/Deit_Dino.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction


def generate_caption(image, max_tokens=200, temperature=1.0, deterministic=False):
    gen_tfms = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], always_apply=True),
        ToTensorV2()
    ])
    image = Image.open(image).convert('RGB')
    image = np.array(image)
    image = gen_tfms(image=image)['image']
    image = image.unsqueeze(0).to(device)  # Move the input image tensor to the same device as the model
    
    # Check if bos_token_id is None and handle it
    if tokenizer.bos_token_id is None:
        # Use cls_token_id or pad_token_id as a fallback
        sequence = torch.ones(1, 1).long().to(device) * (tokenizer.cls_token_id or tokenizer.pad_token_id)
    else:
        sequence = torch.ones(1, 1).long().to(device) * tokenizer.bos_token_id

    # Generate the caption using your model's specific method
    caption = model.generate(
        image,
        sequence,
        max_tokens=max_tokens,
        temperature=temperature,
        deterministic=deterministic
    )
    caption = tokenizer.decode(caption.cpu().numpy(), skip_special_tokens=True)  # Move the generated caption back to CPU for decoding

    return caption
# Placeholder lists for storing generated and ground truth captions
test_cap = []
gen_cap = []

# Generate captions and collect them in the lists
for i in tqdm(range(600), desc="Processing elements", unit="element"):
    det = False
    test = val_df.sample(n=1).values[0]
    test_img, test_caption = test[0], test[1]
    
    t = np.random.uniform(0.5, 1.5)  # Vary the temperature randomly
    if i > 40:
        det = True
    
    gen_caption = generate_caption(test_img, temperature=t, deterministic=det)
    test_cap.append([test_caption])  # Ground truth should be wrapped as a list of references
    gen_cap.append(gen_caption)

# Define BLEU score weights
w1 = (1.0, 0, 0, 0)  # BLEU-1
w2 = (0.5, 0.5, 0, 0)  # BLEU-2
w3 = (0.33, 0.33, 0.33, 0)  # BLEU-3
w4 = (0.25, 0.25, 0.25, 0.25)  # BLEU-4

# Calculate BLEU scores
def calculate_bleu_evaluation(GT_sentences, predicted_sentences):
    smooth = SmoothingFunction().method1  # Optional smoothing to avoid BLEU score of 0 for short sentences
    BLEU_1 = corpus_bleu(GT_sentences, predicted_sentences, weights=w1, smoothing_function=smooth)
    BLEU_2 = corpus_bleu(GT_sentences, predicted_sentences, weights=w2, smoothing_function=smooth)
    BLEU_3 = corpus_bleu(GT_sentences, predicted_sentences, weights=w3, smoothing_function=smooth)
    BLEU_4 = corpus_bleu(GT_sentences, predicted_sentences, weights=w4, smoothing_function=smooth)

    return BLEU_1, BLEU_2, BLEU_3, BLEU_4

# Compute BLEU scores
bleu1_scores, bleu2_scores, bleu3_scores, bleu4_scores = calculate_bleu_evaluation(test_cap, gen_cap)

# Print BLEU scores
print("BLEU-1 score:", bleu1_scores)
print("BLEU-2 score:", bleu2_scores)
print("BLEU-3 score:", bleu3_scores)
print("BLEU-4 score:", bleu4_scores)

from rouge import Rouge

def calculate_custom_rouge_l_evaluation(GT_sentences, predicted_sentences):
    rouge = Rouge()

    # Join the list of reference captions in test_cap to form a single string for each sample
    GT_sentences = [' '.join(references) if isinstance(references, list) else references for references in GT_sentences]
    
    scores = rouge.get_scores(predicted_sentences, GT_sentences, avg=True)

    return scores['rouge-l']['r']  # You can adjust this to return other metrics like 'p' (precision) or 'f' (F1-score)

# Assuming gen_cap and test_cap are your generated and ground truth captions
custom_rouge_l_score = calculate_custom_rouge_l_evaluation(test_cap, gen_cap)
print("Custom Rouge-L score:", custom_rouge_l_score)


from random import randint

test_case = 5
index = [randint(1, 99) for i in range(test_case)]

for i in range(len(index)):
    print(f"Test Case: {i + 1}")
    
    # Joining the list of ground truth captions (test_cap[index[i]]) into a single string
    original_caption = ' '.join(test_cap[index[i]]) if isinstance(test_cap[index[i]], list) else test_cap[index[i]]
    
    print("Original Findings : " + original_caption)
    print("Generated Findings : " + gen_cap[index[i]])
    print("\n\n")

# Tokenize captions for BLEU
references = [[ref.split()] for ref in test_cap]  # List of lists (each reference is a list of tokens)
hypotheses = [hyp.split() for hyp in gen_cap]     # List of generated captions (each is a list of tokens)

from rouge_score import RougeScorer  # Add this import at the top
from rouge import Rouge

# Your existing BLEU score calculation code remains the same...

# Modify the ROUGE calculation function
def calculate_rouge_scores(references, hypotheses):
    """
    Calculate ROUGE-1, ROUGE-2, and ROUGE-L scores.
    """
    scorer = RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []

    for ref, hyp in zip(references, hypotheses):
        # Convert list reference to string if needed
        if isinstance(ref, list):
            ref = ' '.join(ref)
        scores = scorer.score(ref, hyp)
        rouge_1_scores.append(scores['rouge1'].fmeasure)
        rouge_2_scores.append(scores['rouge2'].fmeasure)
        rouge_l_scores.append(scores['rougeL'].fmeasure)

    # Average scores
    rouge_1 = sum(rouge_1_scores) / len(rouge_1_scores)
    rouge_2 = sum(rouge_2_scores) / len(rouge_2_scores)
    rouge_l = sum(rouge_l_scores) / len(rouge_l_scores)

    return rouge_1, rouge_2, rouge_l

# Calculate ROUGE scores
references = [' '.join(ref) if isinstance(ref, list) else ref for ref in test_cap]
rouge_1, rouge_2, rouge_l = calculate_rouge_scores(references, gen_cap)

# Print all metrics in a nice format
print("\nMetrics Summary:")
print("-" * 40)
print(f"{'Metric':<15} {'Score':>10}")
print("-" * 40)
print(f"{'ROUGE-1':<15} {rouge_1:>10.4f}")
print(f"{'ROUGE-2':<15} {rouge_2:>10.4f}")
print(f"{'ROUGE-L':<15} {rouge_l:>10.4f}")