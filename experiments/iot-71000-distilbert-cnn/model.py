"""Explicit architecture hypotheses, NOT recovered author code.

The literal F->768 projection of Eq.4 produces one 768-wide token. A
multi-token interpretation requires an extra projection choice absent in the paper.
"""
import torch
from torch import nn
from transformers import DistilBertConfig,DistilBertModel


class DualHead(nn.Module):
    def __init__(self,n_features,attack_classes=5,variant='proposed',tokens=1,
                 initialization='pretrained',checkpoint='distilbert/distilbert-base-uncased'):
        super().__init__()
        if variant not in ['proposed','distilbert_only','cnn_only','without_attention','focal']:
            raise ValueError(variant)
        self.variant=variant;self.tokens=tokens
        self.projection=nn.Linear(n_features,tokens*768)
        if variant!='cnn_only':
            if initialization=='pretrained':self.encoder=DistilBertModel.from_pretrained(checkpoint)
            elif initialization=='random':self.encoder=DistilBertModel(DistilBertConfig(dropout=.3,attention_dropout=.3))
            else:raise ValueError(initialization)
            # Table IV's dropout is used throughout this independent implementation.
            for layer in self.encoder.modules():
                if isinstance(layer,nn.Dropout):layer.p=.3
            self.encoder_cnn1=nn.Sequential(nn.Conv1d(768,128,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
            self.encoder_cnn2=nn.Sequential(nn.Conv1d(768,128,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        self.raw_cnn=nn.Sequential(nn.Conv1d(1,128,5,padding=2),nn.ReLU(),
            nn.Conv1d(128,128,5,padding=2),nn.ReLU(),nn.AdaptiveAvgPool1d(1))
        dim=128 if variant=='cnn_only' else 768 if variant=='distilbert_only' else 1152
        # Number of fusion heads is not reported. Eight is an explicit assumption.
        self.fusion=nn.MultiheadAttention(dim,8,dropout=.3,batch_first=True)
        self.hidden=nn.Sequential(nn.Linear(dim,128),nn.ReLU(),nn.Dropout(.3))
        self.binary_head=nn.Linear(128,2);self.attack_head=nn.Linear(128,attack_classes)

    def forward(self,x):
        if self.variant=='cnn_only':features=self.raw_cnn(x.unsqueeze(1)).squeeze(-1)
        else:
            embedded=self.projection(x).reshape(len(x),self.tokens,768)
            # DistilBERT inputs_embeds adds its learned positional embeddings internally.
            h=self.encoder(inputs_embeds=embedded).last_hidden_state
            # First projected token acts as pooled token; no undocumented text tokenizer.
            if self.variant=='distilbert_only':features=h[:,0]
            else:features=torch.cat([h[:,0],self.encoder_cnn1(h.transpose(1,2)).squeeze(-1),
                self.encoder_cnn2(h.transpose(1,2)).squeeze(-1),self.raw_cnn(x.unsqueeze(1)).squeeze(-1)],dim=-1)
        if self.variant in ['proposed','focal']:
            features=self.fusion(features[:,None,:],features[:,None,:],features[:,None,:],need_weights=False)[0][:,0]
        hidden=self.hidden(features)
        return self.binary_head(hidden),self.attack_head(hidden)


def multitask_loss(binary_logits,attack_logits,binary_y,attack_y,weights,lam=.5,focal=False):
    binary=nn.functional.cross_entropy(binary_logits,binary_y)
    mask=binary_y.bool()
    if mask.any():
        per=nn.functional.cross_entropy(attack_logits[mask],attack_y[mask],weight=weights,reduction='none')
        if focal:
            # Focal gamma/alpha are unreported: gamma=2 and inverse-frequency weights assumed.
            p=attack_logits[mask].softmax(-1).gather(1,attack_y[mask,None]).squeeze(1)
            per=per*(1-p).square()
        attack=per.mean()
    else:attack=attack_logits.sum()*0
    return lam*binary+(1-lam)*attack


def predict_routed(model,x,benign_id,attack_label_ids):
    b,a=model(x);routed=b.argmax(-1).bool()
    pred=torch.full((len(x),),benign_id,dtype=torch.long,device=x.device)
    attack_label_ids=torch.as_tensor(attack_label_ids,device=x.device)
    pred[routed]=attack_label_ids[a[routed].argmax(-1)]
    # Joint probabilities for honest six-class ROC, rather than hard-label AUC.
    bp=b.softmax(-1);ap=a.softmax(-1);probs=torch.zeros((len(x),len(attack_label_ids)+1),device=x.device)
    probs[:,benign_id]=bp[:,0];probs[:,attack_label_ids]=bp[:,1,None]*ap
    return pred,probs
