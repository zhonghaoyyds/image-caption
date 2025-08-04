import torch
import torch.nn as nn


class Attention(nn.Module):
    def __init__(self, encoder_dim, hidden_dim):
        super(Attention, self).__init__()
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(1)
        self.U = nn.Linear(hidden_dim, hidden_dim)
        self.W = nn.Linear(encoder_dim, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1)


    def forward(self, img_features, hidden_state):
   
        U_h = self.U(hidden_state).unsqueeze(1)
        W_s = self.W(img_features)
        att = self.tanh(W_s + U_h)
        e = self.v(att).squeeze(2)

        alpha = self.softmax(e)
        context = (img_features * alpha.unsqueeze(2)).sum(1)

        return context, alpha

