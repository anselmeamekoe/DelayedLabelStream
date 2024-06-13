import torch.nn as nn
from einops import rearrange

class Vec2matEncoder(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_head=8,
        n_hidden_encoder=2,
        dropout_rate=0.0,
        activation = nn.Identity(), 
        bias = True
    ):
        super(Vec2matEncoder, self).__init__()
        # Dimensions
        self.dim_input = dim_input
         
        dim_output_encoder = dim_input*dim_head
        
        # hidden layer dimensions
        dim_inners = [max(dim_input,dim_output_encoder//2**(n_hidden_encoder-i)) for i in range(n_hidden_encoder)]
        
        # linear layers  
        self.inners_linear = nn.ModuleList([
            nn.Linear(dim_input,dim,bias = bias) if i==0 else nn.Linear(dim_inners[i-1],dim,bias = bias) for i,dim in enumerate(dim_inners)
        ]
        )
        self.out_linear = nn.Linear(dim_inners[-1], dim_output_encoder,bias = bias)
        
        # activation functions
        self.activation = activation
        self.sigmoid = nn.Sigmoid()
        
        # Dropout 
        self.dropout = nn.Dropout(dropout_rate)
    def forward(self, input_):
        
        for linear_layer in self.inners_linear:
            input_ = linear_layer(input_)
            input_ = self.activation(input_)
            input_ = self.dropout(input_)

        out = self.out_linear(input_)
        out = self.sigmoid(out)
        
        # reshape to have (batch, dim_input, d_k) d_k here is the dimension of each head
        out = rearrange(out,'b (h w) -> b h w', h = self.dim_input)
        return out

class SelfReinformentAttention(nn.Module):
    def __init__(
        self,
        dim_input,
        dim_head=8,
        n_hidden_encoder=2,
        dropout_rate=0.0,
        activation = nn.Identity(), 
        bias = True,
        no_interaction_indice= None,
        s = nn.Identity()
    ):
        super(SelfReinformentAttention, self).__init__()
        
        self.dim_input = dim_input
        self.d_k = dim_head
        self.scale = dim_head**-0.5
        
        self.activation = activation
        
        # Encoding
        self.encoder_q= Vec2matEncoder(dim_input = dim_input,
                                        dim_head=dim_head,
                                        n_hidden_encoder=n_hidden_encoder,
                                        dropout_rate=dropout_rate,
                                        activation = activation, 
                                        bias = bias
        )
        self.encoder_k= Vec2matEncoder(dim_input = dim_input,
                                        dim_head=dim_head,
                                        n_hidden_encoder=n_hidden_encoder,
                                        dropout_rate=dropout_rate,
                                        activation = activation, 
                                        bias = bias
        )
        self.s = s
        self.no_interaction_indice = no_interaction_indice
        
    def forward(self, input_):

        q = self.encoder_q(input_)
        k = self.encoder_k(input_)
        
        qk  = q*k*self.scale
        
        att = qk .sum(axis = -1)
        att = self.s(att)
        if self.no_interaction_indice:
            att[:, self.no_interaction_indice]=1
        return att