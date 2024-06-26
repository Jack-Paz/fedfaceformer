import os
import torch
import torch.nn as nn
import math
# from imitator.models.wav2vec import Wav2Vec2Model

from collections import defaultdict

from opacus.grad_sample import GradSampleModule
from opacus.layers.dp_multihead_attention import DPMultiheadAttention


class struct(object):
    def __init__(self, **kwargs):
        for key, val in kwargs.items():
            setattr(self, key, val)

def nonlinearity(x, activation="swish"):
    # swish
    if activation == "swish":
        x = x * torch.sigmoid(x)
    elif activation == "relu":
        x = torch.relu(x)
    return x

def get_inplace_activation(activation):

    if activation == "relu":
        return nn.ReLU(True)
    elif activation == "leakyrelu":
        return nn.LeakyReLU(0.01, True)
    elif activation == "swish":
        return nn.SiLU(True)
    elif activation == "sigmoid":
        return nn.Sigmoid()
    else:
        raise("Error: Invalid activation")

def Normalize(in_channels, norm="batch"):

    if norm == "batch":
         return torch.nn.BatchNorm1d(num_features=in_channels, eps=1e-6, affine=True)
    if norm == "batchfalse":
         return torch.nn.BatchNorm1d(num_features=in_channels, eps=1e-6, affine=False)
    elif norm == "instance":
        return torch.nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, eps=1e-6, affine=True)
    elif norm == "instancefalse":
        return torch.nn.GroupNorm(num_groups=in_channels, num_channels=in_channels, eps=1e-6, affine=False)
    else:
        raise("Enter a valid norm")

def add_norm(norm:str, in_channels:int, enc_layers:list):
    if norm is not None:
        fn = Normalize(in_channels, norm)
        enc_layers.append(fn)

def add_activation(activation:str, layers:list):
    if activation is not None:
        layers.append(get_inplace_activation(activation))

def def_value():
    return None

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def enc_dec_mask(device, dataset, T, S):
    mask = torch.ones(T, S)
    if dataset == "BIWI":
        for i in range(T):
            mask[i, i*2:i*2+2] = 0
    elif dataset == "vocaset":
        for i in range(T):
            mask[i, i] = 0
    return (mask==1).to(device=device)

def casual_mask(n_head, max_seq_len, period):
    mask = (torch.triu(torch.ones(max_seq_len, max_seq_len)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    mask = mask.unsqueeze(0)
    mask = mask.repeat(n_head, 1,1)
    return mask

class motion_decoder(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 activation="leakyrelu",
                 num_dec_layers=1,
                 fixed_channel=True,
                 style_concat=False,
                 **ignore_args):
        super().__init__()
        """
        Progressivly grow the decoder
        """
        self.style_concat = style_concat
        if style_concat:
            in_channels = 2 * in_channels
        if num_dec_layers == 1:
            self.decoder = nn.Sequential()
            final_out_layer = GradSampleModule(nn.Linear(in_channels, out_channels))
            self.decoder.add_module("final_out_layer", final_out_layer)
        else:
            # decoder
            dec_layers = []
            if fixed_channel:
                ch = in_channels
                ch_multi = 1
            else:
                ch = (out_channels - in_channels) // 2**(num_dec_layers-1)
                ch_multi = 2

            dec_layers.append(GradSampleModule(nn.Linear(in_channels, ch)))  # linear
            add_activation(activation, dec_layers)

            for i in range(2, num_dec_layers):
                dec_layers.append(GradSampleModule(nn.Linear(ch, ch_multi*ch)))
                add_activation(activation, dec_layers)
                ch = ch_multi*ch

            decoder = nn.Sequential(*dec_layers)
            self.decoder = nn.Sequential(*decoder)

            final_out_layer = GradSampleModule(nn.Linear(ch, out_channels))
            self.decoder.add_module("final_out_layer", final_out_layer)


        self.init_weight()

    def init_weight(self):
        for name, param in self.decoder.named_parameters():
            if 'bias' in name or "final_out_layer" in name:
                nn.init.constant_(param, 0)
            else:
                nn.init.xavier_uniform_(param)


    def forward(self, gen_viseme_feat, style_emb):
        """
        gen_viseme_feat:  Bs x Nf x feature_dim
        style_emb : Bs x 1 X feature_dim
        """
        if self.style_concat:
            Bs, nf, featdim = gen_viseme_feat.shape
            style_emb = style_emb.repeat(1, nf, 1)
            vertice_out_w_style = torch.cat([gen_viseme_feat, style_emb], dim=-1)
        else:
            vertice_out_w_style = gen_viseme_feat + style_emb

        return self.decoder(vertice_out_w_style)

def swap_dims(x):
    #swap first and second dims for batch_first
    return x.reshape(x.shape[1], x.shape[0], x.shape[2])

class OpacusDecoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, activation=nn.functional.relu, layer_norm_eps=1e-5, batch_first=False, norm_first=False, bias=True, device=None, dtype=None):

        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.batch_first = batch_first
        self.self_attn = DPMultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=self.batch_first)
        self.multihead_attn = DPMultiheadAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=self.batch_first)
        # Implementation of Feedforward model
        self.linear1 = GradSampleModule(nn.Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs))
        self.dropout = nn.Dropout(dropout)
        self.linear2 = GradSampleModule(nn.Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs))

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = activation

    def forward(
        self,
        tgt,
        memory,
        tgt_mask= None,
        memory_mask= None,
        tgt_key_padding_mask= None,
        memory_key_padding_mask= None,
        tgt_is_causal=False,
        memory_is_causal= False):
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal)
            x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask, memory_is_causal)
            x = x + self._ff_block(self.norm3(x))
        else:
            x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask, memory_is_causal))
            x = self.norm3(x + self._ff_block(x))

        return x


    # self-attention block
    def _sa_block(self, x,
                attn_mask, key_padding_mask, is_causal: bool = False):
        # x = self.self_attn(x, x, x,
        #                 attn_mask=attn_mask,
        #                 key_padding_mask=key_padding_mask,
        #                 is_causal=is_causal,
        #                 need_weights=False)[0]
        # attn_mask = swap_dims(attn_mask) #does it need to be swapped?
        x = self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        # x = self.dropout1(x)
        return x

    # multihead attention block
    def _mha_block(self, x, mem,
                attn_mask, key_padding_mask, is_causal: bool = False):
        # x = self.multihead_attn(x, mem, mem,
        #                         attn_mask=attn_mask,
        #                         key_padding_mask=key_padding_mask,
        #                         is_causal=is_causal,
        #                         need_weights=False)[0]
        # attn_mask = swap_dims(attn_mask)
        # mem = swap_dims(mem)
        x = self.multihead_attn(x, mem, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask,need_weights=False)[0]
        return self.dropout2(x)

    # feed forward block
    def _ff_block(self, x):
        # if self.batch_first:
        #     x = swap_dims(x)
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        # if self.batch_first:
        #     x = swap_dims(x)
        return self.dropout3(x)


class imitator(nn.Module):
    def __init__(self, args):
        super(imitator, self).__init__()
        """
        audio: (batch_size, raw_wav)
        template: (batch_size, V*3)
        vertice: (batch_size, seq_len, V*3)
        """
        if isinstance(args, dict):
            args = struct(**args)

        self.train_subjects = args.train_subjects.split(" ")
        self.dataset = args.dataset
        # if args.dp=='opacus':
            
            # print('importing opacus wav2vec')
            # from imitator.models.opacus_wav2vec import Wav2Vec2Model
        # else:
            # from imitator.models.wav2vec import Wav2Vec2Model
        from imitator.models.wav2vec import Wav2Vec2Model

        if os.getenv('WAV2VEC_PATH'):
            wav2vec_path = os.getenv('WAV2VEC_PATH')
            self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_path)
        elif hasattr(args, 'wav2vec_model'):
            wav2vec_path = os.path.join(os.getenv('HOME'), args.wav2vec_model)
            self.audio_encoder = Wav2Vec2Model.from_pretrained(wav2vec_path)
        else:
            self.audio_encoder = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        # print('BREAKING THE CODE JUST TO TEST, DELET')
        # self.audio_encoder = GradSampleModule(nn.Linear(16000, args.feature_dim))
        if args.dp=='opacus':
            print('WARNING: FREEZING WAV2VEC')
            for param in self.audio_encoder.parameters():
                param.requires_grad = False #freeze wav2vec
        
        if not hasattr(args, 'max_seq_len'):
            args.max_seq_len = 600

        # self.audio_encoder.feature_extractor._freeze_parameters()
        self.audio_feature_map = GradSampleModule(nn.Linear(768, args.feature_dim))
        if hasattr(args, 'wav2vec_static_features'):
            self.audio_encoder.generate_static_audio_features = args.wav2vec_static_features # default value is false

        self.PPE = PositionalEncoding(args.feature_dim, max_len=args.max_seq_len)
        self.causal_mh_mask = casual_mask(n_head=32, max_seq_len=args.max_seq_len, period=args.max_seq_len)
        # this thing needs bsz * 4 heads, but batches are random sized, just make it 32 and hope bsz never greater than 8??
        decoder_layer = OpacusDecoderLayer(d_model=args.feature_dim, nhead=4, dim_feedforward=2*args.feature_dim, batch_first=True)
        
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=1)
        
        # style embedding
        self.obj_vector = GradSampleModule(nn.Linear(args.num_identity_classes, args.feature_dim, bias=False))
        self.transformer_ff_features = defaultdict(def_value)
        self.args = args

        self.vertice_map_r = motion_decoder(in_channels=args.feature_dim, out_channels=args.vertice_dim,
                                            num_dec_layers=args.num_dec_layers, fixed_channel=args.fixed_channel,
                                            style_concat=args.style_concat)

    def forward(self, audio, template, vertice, one_hot, criterion, teacher_forcing=True, test_dataset=None):
        
        self.device = audio.device
        if len(audio)==0:
            return torch.FloatTensor(0), [0] #happens during opacus training bc of random batches
        # tgt_mask: :math:`(T, T)`.
        # memory_mask: :math:`(T, S)`.
        template = template.unsqueeze(1)  # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)  # (1, feature_dim)
        frame_num = vertice.shape[1]
        hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
        # breakpoint()
        hidden_states = self.audio_feature_map(hidden_states)
    
        for i in range(frame_num):
            if i==0:
                vertice_emb = obj_embedding.unsqueeze(1) # (1,1,feature_dim)
                style_emb = vertice_emb
                start_token = torch.zeros_like(vertice_emb)
                vertice_input = self.PPE(start_token)
            else:
                vertice_input = self.PPE(vertice_emb)
    
            # get the masks
            tgt_mask = self.causal_mh_mask[:audio.shape[0]*4, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device) # JP added the bsz computation 
            memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
            gen_viseme_feat = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            new_output = gen_viseme_feat[:,-1,:].unsqueeze(1)
            vertice_emb = torch.cat((vertice_emb, new_output), 1)
    
        # add the style only to the decoder
        vertice_out_w_style = self.vertice_map_r(gen_viseme_feat, style_emb)
        vertice_out_w_style = vertice_out_w_style + template
        loss = criterion(vertice_out_w_style, vertice) # (batch, seq_len, V*3)
        loss = torch.mean(loss)

        # set the current epoch viseme feat
        self.batch_viseme_feats = gen_viseme_feat
        
        return loss, vertice_out_w_style

    def style_forward(self, audio, seq_name, template, vertice, one_hot, criterion, teacher_forcing=False):

        assert len(one_hot.shape) == 2

        self.device = audio.device
        train_from_scratch = teacher_forcing
        if self.transformer_ff_features[seq_name] is None or train_from_scratch:

            template = template.unsqueeze(1)  # (1,1, V*3) # (1, feature_dim)
            frame_num = vertice.shape[1]
            hidden_states = self.audio_encoder(audio, self.dataset, frame_num=frame_num).last_hidden_state
            hidden_states = self.audio_feature_map(hidden_states)

            ### run recursive model
            for i in range(frame_num):

                ## setup the starting frame
                if i == 0:
                    obj_embedding = self.obj_vector(one_hot)
                    vertice_emb = obj_embedding.unsqueeze(1)  # (1, 1, feature_dim)
                    style_emb = vertice_emb
                    start_token = torch.zeros_like(vertice_emb)
                    vertice_input = self.PPE(start_token)
                else:
                    vertice_input = self.PPE(vertice_emb)

                ## get the masks
                tgt_mask = self.causal_mh_mask[:audio.shape[0]*4, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device) # JP added the bsz computation 
                tgt_mask = tgt_mask.clone().detach().to(device=self.device)
                memory_mask = enc_dec_mask(self.device, self.dataset, vertice_input.shape[1], hidden_states.shape[1])
                ## transformer decoder
                gen_viseme_feat = self.transformer_decoder(vertice_input,hidden_states,tgt_mask=tgt_mask,memory_mask=memory_mask)
                new_output = gen_viseme_feat[:, -1, :].unsqueeze(1)
                vertice_emb = torch.cat((vertice_emb, new_output), 1)

            self.transformer_ff_features[seq_name] = gen_viseme_feat.detach()
        else:
            style_emb = self.obj_vector(one_hot).unsqueeze(1)  # (1, 1, feature_dim)

        ### extract the features
        gen_viseme_feat = self.transformer_ff_features[seq_name]
        vertice_out_w_style = self.vertice_map_r(gen_viseme_feat, style_emb)
        vertice_out_w_style = vertice_out_w_style + template

        ### compute the loss
        loss = criterion(vertice_out_w_style, vertice)  # (batch, seq_len, V*3)
        loss = torch.mean(loss)
        return loss, vertice_out_w_style

    def predict(self, audio, template, one_hot, test_dataset=None):

        test_dataset = self.dataset if test_dataset is None else test_dataset

        self.device = audio.device
        template = template.unsqueeze(1) # (1,1, V*3)
        obj_embedding = self.obj_vector(one_hot)

        hidden_states = self.audio_encoder(audio, test_dataset).last_hidden_state
        frame_num = hidden_states.shape[1]
        hidden_states = self.audio_feature_map(hidden_states)

        # generate the transformer features
        for i in range(frame_num):
            if i==0:
                vertice_emb = obj_embedding.unsqueeze(1)  # (1,1,feature_dim)
                style_emb = vertice_emb
                start_token = torch.zeros_like(vertice_emb)
                vertice_input = self.PPE(start_token)
            else:
                vertice_input = self.PPE(vertice_emb)

            # get the masks
            tgt_mask = self.causal_mh_mask[:audio.shape[0]*4, :vertice_input.shape[1], :vertice_input.shape[1]].clone().detach().to(device=self.device) # JP added the bsz computation 
            memory_mask = enc_dec_mask(self.device, test_dataset, vertice_input.shape[1], hidden_states.shape[1])
            # run the decoder
            gen_viseme_feat = self.transformer_decoder(vertice_input, hidden_states, tgt_mask=tgt_mask, memory_mask=memory_mask)
            # use the features without style as input to the network
            new_output = gen_viseme_feat[:,-1,:].unsqueeze(1)
            vertice_emb = torch.cat((vertice_emb, new_output), 1)

        # gen_viseme_feat = torch.rand_like(gen_viseme_feat)
        vertice_out_w_style = self.vertice_map_r(gen_viseme_feat, style_emb)
        vertice_out = vertice_out_w_style + template
        return vertice_out