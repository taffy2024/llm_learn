#注意要传入的参数😓
import torch
from torch import nn
import torch.nn.functional as F
import math
X=torch.randn(128,64,512) #  随机生成一个形状为(batch=128,time=64,dimension=512)的三维张量
print(X.shape)

special_tokens = ["<UNK>", "<PAD>", "<SOS>", "<EOS>"]
vocab = {"<UNK>":0,"<PAD>":1,"<SOS>":2,"<EOS>":3}  #未知词，填充标记，序列开始，序列结束
d_model=512 #每个词元被表示成一个 512 维的向量
n_head=8 #注意力头数目
# 词嵌入层：将序列中的每个离散词元，分别转化为一个d_model维的向量，最终得到一个词向量序列
class TokenEmbedding(nn.Embedding):
    def __init__(self,vocab_size,d_model): #vocab_size:词汇表大小(不同的token总数)
        super(TokenEmbedding,self).__init__(vocab_size,d_model,padding_idx=1) 
# 输入形状为(batch_size,seq_len)的词元索引序列
# 输出形状为(batch_size,seq_len,d_model)的词向量序列
#padding_idx=1确保索引为1的填充符号始终映射为全 0 向量，避免其干扰模型对有效词元的学习。
class PositionalEmbedding(nn.Module):
    def __init__(self,d_model,maxlen,device):
        super(PositionalEmbedding,self).__init__()
        self.encoding=torch.zeros(maxlen,d_model,device=device) #创建一个全零的二维张量，形状为(maxlen,d_model)
        self.encoding.requires_grad_(False)  #不使用梯度
        
        # 生成位置索引(0到maxlen-1)
        pos=torch.arange(0,maxlen,device=device) #arange：生成等差数列的张量
        pos=pos.float().unsqueeze(1) #在第1维度插入新维度
        #生成索引维度(步长为2，对应偶数维度)
        _2i=torch.arange(0,d_model,2,device=device)
        # 偶数维度
        self.encoding[:,0::2]=torch.sin(pos/(10000**(_2i/d_model)))
        # 奇数维度
        self.encoding[:,1::2]=torch.cos(pos/(10000**(_2i/d_model)))

    def forward(self,x):
        seq_len=x.shape[1] # x 是词向量序列，形状为 (batch_size, seq_len, d_model)，取实际序列长度
        return self.encoding[:seq_len,:] # 返回前 seq_len 个位置的嵌入，形状为 (seq_len, d_model)
class TransformerEmbedding(nn.Module):
    def __init__(self,vocab_size,d_model,maxlen,drop_prob,device):
        super(TransformerEmbedding,self).__init__()
        self.token=TokenEmbedding(vocab_size,d_model)
        self.position=PositionalEmbedding(d_model,maxlen,device)
        self.drop_out=nn.Dropout(p=drop_prob) #训练时随机失活部分元素，防止过拟合

    def forward(self,x):
        token=self.token(x)
        position=self.position(x)
        return self.drop_out(token+position)
class LayerNorm(nn.Module):
    def __init__(self,d_model,eps=1e-10):
        super(LayerNorm,self).__init__()
        self.gamma=nn.Parameter(torch.ones(d_model))
        self.beta=nn.Parameter(torch.zeros(d_model))
        self.eps=eps

    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        var=x.var(-1,unbiased=False,keepdim=True)
        out=(x-mean)/torch.sqrt(var+self.eps)
        out=self.gamma*out+self.beta
        return out
class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, hidden, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, hidden)
        self.fc2 = nn.Linear(hidden, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
class Multi_head_attention(nn.Module):
    def __init__(self,d_model,n_head)->None:  #->表示函数无返回值  d_model:模型的隐藏层维度 n_head:注意力头的数量
        super(Multi_head_attention,self).__init__()
        # 将传入的参数保存为类属性，后续在forward中使用
        self.n_head=n_head
        self.d_model=d_model

        #创建三个线性变换层，将输入向量投影到QKV空间
        self.w_q=nn.Linear(d_model,d_model)
        self.w_k=nn.Linear(d_model,d_model)
        self.w_v=nn.Linear(d_model,d_model)

        # 所有头的输出拼接后仍为d_model
        self.w_combine=nn.Linear(d_model,d_model) #线性映射层
        self.softmax=nn.Softmax(dim=-1)

    def forward(self,q,k,v,mask=None):
        batch,time,dimension=q.shape
        n_d=self.d_model//self.n_head
        q,k,v=self.w_q(q),self.w_k(k),self.w_v(v)
        #把向量拆分成多个头(batch,time,dimension)->(batch,time,n_head,n_d)
        q=q.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        k=k.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
        v=v.view(batch,time,self.n_head,n_d).permute(0,2,1,3)
     
        score=q@k.transpose(2,3)/math.sqrt(n_d)
        if mask is not None:

            # mask=torch.tril(torch.ones(time,time,dtype=bool))
            score=score.masked_fill(mask==0,float("-inf"))

        score=self.softmax(score)@v

        score=score.permute(0,2,1,3).contiguous().view(batch,time,dimension)

        output=self.w_combine(score)
        return output
    
attention=Multi_head_attention(d_model,n_head)
output=attention(X,X,X)
print(output,output.shape)
class EncoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob)->None:
        super(EncoderLayer,self).__init__()
        self.attention=Multi_head_attention(d_model,n_head)
        self.norm1=LayerNorm(d_model)
        self.drop1=nn.Dropout(drop_prob)

        self.ffn=PositionwiseFeedForward(d_model,ffn_hidden,drop_prob)
        self.norm2=LayerNorm(d_model)
        self.drop2=nn.Dropout(drop_prob)

    def forward(self,x,mask=None):
        _x=x
        x=self.attention(x,x,x,mask)
        
        x=self.drop1(x)
        x=self.norm1(x+_x)
        
        _x=x
        x=self.ffn(x)

        x=self.drop2(x)        
        x=self.norm2(x+_x)
        return x
class DecoderLayer(nn.Module):
    def __init__(self,d_model,ffn_hidden,n_head,drop_prob):
        super(DecoderLayer,self).__init__()
        self.attention=Multi_head_attention(d_model,n_head)
        self.norm1=LayerNorm(d_model)
        self.dropout1=nn.Dropout(drop_prob)

        self.cross_attention=Multi_head_attention(d_model,n_head)
        self.norm2=LayerNorm(d_model)
        self.dropout2=nn.Dropout(drop_prob)

        self.ffn=PositionwiseFeedForward(d_model,ffn_hidden,drop_prob)
        self.norm3=LayerNorm(d_model)
        self.dropout3=nn.Dropout(drop_prob)

    def forward(self,dec,enc,time_mask,s_mask):
        _x=dec
        x=self.attention(dec,dec,dec,time_mask) # 下三角掩码（因果掩码）

        x=self.dropout1(x)
        x=self.norm1(x+_x)

        if enc is not None:
            _x=x
            x=self.cross_attention(x,enc,enc,s_mask) #位置掩码

            x=self.dropout2(x)
            x=self.norm2(x+_x)

        #前馈网络 + 残差连接 + 层归一化
        _x=x
        x=self.ffn(x)
        x=self.dropout3(x)
        x=self.norm3(x+_x)
        return x
class Encoder(nn.Module):
    def __init__(self,env_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,drop_prob,device):
        super(Encoder, self).__init__()

        self.embedding = TransformerEmbedding(
            env_voc_size, d_model, max_len, drop_prob, device
        )

        self.layers = nn.ModuleList(
            [
                EncoderLayer(d_model, ffn_hidden, n_head, drop_prob)
                for _ in range(n_layer)
            ]
        )

    def forward(self, x, s_mask):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x, s_mask)
        return x
class Decoder(nn.Module):
    def __init__(self, dec_voc_size,max_len,d_model,ffn_hidden,n_head,n_layer,drop_prob,device):
        super(Decoder,self).__init__()

        self.embedding=TransformerEmbedding(dec_voc_size, d_model, max_len, drop_prob, device)

        self.layers=nn.ModuleList(
            [DecoderLayer(d_model,ffn_hidden,n_head,drop_prob)for _ in range(n_layer)] 
        )
        self.fc=nn.Linear(d_model,dec_voc_size)

    def forward(self,dec,enc,time_mask,s_mask):
        dec=self.embedding(dec)
        for layer in self.layers:
            dec=layer(dec,enc,time_mask,s_mask)
        dec=self.fc(dec)
        return dec
class Transformer(nn.Module):
    def __init__(self, src_pad_idx,trg_pad_idx,enc_voc_size,dec_voc_size,max_len,d_model,n_head,ffn_hidden,n_layers,drop_prob,device):
        super(Transformer,self).__init__()

        self.encoder=Encoder(enc_voc_size,max_len,d_model,ffn_hidden,n_head,n_layers,drop_prob,device)
        self.decoder = Decoder(dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, drop_prob, device)


        self.src_pad_idx=src_pad_idx
        self.trg_pad_idx=trg_pad_idx
        self.device=device

    def make_pad_mask(self,q,k,pad_idx_q,pad_idx_k):
        len_q,len_k=q.size(1),k.size(1)

        # 维度(batch,time,len_q,len_k)
        q=q.ne(pad_idx_q).unsqueeze(1).unsqueeze(3)
        q=q.repeat(1,1,1,len_k)

        k=k.ne(pad_idx_k).unsqueeze(1).unsqueeze(3)
        k=k.repeat(1,1,len_q,1)

        mask=q&k  #全一则一
        return mask


    def make_causal_mask(self,q,k):
        len_q,len_k=q.size(1),k.size(1)
        mask=torch.tril(torch.ones(len_q,len_k)).type(torch.BoolTensor).to(self.device)
        return mask
    
    def forward(self,src,trg):
        src_mask=self.make_pad_mask(src,src,self.src_pad_idx,self.src_pad_idx)
        trg_mask=self.make_pad_mask(trg,trg,self.trg_pad_idx,self.trg_pad_idx)*self.make_causal_mask(trg,trg)
        src_trg_mask=self.make_pad_mask(trg,src,self.trg_pad_idx,self.src_pad_idx)

        enc=self.encoder(src,src_mask)
        output=self.decoder(trg,enc,trg_mask,src_trg_mask)
        return output