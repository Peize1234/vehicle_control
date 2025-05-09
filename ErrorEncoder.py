import torch
from torch import nn
import math


class PositionalEncoding(nn.Module):
    """
    compute sinusoid encoding.
    """

    def __init__(self, d_feature, max_len):
        """
        constructor of sinusoid encoding class

        :param d_feature: dimension of model
        :param max_len: max sequence length
        """
        super(PositionalEncoding, self).__init__()

        # same size with input matrix (for adding with input matrix)
        self.encoding = torch.zeros(max_len, d_feature)
        self.encoding.requires_grad = False  # we don't need to compute gradient

        pos = torch.arange(0, max_len)
        pos = pos.float().unsqueeze(dim=1)
        # 1D => 2D unsqueeze to represent word's position

        _2i = torch.arange(0, d_feature, step=2).float()
        # 'i' means index of d_model (e.g. embedding size = 50, 'i' = [0,50])
        # "step=2" means 'i' multiplied with two (same with 2 * i)

        self.encoding[:, 0::2] = torch.sin(pos / (10000 ** (_2i / d_feature)))
        self.encoding[:, 1::2] = torch.cos(pos / (10000 ** (_2i / d_feature)))
        # compute positional encoding to consider positional information of words

    def forward(self, x):
        # self.encoding
        # [max_len = 512, d_model = 512]

        batch_size, seq_len, _ = x.size()
        # [batch_size = 128, seq_len = 30]

        return self.encoding[:seq_len, :]
        # [seq_len = 30, d_model = 512]
        # it will add with tok_emb : [128, 30, 512]


class Embedding(nn.Module):
    def __init__(self, d_feature, max_seq_len, num_heads=8, dropout=0.1):
        super().__init__()

        self.dim_up = nn.Linear(d_feature, d_feature * num_heads)
        self.pose_embedder = PositionalEncoding(d_feature=d_feature * num_heads, max_len=max_seq_len)
        self.drop_out = nn.Dropout(p=dropout)

    def forward(self, x):
        # x : [batch_size, seq_len, d_feature]
        x = self.dim_up(x)
        # x : [batch_size, seq_len, d_feature * num_heads]

        pose_embedding = self.pose_embedder(x).to(x.device)
        # print(pose_embedding.shape)
        # x : [batch_size, seq_len, d_feature * num_heads]
        x = self.drop_out(x + pose_embedding[None, :, :])
        # x : [batch_size, seq_len, d_feature * num_heads]
        return x


class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, q, k, v, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = k.size()

        # 1. dot product Query with Key^T to compute similarity
        k_t = k.transpose(2, 3)  # transpose
        score = (q @ k_t) / math.sqrt(d_tensor)  # scaled dot product

        # 2. apply masking (opt)
        if mask is not None:
            score = score.masked_fill(mask == 0, -10000)

        # 3. pass them softmax to make [0, 1] range
        score = self.softmax(score)

        # 4. multiply with Value
        v = score @ v

        return v, score


class MultiHeadAttention(nn.Module):
    def __init__(self, d_feature, num_heads):
        super().__init__()

        self.num_heads = num_heads
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_feature, d_feature)
        self.w_k = nn.Linear(d_feature, d_feature)
        self.w_v = nn.Linear(d_feature, d_feature)
        self.w_concat = nn.Linear(d_feature, d_feature)

    def forward(self, q, k, v, mask=None):
        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)

        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map
        # TODO : we should implement visualization

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.num_heads
        tensor = tensor.view(batch_size, length, self.num_heads, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model, hidden, drop_prob=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, hidden)
        self.linear2 = nn.Linear(hidden, d_model)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=drop_prob)

    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class EncoderBlock(nn.Module):

    def __init__(self, input_dim, trans_dim, num_heads, drop_prob):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_feature=input_dim, num_heads=num_heads)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.trans_sample = nn.Sequential(
            nn.Linear(input_dim, trans_dim),
            nn.LeakyReLU(),
        )
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.norm2 = nn.LayerNorm(trans_dim)

    def forward(self, x, src_mask=None):
        # 1. compute self attention
        _x = x
        x = self.attention(q=x, k=x, v=x, mask=src_mask)

        # 2. add and norm
        x = self.dropout1(x)
        x = self.norm1(x + _x)

        # 3. positionwise feed forward network
        # _x = x
        x_trans = self.trans_sample(x)

        # 4. add and norm
        x_trans = self.dropout2(x_trans)
        x_trans = self.norm2(x_trans)
        return x, x_trans


class ErrorEncoder(nn.Module):
    def __init__(self, input_dim, out_dim, out_encoder_dim=1, seq_len=10, num_heads=8, width=1, drop_prob=0.1):
        super().__init__()

        # embedding layer
        self.embedder = Embedding(d_feature=input_dim, max_seq_len=seq_len, num_heads=num_heads * width,
                                  dropout=drop_prob)

        embedding_out_dim = input_dim * num_heads * width
        self.encoder_block1 = EncoderBlock(input_dim=embedding_out_dim, trans_dim=embedding_out_dim // 2,
                                           num_heads=num_heads, drop_prob=drop_prob)
        self.encoder_align1 = nn.Linear(embedding_out_dim, out_dim)

        self.encoder_block2 = EncoderBlock(input_dim=embedding_out_dim // 2, trans_dim=embedding_out_dim // 4,
                                           num_heads=num_heads // 2, drop_prob=drop_prob)
        self.encoder_align2 = nn.Linear(embedding_out_dim // 2, out_dim)

        self.encoder_block3 = EncoderBlock(input_dim=embedding_out_dim // 4, trans_dim=embedding_out_dim // 8,
                                           num_heads=num_heads // 4, drop_prob=drop_prob)
        self.encoder_align3 = nn.Linear(embedding_out_dim // 4, out_dim)

        self.encoder_block4 = EncoderBlock(input_dim=embedding_out_dim // 8, trans_dim=embedding_out_dim // 8,
                                           num_heads=num_heads // 8, drop_prob=drop_prob)
        self.encoder_align4 = nn.Linear(embedding_out_dim // 8, out_dim)

        self.out_encoder = nn.Sequential(
            nn.Linear(embedding_out_dim // 8, out_encoder_dim),
        )

        self.decoder_block1 = EncoderBlock(input_dim=embedding_out_dim // 8, trans_dim=embedding_out_dim // 4,
                                           num_heads=num_heads // 8, drop_prob=drop_prob)
        # self.dim_align1 = nn.Linear(embedding_out_dim // 8, out_dim)

        self.decoder_block2 = EncoderBlock(input_dim=embedding_out_dim // 4, trans_dim=embedding_out_dim // 2,
                                           num_heads=num_heads // 4, drop_prob=drop_prob)

        self.decoder_block3 = EncoderBlock(input_dim=embedding_out_dim // 2, trans_dim=embedding_out_dim,
                                           num_heads=num_heads // 2, drop_prob=drop_prob)

        self.decoder_block4 = EncoderBlock(input_dim=embedding_out_dim, trans_dim=embedding_out_dim,
                                           num_heads=num_heads, drop_prob=drop_prob)

        self.out_decoder = nn.Sequential(
            nn.Linear(embedding_out_dim, out_dim),
        )

    def forward(self, x):
        # x : [batch_size, seq_len, d_feature]
        x = self.embedder(x)
        # x : [batch_size, seq_len, d_feature * num_heads]

        # [batch_size, seq_len, input_dim], [batch_size, seq_len, trans_dim]
        out_encoder1, out_encoder_trans1 = self.encoder_block1(x)
        out_encoder2, out_encoder_trans2 = self.encoder_block2(out_encoder_trans1)
        out_encoder3, out_encoder_trans3 = self.encoder_block3(out_encoder_trans2)
        out_encoder4, out_encoder_trans4 = self.encoder_block4(out_encoder_trans3)

        out_supervise = self.out_encoder(out_encoder_trans4)
        out_supervise = torch.mean(out_supervise, dim=1)
        out_supervise = torch.sigmoid(out_supervise)

        # out_decoder4, out_decoder_trans4 = self.decoder_block1(out_encoder4)
        # out_decoder3, out_decoder_trans3 = self.decoder_block2(out_decoder_trans4 + out_encoder3)
        # out_decoder2, out_decoder_trans2 = self.decoder_block3(out_decoder_trans3 + out_encoder2)
        # out_decoder1, out_decoder_trans1 = self.decoder_block4(out_decoder_trans2 + out_encoder1)
        #
        # out_error_feature = self.out_decoder(out_decoder_trans1)
        # out_error_feature = torch.mean(out_error_feature, dim=1)

        out_error_feature = torch.mean(self.encoder_align1(out_encoder1), dim=1) + \
                            torch.mean(self.encoder_align2(out_encoder2), dim=1) + \
                            torch.mean(self.encoder_align3(out_encoder3), dim=1) + \
                            torch.mean(self.encoder_align4(out_encoder4), dim=1)

        return out_supervise, out_error_feature

    def freeze_encoder(self):
        for param in self.embedder.parameters():
            param.requires_grad = False
        for param in self.encoder_block1.parameters():
            param.requires_grad = False
        for param in self.encoder_block2.parameters():
            param.requires_grad = False
        for param in self.encoder_block3.parameters():
            param.requires_grad = False
        for param in self.encoder_block4.parameters():
            param.requires_grad = False
        for param in self.out_encoder.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.embedder.parameters():
            param.requires_grad = True
        for param in self.encoder_block1.parameters():
            param.requires_grad = True
        for param in self.encoder_block2.parameters():
            param.requires_grad = True
        for param in self.encoder_block3.parameters():
            param.requires_grad = True
        for param in self.encoder_block4.parameters():
            param.requires_grad = True
        for param in self.out_encoder.parameters():
            param.requires_grad = True

    def save_encoder_state_dict(self, file_name="encoder_state_dict.pth"):
        embedding_state_dict = self.embedder.state_dict()
        encoder_block_state_dict1 = self.encoder_block1.state_dict()
        encoder_block_state_dict2 = self.encoder_block2.state_dict()
        encoder_block_state_dict3 = self.encoder_block3.state_dict()
        encoder_block_state_dict4 = self.encoder_block4.state_dict()
        out_encoder_state_dict = self.out_encoder.state_dict()

        torch.save({
            "embedding_state_dict": embedding_state_dict,
            'encoder_block1': encoder_block_state_dict1,
            'encoder_block2': encoder_block_state_dict2,
            'encoder_block3': encoder_block_state_dict3,
            'encoder_block4': encoder_block_state_dict4,
            'out_encoder': out_encoder_state_dict
        }, f"./models/{file_name}")

    def load_encoder_state_dict(self, file_name="encoder_state_dict.pth"):
        state_dict = torch.load(f"./models/{file_name}")
        self.embedder.load_state_dict(state_dict['embedding_state_dict'])
        self.encoder_block1.load_state_dict(state_dict['encoder_block1'])
        self.encoder_block2.load_state_dict(state_dict['encoder_block2'])
        self.encoder_block3.load_state_dict(state_dict['encoder_block3'])
        self.encoder_block4.load_state_dict(state_dict['encoder_block4'])
        self.out_encoder.load_state_dict(state_dict['out_encoder'])

    def save_decoder_state_dict(self, file_name="decoder_state_dict.pth"):
        decoder_block_state_dict1 = self.decoder_block1.state_dict()
        decoder_block_state_dict2 = self.decoder_block2.state_dict()
        decoder_block_state_dict3 = self.decoder_block3.state_dict()
        decoder_block_state_dict4 = self.decoder_block4.state_dict()
        out_decoder_state_dict = self.out_decoder.state_dict()

        torch.save({
            'decoder_block1': decoder_block_state_dict1,
            'decoder_block2': decoder_block_state_dict2,
            'decoder_block3': decoder_block_state_dict3,
            'decoder_block4': decoder_block_state_dict4,
            'out_decoder': out_decoder_state_dict
        }, f"./models/{file_name}")

    def load_decoder_state_dict(self, file_name="decoder_state_dict.pth"):
        state_dict = torch.load(f"./models/{file_name}")
        self.decoder_block1.load_state_dict(state_dict['decoder_block1'])
        self.decoder_block2.load_state_dict(state_dict['decoder_block2'])
        self.decoder_block3.load_state_dict(state_dict['decoder_block3'])
        self.decoder_block4.load_state_dict(state_dict['decoder_block4'])
        self.out_decoder.load_state_dict(state_dict['out_decoder'])

    def save_all_state_dict(self, file_name="error_encoder_all_state_dict.pth"):
        torch.save(self.state_dict(), f"./models/{file_name}")

    def load_all_state_dict(self, file_name="error_encoder_all_state_dict.pth"):
        self.load_state_dict(torch.load(f"./models/{file_name}"))


if __name__ == '__main__':
    model = ErrorEncoder(input_dim=6, out_dim=20, out_encoder_dim=1, seq_len=10, num_heads=8, drop_prob=0.1)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    x = torch.randn(50, 10, 6).to(device)
    y = model(x)
    print(y[0].shape, y[1].shape)
    # print(y[0][0, :, 0])

    # model.load_encoder_state_dict()
    # model.eval()
