
import torch.nn as nn
import torch
import torch.nn.functional as F
import math

NUMERIC_TOKEN_OFFSET = 1000.0
FEED_FORWARD_DENSE = "dense"


class DenseFeedForward(nn.Module):
    """The publication model's two-layer GELU feed-forward sublayer."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.GELU(),
            nn.Linear(input_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


def validate_dense_state_dict(state_dict):
    """Reject checkpoints that are not from the published dense model."""
    keys = state_dict.keys()
    has_dense = any(".dense_ffn." in key for key in keys)
    if not has_dense:
        raise ValueError(
            "Checkpoint is not compatible with the published dense PadSpectra model"
        )
    return None


# attention mechanism
class Attention(nn.Module):
    def __init__(self, input_dim, heads_num, has_query_weight=True):
        super().__init__()
        self.input_dim = input_dim
        self.heads_num = heads_num
        self.head_dim = input_dim // heads_num

        if self.head_dim * heads_num != input_dim:
            raise ValueError(
                f"Embedding size needs to be divisible by heads_num {heads_num}"
            )

        self.getvalues = nn.Linear(input_dim, input_dim, bias=False)
        self.getkeys = nn.Linear(input_dim, input_dim, bias=False)

        if has_query_weight:
            self.getqueries = nn.Linear(input_dim, input_dim, bias=False)

        self.mh_l = nn.Linear(input_dim, input_dim, bias=False)
        self.has_query_weight = has_query_weight

    def forward(self, values_in, keys_in, query_in, mask_k=None, mask_q=None):

        # input size: (B, T, D)
        B = values_in.shape[0]
        T_value = values_in.shape[1]
        T_key = keys_in.shape[1]
        T_query = query_in.shape[1]

        values = self.getvalues(values_in)
        keys = self.getkeys(keys_in)
        if self.has_query_weight:
            queries = self.getqueries(query_in)
        else:
            queries = query_in

        values = values.reshape(B, T_value, self.heads_num, self.head_dim)
        keys = keys.reshape(B, T_key, self.heads_num, self.head_dim)
        queries = queries.reshape(B, T_query, self.heads_num, self.head_dim)

        energy = torch.einsum("bqhd,bkhd->bhqk", [queries, keys])

        # mask_k (B, T_key, 1) and mask_q (B, T_query, 1)
        if mask_k is not None and mask_q is not None:
            mask_energy = mask_q & mask_k.squeeze(2).unsqueeze(1)  # (B, T_query, T_key)
            mask_energy = mask_energy.unsqueeze(1)  # (B, 1, T_query, T_key)
            energy = energy.masked_fill(~mask_energy, -1e4)
        elif mask_k is not None and mask_q is None:
            mask_energy = mask_k.squeeze(2).unsqueeze(1).unsqueeze(1)  # (B, 1, 1, T_key)
            energy = energy.masked_fill(~mask_energy, -1e4)
        elif mask_k is None and mask_q is not None:
            mask_energy = mask_q.unsqueeze(1)  # (B, 1, T_query, 1)
            energy = energy.masked_fill(~mask_energy, -1e4)

        attention = torch.softmax(energy / math.sqrt(self.head_dim), dim=3)

        out = torch.einsum("bhqk,bkhd->bqhd", [attention, values]).reshape(
            B, T_query, self.heads_num * self.head_dim
        )
        out = self.mh_l(out)
        return out


class TransformerLayer(nn.Module):
    def __init__(
        self,
        input_dim,
        heads,
        out_dim=None,
        has_query_weight=True,
    ):
        super().__init__()
        # self.input_dim = input_dim
        if out_dim is None:
            out_dim = input_dim
        self.out_dim = out_dim
        self.norm0 = nn.LayerNorm(input_dim)
        self.attention = Attention(input_dim, heads, has_query_weight)
        self.norm1 = nn.LayerNorm(input_dim)
        self.dense_ffn = DenseFeedForward(input_dim, out_dim)

    def forward(self,key, query=None, value=None,  mask_k=None, mask_q=None):

        key_norm = self.norm0(key)
        if value is None:
            value_norm = key_norm
        else:
            value_norm = self.norm0(value)

        if query is None:
            query_norm = key_norm
        else:
            query_norm = self.norm0(query)

        if mask_k is not None and query is None:
            mask_q = mask_k
        attention = self.attention(value_norm, key_norm, query_norm, mask_k, mask_q)

        if mask_q is not None:
            attention = attention.masked_fill(~mask_q, 0)

        # residual connection
        if query is None:
            attention = attention + key
        else:
            attention = attention + query

        out = self.norm1(attention)
        out = self.dense_ffn(out)
        if self.out_dim == attention.shape[-1]:
            out = out + attention # residual connection

        if mask_q is not None:
            out = out.masked_fill(~mask_q, 0)
        return out

class WordEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size, pad_index):
        super().__init__()
        self.pad_index = pad_index
        self.vocab_emb = nn.Embedding(vocab_size, embed_size, padding_idx=pad_index)
        self.float_proj = nn.Linear(1, embed_size)  # Project float to embed space
        self.number_index = vocab_size  # Used only to mask for floats

    def forward(self, inputwords):
        numeric_mask = inputwords < -(NUMERIC_TOKEN_OFFSET / 2)
        mask = (inputwords != self.pad_index)

        inputwords_clipped = inputwords.clone()
        inputwords_clipped[numeric_mask] = 0  # dummy index for floats

        word_emb = self.vocab_emb(inputwords_clipped.long())  # (B, T, D)

        # Runtime context encodes a standardized number z as -z-offset so
        # numerical tokens cannot collide with categorical indices. Projecting
        # all two-token entries and selecting with ``where`` avoids a
        # GPU-to-CPU synchronization from ``numeric_mask.any()`` every forward.
        float_values = (-inputwords - NUMERIC_TOKEN_OFFSET).unsqueeze(-1)
        float_emb = self.float_proj(float_values).to(dtype=word_emb.dtype)
        word_emb = torch.where(numeric_mask.unsqueeze(-1), float_emb, word_emb)

        return word_emb, mask


class WordPosEncoding(nn.Module):
    def __init__(self, embed_size, maxwords):
        super().__init__()
        self.embed_size = embed_size

        # create constant 'pe' matrix with values dependant on pos and i
        pe = torch.zeros(maxwords, embed_size)
        for pos in range(maxwords):
            for i in range(0, embed_size, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / embed_size)))
                if i + 1 < embed_size:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * (i + 1)) / embed_size)))

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, inputwords):
        # make embeddings relatively larger
        input_embedding = inputwords * math.sqrt(self.embed_size)
        # add constant to embedding
        input_embedding = input_embedding + self.pe[:, :inputwords.size(1), :]
        # input_embedding = input_embedding * mask.unsqueeze(2)
        return input_embedding


class PropEncoder(nn.Module):
    def __init__(
        self,
        embedsize,
        headnum,
        vocab_size,
        pad_index,
        maxwords=20,
    ):
        super().__init__()

        self.embedding = WordEmbedding(vocab_size, embedsize, pad_index)
        self.wposencoding = WordPosEncoding(embedsize, maxwords)

        # word encoding
        self.token_w0 = nn.Parameter(torch.ones(1, 1, embedsize))
        self.transformer_w1 = TransformerLayer(
            embedsize,
            headnum,
            has_query_weight=False,
        )

    def forward(self, inputwords):
        # word embedding
        embeddedwords, mask = self.embedding(inputwords)

        # get the shape of the input
        B = embeddedwords.shape[0]
        T_s = embeddedwords.shape[1]
        T_w = embeddedwords.shape[2]
        embeddedwords = embeddedwords.reshape(B * T_s, T_w, -1)

        mask_kw = mask.reshape(B * T_s, T_w, -1)  # key mask for word
        mask_ks = (mask.sum(dim=2) != 0).unsqueeze(2)  # key mask for sentence
        mask_qw = mask_ks.reshape(B * T_s, 1, 1)  # query mask for word

        # position encoding for word
        embeddedwords = self.wposencoding(embeddedwords)

        # Pool each property's word tokens into one context embedding.
        token_w0 = self.token_w0.expand(B * T_s, -1, -1)
        outtrans_w1 = self.transformer_w1(embeddedwords, token_w0, None, mask_kw, mask_qw)
        outtextembedding = outtrans_w1.reshape(B, T_s, -1)

        return outtextembedding

class SpecEncoder(nn.Module):
    def __init__(self, bandwidth, bandnum):
        super().__init__()

        self.bandwidth = bandwidth
        self.bandnum = bandnum

        # Embedding layers
        self.specposencoding = nn.Parameter(torch.rand(1, bandnum, bandwidth))

    def forward(self, spectra):
        # spectra shape: (batch, bandnum, bandwidth)
        specembedding = spectra + self.specposencoding  # broadcast positional encoding

        return specembedding

class AutoEncoder(nn.Module):
    def __init__(
        self,
        headnum,
        vocab_size,
        pad_index,
        bandwidth=50,
        bandnum=43,
        maxwords=20,
        hidden_dim=180,
    ):
        super().__init__()
        self.projectin = nn.Linear(bandwidth, hidden_dim, bias=False)
        self.specencoder = SpecEncoder(hidden_dim, bandnum)
        self.propencoder = PropEncoder(
            hidden_dim,
            headnum,
            vocab_size,
            pad_index,
            maxwords,
        )

        self.transformer0 = TransformerLayer(
            hidden_dim, headnum, hidden_dim
        )
        self.transformer1 = TransformerLayer(
            hidden_dim, headnum, hidden_dim
        )
        self.transformer2 = TransformerLayer(
            hidden_dim, headnum, hidden_dim
        )
        self.transformer3 = TransformerLayer(
            hidden_dim, headnum, hidden_dim
        )
        self.transformer4 = TransformerLayer(
            hidden_dim, headnum, hidden_dim
        )
        self.transformer5 = TransformerLayer(
            hidden_dim, headnum, bandwidth
        )

        self.convsmooth = nn.Sequential(
            nn.Conv1d(1, hidden_dim//4, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(hidden_dim//4, 1, kernel_size=5, padding=2)
        )

        self.bandwidth = bandwidth

    def forward(self, spectra, propdata):
        # input shape: (batch, bandnum, bandwidth)
        # get mask
        mask = (spectra == 0).any(dim=2)  # Shape: (batch, bandnum)
        mask = mask.unsqueeze(2)  # Shape: (batch, bandnum, 1)

        # spectra shape: (batch, bandnum, bandwidth)
        spectra_input = self.projectin(spectra)  # Project spectra to hidden dimension (batch, bandnum, hidden_dim)

        # encode spectra
        specembedding = self.specencoder(spectra_input)

        # encode properties
        propembedding = self.propencoder(propdata)

        # Concatenate the embeddings
        allembedding = torch.cat((specembedding, propembedding), dim=1)
        spec_mask = torch.ones(
            spectra.shape[0], spectra.shape[1], 1, dtype=torch.bool, device=spectra.device
        )
        prop_mask = (propembedding != 0).any(dim=2, keepdim=True)
        mask_embedding = torch.cat((spec_mask, prop_mask), dim=1)

        out_trans0 = self.transformer0(allembedding, specembedding, None, mask_embedding) + specembedding
        allembedding = torch.cat((out_trans0, propembedding), dim=1)
        out_trans1 = self.transformer1(allembedding, out_trans0, None, mask_embedding) + out_trans0
        allembedding = torch.cat((out_trans1, propembedding), dim=1)
        out_trans2 = self.transformer2(allembedding, out_trans1, None, mask_embedding) + out_trans1
        allembedding = torch.cat((out_trans2, propembedding), dim=1)
        out_trans3 = self.transformer3(allembedding, out_trans2, None, mask_embedding) + out_trans2
        allembedding = torch.cat((out_trans3, propembedding), dim=1)
        out_trans4 = self.transformer4(allembedding, out_trans3, None, mask_embedding) + out_trans3
        allembedding = torch.cat((out_trans4, propembedding), dim=1)
        out_spectra = self.transformer5(allembedding, out_trans4, None, mask_embedding)
        smooth_in = out_spectra.reshape(out_spectra.shape[0], -1).unsqueeze(1)
        out_spectra = self.convsmooth(smooth_in)
        out_spectra = out_spectra.reshape(out_spectra.shape[0], spectra.shape[1], -1)

        return out_spectra, mask

    def pad(self, spectra, propdata, smoothing = True):
        pad_spectra, mask_pad = self.forward(spectra, propdata)
        pad_spectra = pad_spectra.to(dtype=spectra.dtype)
        device_input = spectra.device
        mask_pad = mask_pad.repeat(1,1,self.bandwidth)
        mask_pad_left = torch.cat((mask_pad[:,1:,:],mask_pad[:,-1:,:]),dim=1)
        mask_pad_right = torch.cat((mask_pad[:,:1,:],mask_pad[:,:-1,:]),dim=1)
        mask_pad_broad = mask_pad_left | mask_pad_right
        pad_spectra[spectra>0] = spectra[spectra>0]
        if smoothing:
            windowsize = 15
            smoothrange = 7
            pad = windowsize//2
            conv_kernel = torch.ones(1, 1, windowsize) / windowsize
            conv_kernel = conv_kernel.to(device_input)
            spectra_to_smooth = torch.cat((pad_spectra[:,:-1,-smoothrange-pad:],pad_spectra[:,1:,:smoothrange + pad]),dim=2)
            spectra_smoothed = F.conv1d(spectra_to_smooth, conv_kernel.expand(spectra_to_smooth.size(1), -1, -1), groups=spectra_to_smooth.size(1))
            pad_spectra[:,:-1,-smoothrange:] = spectra_smoothed[:,:,:smoothrange]
            pad_spectra[:,1:,:smoothrange] = spectra_smoothed[:,:,-smoothrange:]

        pad_spectra[~mask_pad_broad] = spectra[~mask_pad_broad]

        return pad_spectra
