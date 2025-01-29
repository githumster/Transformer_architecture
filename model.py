import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_seq_len=500, dropout_proba=0.1):

        super(PositionalEncoding, self).__init__()

        self.max_seq_len=max_seq_len

        self.d_model=d_model


        pe_table=self.get_pe_table()

        self.register_buffer('pe_table' , pe_table)

        self.dropout=nn.Dropout(dropout_proba)



    def get_pe_table(self):

        position_idxs=torch.arange(self.max_seq_len).unsqueeze(1)

        embedding_idxs=torch.arange(self.d_model).unsqueeze(0)


        angle_rads = position_idxs * 1/torch.pow(10000, (2*(embedding_idxs//2))/self.d_model)


        angle_rads[:, 0::2] = torch.sin(angle_rads[:, 0::2])

        angle_rads[:, 1::2] = torch.cos(angle_rads[:, 1::2])



        pe_table = angle_rads.unsqueeze(0) 



        return pe_table



    def forward(self, embeddings_batch):

        seq_len = embeddings_batch.size(1)

        pe_batch = self.pe_table[:, :seq_len].clone().detach()

        return self.dropout(embeddings_batch + pe_batch)
    


class AddAndNorm(nn.Module):

    def __init__(self, d_model):

        super(AddAndNorm, self).__init__()



        self.layer_norm=nn.LayerNorm(d_model)



    def forward(self, x, residual):

        return self.layer_norm(x+residual)



class PositionWiseFeedForwardNet(nn.Module):

    def __init__(self, d_model, d_ff):

        super(PositionWiseFeedForwardNet, self).__init__()

        self.w_1 = nn.Linear(d_model, d_ff)

        self.w_2 = nn.Linear(d_ff, d_model)




        self.dropout = nn.Dropout(0.1)



    def forward(self, x):

        return self.w_2(self.dropout(torch.relu(self.w_1(x))))
    

class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_head):
        super(ScaledDotProductAttention, self).__init__()

        self.d_head = d_head

        self.attention_dropout = nn.Dropout(p=0.1)

    def forward(self, q, k, v, mask=None):
        # q, k, v dims: (batch_size, n_heads, seq_len, d_head)

        attention_weights = torch.matmul(q, k.transpose(-2, -1))  
        scaled_attention_weights = attention_weights / math.sqrt(self.d_head)  

        if mask is not None:
            scaled_attention_weights = scaled_attention_weights.masked_fill(mask == 0, float('-inf')) # (batch_size, n_heads, seq_len, seq_len)

       
        scaled_attention_weights = nn.functional.softmax(scaled_attention_weights, dim=-1) # (batch_size, n_heads, seq_len, seq_len)

      
        scaled_attention_weights = self.attention_dropout(scaled_attention_weights) # (batch_size, n_heads, seq_len, seq_len)

        weighted_v = torch.matmul(scaled_attention_weights, v) # (batch_size, n_heads, seq_len, d_head)

        return weighted_v
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.n_heads= n_heads

        assert d_model % n_heads == 0
        self.d_head = d_model // n_heads

        self.dot_product_attention_layer= ScaledDotProductAttention(self.d_head)

        self.W_0 = nn.Linear(d_model, d_model)

    def _split_into_heads(self, q,k,v):
        q= q.view(q.size(0), q.size(1), self.n_heads, self.d_head) # (batch_size, seq_len, n_heads, d_head)
        k= k.view(k.size(0), k.size(1), self.n_heads, self.d_head) # (batch_size, seq_len, n_heads, d_head)
        v= v.view(v.size(0), v.size(1), self.n_heads, self.d_head) # (batch_size, seq_len, n_heads, d_head)

        q= q.transpose(1,2) # (batch_size, n_heads, seq_len, d_head)
        k= k.transpose(1,2) # (batch_size, n_heads, seq_len, d_head)
        v= v.transpose(1,2) # (batch_size, n_heads, seq_len, d_head)

        return q,k,v

    def _concatenate_heads(self,attention_output):
        attention_output = attention_output.transpose(1,2).contiguous() # (batch_size, seq_len, n_heads, d_head)
        attention_output = attention_output.view(attention_output.size(0), attention_output.size(1), -1) # (batch_size, seq_len, n_heads * d_head)

        return attention_output

    def forward(self, q, k, v, mask=None):
        q,k,v= self._split_into_heads(q,k,v) # (batch_size, n_heads, seq_len, d_head)
        attention_output = self.dot_product_attention_layer(q, k, v, mask) # (batch_size, n_heads, seq_len, d_head)
        attention_output = self._concatenate_heads(attention_output) # (batch_size, seq_len, n_heads * d_head)

        attention_output = self.W_0(attention_output) # (batch_size, seq_len, d_model)

        return attention_output


class TransformerEncoderBlock(nn.Module):

    def __init__(self, d_model, n_heads, d_ff, dropout_proba):

        super(TransformerEncoderBlock, self).__init__()



        self.W_q = nn.Linear(d_model, d_model)

        self.W_k = nn.Linear(d_model, d_model)

        self.W_v = nn.Linear(d_model, d_model)



        self.mha_layer=MultiHeadAttention(d_model, n_heads)

        self.dropout_layer_1=nn.Dropout(dropout_proba)

        self.add_and_norm_layer_1 = AddAndNorm(d_model)



        self.ffn_layer = PositionWiseFeedForwardNet(d_model, d_ff)

        self.dropout_layer_2=nn.Dropout(dropout_proba)

        self.add_and_norm_layer_2 = AddAndNorm(d_model)



    def forward(self, x, mask):

        # x dims: (batch_size, src_seq_len, d_model)

        # mask dim: (batch_size, 1, 1, src_seq_len)



        q = self.W_q(x) # (batch_size, src_seq_len, d_model)

        k = self.W_k(x) # (batch_size, src_seq_len, d_model)

        v = self.W_v(x) # (batch_size, src_seq_len, d_model)



        mha_out = self.mha_layer(q, k, v, mask) # (batch_size, src_seq_len, d_model)

        mha_out= self.dropout_layer_1(mha_out) # (batch_size, src_seq_len, d_model)

        mha_out = self.add_and_norm_layer_1(x, mha_out) # (batch_size, src_seq_len, d_model)



        ffn_out = self.ffn_layer(mha_out) # (batch_size, src_seq_len, d_model)

        ffn_out= self.dropout_layer_2(ffn_out) # (batch_size, src_seq_len, d_model)

        ffn_out = self.add_and_norm_layer_2(mha_out, ffn_out)  # (batch_size, src_seq_len, d_model)



        return ffn_out
    


class TransformerEncoder(nn.Module):

    def __init__(self, n_blocks, n_heads, d_model, d_ff, dropout_proba=0.1):

        super(TransformerEncoder, self).__init__()



        self.encoder_blocks=nn.ModuleList([TransformerEncoderBlock(d_model, n_heads, d_ff, dropout_proba) for _ in range(n_blocks)])



    def forward(self, x, mask):

        for encoder_block in self.encoder_blocks:

            x = encoder_block(x, mask)

        return x
    

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout_proba):
        super(TransformerDecoderBlock, self).__init__()

        self.W_q_1 = nn.Linear(d_model, d_model)
        self.W_k_1 = nn.Linear(d_model, d_model)
        self.W_v_1 = nn.Linear(d_model, d_model)

        self.mha_layer_1=MultiHeadAttention(d_model, n_heads)
        self.dropout_layer_1=nn.Dropout(dropout_proba)
        self.add_and_norm_1 = AddAndNorm(d_model)

        self.W_q_2 = nn.Linear(d_model, d_model)
        self.W_k_2 = nn.Linear(d_model, d_model)
        self.W_v_2 = nn.Linear(d_model, d_model)

        self.mha_layer_2=MultiHeadAttention(d_model, n_heads)
        self.dropout_layer_2=nn.Dropout(dropout_proba)
        self.add_and_norm_2 = AddAndNorm(d_model)

        self.ffn_layer = PositionWiseFeedForwardNet(d_model, d_ff)
        self.dropout_layer_3=nn.Dropout(dropout_proba)
        self.add_and_norm_3 = AddAndNorm(d_model)

    def forward(self, x, encoder_output, src_mask, trg_mask):
        # x dims: (batch_size, trg_seq_len, d_model)
        # encoder_output dims: (batch_size, src_seq_len, d_model)
        # src_mask dim: (batch_size, 1, 1, src_seq_len)
        # trg_mask dim: (batch_size, 1, trg_seq_len, trg_seq_len)

        # 1st attention layer, trg_mask is used here
        q_1 = self.W_q_1(x) # (batch_size, trg_seq_len, d_model)
        k_1 = self.W_k_1(x) # (batch_size, trg_seq_len, d_model)
        v_1 = self.W_v_1(x) # (batch_size, trg_seq_len, d_model)

        mha_layer_1_out = self.mha_layer_1(q_1, k_1, v_1, trg_mask) # (batch_size, trg_seq_len, d_model)
        mha_layer_1_out= self.dropout_layer_1(mha_layer_1_out) # (batch_size, trg_seq_len, d_model)
        mha_layer_1_out = self.add_and_norm_1(mha_layer_1_out, x) # (batch_size, trg_seq_len, d_model)

        # 2nd attention layer, src_mask is used here
        q_2 = self.W_q_2(mha_layer_1_out) # (batch_size, trg_seq_len, d_model)
        k_2 = self.W_k_2(encoder_output) # (batch_size, src_seq_len, d_model)
        v_2 = self.W_v_2(encoder_output) # (batch_size, src_seq_len, d_model)

        mha_layer_2_out = self.mha_layer_2(q_2, k_2, v_2, src_mask) # (batch_size, trg_seq_len, d_model)
        mha_layer_2_out= self.dropout_layer_2(mha_layer_2_out) # (batch_size, trg_seq_len, d_model)
        mha_layer_2_out = self.add_and_norm_2(mha_layer_2_out, mha_layer_1_out) # (batch_size, trg_seq_len, d_model)

        # Position-wise feed forward
        ffn_out = self.ffn_layer(mha_layer_2_out) # (batch_size, trg_seq_len, d_model)
        ffn_out= self.dropout_layer_3(ffn_out) # (batch_size, trg_seq_len, d_model)
        ffn_out = self.add_and_norm_3(ffn_out, mha_layer_2_out) # (batch_size, trg_seq_len, d_model)

        return ffn_out


class TransformerDecoder(nn.Module):

    def __init__(self, n_blocks, n_heads, d_model, d_ff, dropout_proba):

        super(TransformerDecoder, self).__init__()



        self.decoder_blocks=nn.ModuleList([TransformerDecoderBlock(d_model, n_heads, d_ff, dropout_proba) for _ in range(n_blocks)])



    def forward(self, x, encoder_output, src_mask, trg_mask):

        for decoder_block in self.decoder_blocks:

            x = decoder_block(x, encoder_output, src_mask, trg_mask)

        return x
    


class TransformerEncoderDecoder(nn.Module):

    def __init__(self,d_model, n_blocks, src_vocab_size, trg_vocab_size, n_heads, d_ff, dropout_proba):

        super(TransformerEncoderDecoder, self).__init__()

        self.dropout_proba = dropout_proba

        self.d_model=d_model



        # Encoder part

        self.src_embedding = nn.Embedding(src_vocab_size, d_model)

        self.src_pos_embedding= PositionalEncoding(d_model)

        self.encoder= TransformerEncoder(n_blocks, n_heads, d_model, d_ff, dropout_proba)



        # Decoder part

        self.trg_embedding = nn.Embedding(trg_vocab_size, d_model)

        self.trg_pos_embedding= PositionalEncoding(d_model)

        self.decoder= TransformerDecoder(n_blocks, n_heads, d_model, d_ff, dropout_proba)



        # Linear mapping to vocab size

        self.linear = nn.Linear(d_model, trg_vocab_size)


        self.init_with_xavier()


        self.src_embedding.weight = self.trg_embedding.weight

        self.linear.weight = self.trg_embedding.weight



    def encode(self, src_token_ids, src_mask):

        # Encoder part

        src_embeddings = self.src_embedding(src_token_ids) * math.sqrt(self.d_model) # (batch_size, src_seq_len, d_model)

        src_embeddings = self.src_pos_embedding(src_embeddings) # (batch_size, src_seq_len, d_model)

        encoder_outputs = self.encoder(src_embeddings, src_mask) # (batch_size, src_seq_len, d_model)



        return encoder_outputs



    def decode(self, trg_token_ids, encoder_outputs, src_mask, trg_mask):

        # Decoder part

        trg_embeddings = self.trg_embedding(trg_token_ids) * math.sqrt(self.d_model) # (batch_size, trg_seq_len, d_model)

        trg_embeddings = self.trg_pos_embedding(trg_embeddings) # (batch_size, trg_seq_len, d_model)

        decoder_outputs = self.decoder(trg_embeddings, encoder_outputs, src_mask, trg_mask) # (batch_size, trg_seq_len, d_model)



        # Linear mapping to vocab size

        linear_out = self.linear(decoder_outputs) # (batch_size, trg_seq_len, trg_vocab_size)



        return linear_out



    def forward(self, src_token_ids, trg_token_ids, src_mask, trg_mask):



        encoder_outputs= self.encode(src_token_ids, src_mask) # (batch_size, src_seq_len, d_model)

        decoder_outputs= self.decode(trg_token_ids, encoder_outputs, src_mask, trg_mask) # (batch_size, trg_seq_len, d_model)



        return decoder_outputs



    def init_with_xavier(self):

        for name, p in self.named_parameters():

            if p.dim() > 1:

                nn.init.xavier_uniform_(p)


class MachineTranslationTransformer(nn.Module):

    def __init__(self, d_model,n_blocks,src_vocab_size,trg_vocab_size,n_heads,d_ff, dropout_proba):

        super(MachineTranslationTransformer, self).__init__()



        self.transformer_encoder_decoder=TransformerEncoderDecoder(

            d_model,

            n_blocks,

            src_vocab_size,

            trg_vocab_size,

            n_heads,

            d_ff,

            dropout_proba

        )



    def _get_pad_mask(self, token_ids, pad_idx=0):

        pad_mask= (token_ids != pad_idx).unsqueeze(-2) # (batch_size, 1, seq_len)

        return pad_mask.unsqueeze(1)



    def _get_lookahead_mask(self, token_ids):

        sz_b, len_s = token_ids.size()

        subsequent_mask = (1 - torch.triu(torch.ones((1, len_s, len_s), device=token_ids.device), diagonal=1)).bool()

        return subsequent_mask.unsqueeze(1)



    def forward(self, src_token_ids, trg_token_ids):




        trg_token_ids=trg_token_ids[:, :-1]



        src_mask = self._get_pad_mask(src_token_ids) # (batch_size, 1, 1, src_seq_len)

        trg_mask = self._get_pad_mask(trg_token_ids) & self._get_lookahead_mask(trg_token_ids)  # (batch_size, 1, trg_seq_len, trg_seq_len)



        return self.transformer_encoder_decoder(src_token_ids, trg_token_ids, src_mask, trg_mask)



    def preprocess(self, sentence, tokenizer):

        device = next(self.parameters()).device



        src_token_ids=tokenizer.encode(sentence).ids

        src_token_ids=torch.tensor(src_token_ids, dtype=torch.long).to(device)

        src_token_ids=src_token_ids.unsqueeze(0) 



        return src_token_ids



    def translate(self, sentence, tokenizer, max_tokens=100, skip_special_tokens=False):



        device = next(self.parameters()).device


        eos_id=tokenizer.token_to_id('[EOS]')

        bos_id=tokenizer.token_to_id('[BOS]')

        pad_id = tokenizer.token_to_id('[PAD]')


        src_token_ids=self.preprocess(sentence, tokenizer)



        trg_token_ids=torch.LongTensor([bos_id]).unsqueeze(0).to(device) # (1, 1)




        src_mask=self._get_pad_mask(src_token_ids) # (batch_size, src_seq_len)



        encoder_output=self.transformer_encoder_decoder.encode(src_token_ids, src_mask) # (batch_size, src_seq_len, d_model)



        while True:



            trg_mask=self._get_lookahead_mask(trg_token_ids)  

            decoder_output=self.transformer_encoder_decoder.decode(trg_token_ids, encoder_output, src_mask, trg_mask)



          
            softmax_output=nn.functional.log_softmax(decoder_output, dim=-1) # (batch_size, trg_seq_len, trg_vocab_size)

            softmax_output_last=softmax_output[:, -1, :] # (batch_size, trg_vocab_size)

            _, token_id=softmax_output_last.max(dim=-1) # (batch_size, trg_seq_len)


            if token_id.item() == eos_id or trg_token_ids.size(1) == max_tokens:

                trg_token_ids=torch.cat([trg_token_ids, token_id.unsqueeze(0)], dim=-1) # (batch_size, trg_seq_len+1)

                break



            trg_token_ids=torch.cat([trg_token_ids, token_id.unsqueeze(0)], dim=-1) # (batch_size, trg_seq_len+1)



        decoded_output=tokenizer.decode(trg_token_ids.squeeze(0).detach().cpu().numpy(), skip_special_tokens=skip_special_tokens)



        return decoded_output
    

