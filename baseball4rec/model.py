import logging
import torch
import copy
import torch.nn as nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)


class Baseball4Rec(nn.Module):
    def __init__(
        self,
        n_pitcher_cont,
        pitcher_len,
        n_batter_cont,
        batter_len,
        n_state_cont,
        n_encoder_layer,
        n_decoder_layer,
        n_concat_layer,
        num_y_pitching=144,
        d_model=64, 
        d_att=50, 
        nhead=2,
        dim_feedforward=256, 
        dropout=0.1,
    ):
        """
        Args:
            n_pitcher_cont, n_batter_cont, n_state_cont: the number of each continuous input features (required).
            n_encoder_layer: the number of encoder layers (required).
            n_decoder_layer: the number of decoder layers (required).
            n_concat_layer: the number of concat layers (required).
            num_y_pitching: the number of pitching type (default=9).
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (default=2).
            dim_feedforward: the dimension of the feedforward network model (default=128).
            dropout: the dropout value (default=0.1).
        """
        super(Baseball4Rec, self).__init__()
        
        self.d_model=d_model
        self.num_y_pitching = num_y_pitching

        self.pitcher_embedding = IntegratedEmbedding(d_model, n_pitcher_cont, 2, 2, 2) #n_pitcher_cont -> 4
        self.pitcher_id_embedding = nn.Embedding(num_embeddings = pitcher_len, embedding_dim = d_model)
        
        self.batter_embedding = IntegratedEmbedding(d_model, n_batter_cont, 2, 2) #n_batter_cont -> 2
        self.batter_id_embedding = nn.Embedding(num_embeddings = batter_len, embedding_dim = d_model)
        self.state_embedding = IntegratedEmbedding(d_model, n_state_cont, 10, 8, 9)

        
        self.ball_embedding = nn.Embedding(144, d_model) #아이템 임베딩차원만 두배
        #self.b = nn.Linear(d_model, d_model*2, bias=False)


        self.att_w = nn.Linear(d_model, d_att)
        self.att_h = nn.Linear(d_att, 1, bias=False)
        
        self.att_w2 = nn.Linear(d_model, d_att)
        self.att_h2 = nn.Linear(d_att, 1, bias=False)
        self.att_w3 = nn.Linear(d_model, d_att)
        self.att_h3 = nn.Linear(d_att, 1, bias=False)
        
        self.concat_layers = nn.ModuleList([])
        for layer in range(n_concat_layer):
            new_layer = TransformerEncoderBlock(d_model, nhead, dim_feedforward, dropout)
            self.concat_layers.append(new_layer)

        self.dropout = nn.Dropout(dropout)
        # 분류기용
        self.dropout_pitcher = nn.Dropout(dropout)
        self.pitching_classifier = nn.Linear(d_model, self.num_y_pitching)


    

    def forward(
        self,
        pitcher_id,
        pitcher_discrete,
        pitcher_continuous,
        
        batter_id,
        batter_discrete,
        batter_continuous,
        
        state_discrete, 
        state_continuous,
        pitch_mask,
    ):
        pitcher_id = self.pitcher_id_embedding(pitcher_id)
        batter_id = self.batter_id_embedding(batter_id) # 투수와 타자 id에 대한 임베딩 

        
        pitcher_x = self.pitcher_embedding(pitcher_discrete, pitcher_continuous) 
        batter_x = self.batter_embedding(batter_discrete, batter_continuous)
        
        
        state_x = self.state_embedding(state_discrete, state_continuous)

        concat_x = torch.cat([pitcher_x, batter_x, state_x], dim=0) # n_feat x 50 x d_model  50배치 d_model차원
        for layer in self.concat_layers:  # 셀프어탠션 전체 태움 
            concat_x = layer(concat_x)    # F x d
            
        
        
        ## 피쳐끼리 어탠션 후, 투수id 타자id와 다시 어탠션
        concat_x = self.dropout(concat_x)
        att=self.att_h2(self.att_w2(concat_x))
        att_weight2 = torch.nn.functional.softmax( att, dim=0) # 155 x batch_size x 1
        feature_representation2 = (att_weight2.view(-1,1,33)).bmm(concat_x.view(-1,33,self.d_model)) # batch_size x 1 x d_model  연속형 변수 9,9 개로 줄인거 42 -> 33(mask뺌)
        
        #concat_y = torch.cat([pitcher_id.unsqueeze(0), batter_id.unsqueeze(0), feature_representation2.view(1,-1,self.d_model)], dim=0) # 3 x batch_size x d_model 
        concat_y = torch.cat([pitcher_id.unsqueeze(0), feature_representation2.view(1,-1,self.d_model)], dim=0) # 2 x batch_size x d_model           ##타자뺀거


        att_weight3 = torch.nn.functional.softmax(self.att_h3(self.att_w3(concat_y)) , dim=0) # 3 x batch_size x 1
        #final_representation = (att_weight3.view(-1,1,3)).bmm(concat_y.view(-1,3,self.d_model)) # batch_size x 1 x d_model 
        final_representation = (att_weight3.view(-1,1,2)).bmm(concat_y.view(-1,2,self.d_model)) # batch_size x 1 x d_model                            ##타자뺀거
        
        
        
        
        
        
        
         
        #  구종별 임베딩 생성 / inner product 후 결과 산출
        item_embs = self.ball_embedding(torch.arange(144))
        scores = torch.matmul(final_representation, item_embs.transpose(0,1)) # 50 x 1 x 9
        
        scores = scores.squeeze(1)
        if pitch_mask is not None:
            scores = scores.masked_fill(pitch_mask, -1e5)


        
        return scores, att_weight2, att_weight3
    
       
    
    



class IntegratedEmbedding(nn.Module):
    def __init__(self, d_model, n_continuous_feature, *n_each_discrete_feature):
        super(IntegratedEmbedding, self).__init__()
        self.cont_emb = ContinuousEmbedding(n_continuous_feature, d_model)
        self.disc_emb = DiscreteEmbedding(d_model, *n_each_discrete_feature)

    def forward(self, x_disc, x_cont):
        x_disc = self.disc_emb(x_disc)
        x_cont = self.cont_emb(x_cont)
        return torch.cat((x_disc, x_cont), dim=1).transpose(1, 0)


class ContinuousEmbedding(nn.Module):
    def __init__(self, n_feature, d_model):
        """ 
        Project scalar feature to vector space by matrix multiplication
        (N x n_feature) -> (N x n_feature x d_model)    
        """
        super(ContinuousEmbedding, self).__init__()
        self.weights = nn.Parameter(torch.randn((n_feature, d_model)))
    def forward(self, x):
        
        return x.unsqueeze(-1).mul(self.weights)


class DiscreteEmbedding(nn.Module):
    def __init__(self, d_model, *num_embedding_features):
        """ 
        Project Discrete feature to vector space by Embedding layers
        (N x n_feature) -> (N x n_feature x d_model)    
        
        Args:
            num_embedding_features : The number of unique values of each discrete features
        """
        super(DiscreteEmbedding, self).__init__()
        self.embedding_layers = nn.ModuleList([])
        for i in range(len(num_embedding_features)):
            new_layer = nn.Embedding(num_embedding_features[i], d_model)
            self.embedding_layers.append(new_layer)

    def forward(self, x):
        out = []
        for i in range(len(self.embedding_layers)):
            out_ = self.embedding_layers[i](x[:, i])  # N x d
            out.append(out_)  # N x d
        return torch.stack(out, dim=1)  # N x n_feature x d









# 셀프 어텐션을 위한 블락
class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward, dropout):
        """
        Args:
            d_model: the number of expected features in the input (required).
            nhead: the number of heads in the multiheadattention models (required).
            dim_feedforward: the dimension of the feedforward network model (required).
            dropout: the dropout value (required).
        """
        super(TransformerEncoderBlock, self).__init__()
        self.transformer = TransformerBlock(d_model, nhead, dropout)
        self.feedforward = FeedForwardBlock(d_model, dim_feedforward, dropout)

    def forward(self, x, x_key_padding_mask=None, x_attn_mask=None):
        """
        x : input of the encoder layer
        """
        x = self.transformer(x, x, x, key_padding_mask=x_key_padding_mask, attn_mask=x_attn_mask)
        x = self.feedforward(x)
        return x





class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dropout):
        super(TransformerBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key, value, key_padding_mask=None, attn_mask=None):
        x = self.self_attn(
            query, key, value, key_padding_mask=key_padding_mask, attn_mask=attn_mask
        )[0]
        x = query + self.dropout(x)
        x = self.norm(x)
        return x


class FeedForwardBlock(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout):
        super(FeedForwardBlock, self).__init__()
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout1 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.dropout2 = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x2 = self.linear2(self.dropout1(F.relu(self.linear1(x))))
        x = F.relu(x + self.dropout2(x2))
        x = self.norm(x)
        return x
