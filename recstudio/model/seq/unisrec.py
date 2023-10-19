import torch
import torch.nn.functional as F
from typing import Dict
from recstudio.ann import sampler
from recstudio.data import dataset
from recstudio.model.module import functional as recfn
from recstudio.model import basemodel, loss_func, module, scorer
from recstudio.model.seq.sasrec import SASRec, SASRecQueryEncoder


class PWLayer(torch.nn.Module):
    """Single Parametric Whitening Layer
    """
    def __init__(self, input_size, output_size, dropout=0.0):
        super(PWLayer, self).__init__()

        self.dropout = torch.nn.Dropout(p=dropout)
        self.bias = torch.nn.Parameter(torch.zeros(input_size), requires_grad=True)
        self.lin = torch.nn.Linear(input_size, output_size, bias=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

    def forward(self, x):
        return self.lin(self.dropout(x) - self.bias)


class MoEAdaptorLayer(torch.nn.Module):
    """MoE-enhanced Adaptor
    """
    def __init__(self, n_exps, layers, dropout=0.0, noise=True):
        super(MoEAdaptorLayer, self).__init__()

        self.n_exps = n_exps
        self.noisy_gating = noise

        self.experts = torch.nn.ModuleList([PWLayer(layers[0], layers[1], dropout) for i in range(n_exps)])
        self.w_gate = torch.nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)
        self.w_noise = torch.nn.Parameter(torch.zeros(layers[0], n_exps), requires_grad=True)

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        clean_logits = x @ self.w_gate    
        if self.noisy_gating and train:
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = ((F.softplus(raw_noise_stddev) + noise_epsilon))
            noisy_logits = clean_logits + (torch.randn_like(clean_logits).to(x.device) * noise_stddev)
            logits = noisy_logits
        else:
            logits = clean_logits

        gates = F.softmax(logits, dim=-1)
        return gates

    def forward(self, x):
        gates = self.noisy_top_k_gating(x, self.training) # (B, n_E)
        expert_outputs = [self.experts[i](x).unsqueeze(-2) for i in range(self.n_exps)] # [(B, 1, D)]
        expert_outputs = torch.cat(expert_outputs, dim=-2)
        multiple_outputs = gates.unsqueeze(-1) * expert_outputs
        return multiple_outputs.sum(dim=-2)
    
class UnisRecItemEncoder(torch.nn.Module):
    def __init__(self, train_stage, embed_dim, num_items, feat_name, fiid, moe_adaptor
                 ) -> None:
        super().__init__()
        self.item_embedding = torch.nn.Embedding(num_items, embed_dim, padding_idx=0)
        self.train_stage = train_stage
        self.fiid = fiid
        self.feat_name = feat_name
        self.moe_adaptor = moe_adaptor


    def forward(self, batch, encode_feat_query = False, need_drop = False):
        feat_name = self.feat_name
        if need_drop:
            feat_name = feat_name + '_drop'
        if encode_feat_query:
            if self.train_stage == 'transductive_ft':
                return self.moe_adaptor(batch["in_" + feat_name]) + self.item_embedding(batch[self.fiid])
            else:
                return self.moe_adaptor(batch["in_" + feat_name ])

        else:
            if self.train_stage == 'transductive_ft':
                return self.moe_adaptor(batch[self.feat_name]) + self.item_embedding(batch[self.fiid])
            else:
                return self.moe_adaptor(batch[self.feat_name])
            


class UniSRecQueryEncoder(SASRecQueryEncoder):
    def __init__(self, fiid, feat_name, train_stage, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder,
            bidirectional=False, training_pooling_type='last', eval_pooling_type='last') -> None:
        super().__init__(fiid, embed_dim, max_seq_len, n_head, hidden_size, dropout, activation, layer_norm_eps, n_layer, item_encoder,
            bidirectional, training_pooling_type, eval_pooling_type)
        self.feat_name = feat_name
        self.train_stage = train_stage

        
    def forward(self, batch, need_pooling=True, need_drop=False):
        user_hist = batch['in_'+self.fiid]
        positions = torch.arange(user_hist.size(1), dtype=torch.long, device=user_hist.device)
        positions = positions.unsqueeze(0).expand_as(user_hist)
        position_embs = self.position_emb(positions)

        item_emb_list = self.item_encoder(batch, encode_feat_query=True, need_drop = need_drop)

        seq_embs = item_emb_list + position_embs

        mask4padding = user_hist == 0  # BxL
        L = user_hist.size(-1)
        if not self.bidirectional:
            attention_mask = torch.triu(torch.ones((L, L), dtype=torch.bool, device=user_hist.device), 1)
        else:
            attention_mask = torch.zeros((L, L), dtype=torch.bool, device=user_hist.device)
        transformer_out = self.transformer_layer(
            src=self.dropout(seq_embs+position_embs),
            mask=attention_mask,
            src_key_padding_mask=mask4padding)  # BxLxD

        if need_pooling:
            if self.training:
                if self.training_pooling_type == 'mask':
                    return self.training_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
                else:
                    return self.training_pooling_layer(transformer_out, batch['seqlen'])
            else:
                if self.eval_pooling_type == 'mask':
                    return self.eval_pooling_layer(transformer_out, batch['seqlen'], mask_token=batch['mask_token'])
                else:
                    return self.eval_pooling_layer(transformer_out, batch['seqlen'])
        else:
            return transformer_out


    


class UniSRec(SASRec):
    def __init__(self, config):
        super().__init__(config)
        self.train_stage = config['model']['train_stage']
        self.temperature = config['model']['temperature']
        self.lam = config['model']['lamda']
        # self.item_embedding = torch.nn.Embedding(self.num_items, self.embed_dim, padding_idx=0)
        assert self.train_stage in [
            'pretrain', 'inductive_ft', 'transductive_ft'
        ], f'Unknown train stage: [{self.train_stage}]'
        if self.train_stage in ['pretrain', 'inductive_ft']:
            self.item_embedding = None
            # for `transductive_ft`, `item_embedding` is defined in SASRec base model
        if self.train_stage in ['inductive_ft', 'transductive_ft']:
            # `plm_embedding` in pre-train stage will be carried via dataloader
            # self.plm_embedding = copy.deepcopy(dataset.plm_embedding)
            pass
        self.moe_adaptor = MoEAdaptorLayer(
            config['model']['n_exps'],
            config['model']['adaptor_layers'],
            config['model']['adaptor_dropout_prob']
        )


    def _init_model(self, train_data, drop_unused_field=True):
        for feat in train_data._get_feat_list():
            for col in feat.fields:
                if train_data.field2type[col] == 'text':   
                    train_data._text_process(word_drop_ratio=0.5, col=col, feat=feat.data)
                    new_col = col + '_feat_drop'
                    feat.fields.append(new_col)
                    feat.data[new_col] = torch.stack([torch.from_numpy(d) for d in feat.data[new_col]])

        for col in train_data.item_feat.fields:
            if col.endswith('_feat'):
                self.feat_name = col
                break
        super()._init_model(train_data, drop_unused_field)
    

    def _set_data_field(self, data):
        use_field = [data.fuid, data.fiid, data.frating]
        for col in data.field:
            if data.field2type[col] == 'text_feat':
                use_field.append(col)

        data.use_field = set(use_field)

    
    def fit(self,
        train_data,
        val_data=None,
        run_mode='light',
        config = None,
          **kwargs):
        if self.train_stage == 'pretrain':
            super().fit(train_data=train_data, val_data=val_data, run_mode=run_mode, config=config, **kwargs)
            self.train_stage = 'transductive_ft'
            super().fit(train_data=train_data, val_data=val_data, run_mode=run_mode, config=config, **kwargs)
        else:
            super().fit()   

    def _get_item_encoder(self, train_data):

        return UnisRecItemEncoder(self.train_stage, train_data.num_items, self.embed_dim, self.feat_name, self.fiid, self.moe_adaptor)
    

    def _get_query_encoder(self, train_data):
        model_config = self.config['model']
        return UniSRecQueryEncoder(
            fiid=self.fiid, feat_name=self.feat_name, embed_dim=self.embed_dim, train_stage=self.train_stage,
            max_seq_len=train_data.config['max_seq_len'], n_head=model_config['head_num'],
            hidden_size=model_config['hidden_size'], dropout=model_config['dropout_rate'],
            activation=model_config['activation'], layer_norm_eps=model_config['layer_norm_eps'],
            n_layer=model_config['layer_num'],
            item_encoder=self.item_encoder
        )

    def _get_sampler(self, train_data):
        return None
    
    def training_step(self, batch):
        
        output = self.forward(batch = batch, return_query = True)
        score = output['score']
        score['label'] = batch[self.frating]
        if self.train_stage != 'pretrain':
            return self.loss_fn(**score)
        else:
            query = output['query']
            query = F.normalize(query, dim=1)
            # Remove sequences with the same next item
            pos_id = batch[self.fiid]
            same_pos_id = pos_id.unsqueeze(1) == pos_id.unsqueeze(0)
            same_pos_id = torch.logical_xor(same_pos_id, torch.eye(pos_id.shape[0], dtype=torch.bool, device=pos_id.device))
            
            loss_seq_item = self.seq_item_contrastive_task(query, same_pos_id, batch)
            loss_seq_seq = self.seq_seq_contrastive_task(query, same_pos_id, batch)

            loss = loss_seq_item + self.lam * loss_seq_seq
            # print('loss_seq_item: ', loss_seq_item.item(), 'loss_seq_seq: ', loss_seq_seq.item())
            return loss
    
    # def forward(
    #         self,
    #         batch: Dict,
    #         return_query: bool = False,
    #         encode_feat_query: bool = False
    #     ):

    #     output = {}
    #     pos_items = self._get_item_feat(batch)
    #     pos_item_vec = self.item_encoder(pos_items)

    #     query = self.query_encoder(batch, encode_feat_query=encode_feat_query)
    #     pos_score = self.score_func(query, pos_item_vec)
    #     output['score'] = {'pos_score': pos_score}
    #     output['query'] = query
    #     return output
    
    def seq_item_contrastive_task(self, seq_output, same_pos_id, batch):

        pos_items_emb = self.moe_adaptor(batch[self.feat_name])
        pos_items_emb = F.normalize(pos_items_emb, dim=1)

        pos_logits = (seq_output * pos_items_emb).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, pos_items_emb.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def seq_seq_contrastive_task(self, seq_output, same_pos_id, batch):
        # item_seq_aug = batch[self.feat_name + '_drop']
        # #may have some problem
        # item_seq_len_aug = batch['seqlen']
        # item_emb_list_aug = self.moe_adaptor(batch['in_'+self.feat_name + '_drop'])
        # seq_output_aug = self.forward(item_seq_aug, item_emb_list_aug, item_seq_len_aug)
        seq_output_aug = self.query_encoder(batch, need_pooling=True, need_drop=True)
        seq_output_aug = F.normalize(seq_output_aug, dim=1)

        pos_logits = (seq_output * seq_output_aug).sum(dim=1) / self.temperature
        pos_logits = torch.exp(pos_logits)

        neg_logits = torch.matmul(seq_output, seq_output_aug.transpose(0, 1)) / self.temperature
        neg_logits = torch.where(same_pos_id, torch.tensor([0], dtype=torch.float, device=same_pos_id.device), neg_logits)
        neg_logits = torch.exp(neg_logits).sum(dim=1)

        loss = -torch.log(pos_logits / neg_logits)
        return loss.mean()

    def _get_score_func(self):
        return scorer.CosineScorer() 