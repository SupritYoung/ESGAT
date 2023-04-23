import torch
import torch.nn
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class RefiningStrategy(nn.Module):
    def __init__(self, hidden_dim, edge_dim, dim_e, dropout_ratio=0.5):
        super(RefiningStrategy, self).__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.dim_e = dim_e
        self.dropout = dropout_ratio
        # TODO
        self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 3, self.dim_e)
        # self.W = nn.Linear(self.hidden_dim * 2 + self.edge_dim * 1, self.dim_e)

    def forward(self, edge, node1, node2):
        batch, seq, seq, edge_dim = edge.shape
        node = torch.cat([node1, node2], dim=-1)

        edge_diag = torch.diagonal(edge, offset=0, dim1=1, dim2=2).permute(0, 2, 1).contiguous()
        edge_i = edge_diag.unsqueeze(1).expand(batch, seq, seq, edge_dim)
        edge_j = edge_i.permute(0, 2, 1, 3).contiguous()
        edge = self.W(torch.cat([edge, edge_i, edge_j, node], dim=-1))

        # edge = self.W(torch.cat([edge, node], dim=-1))

        return edge


class GraphConvLayer(nn.Module):
    """ A GCN module operated on dependency graphs. """

    def __init__(self, device, gcn_dim, edge_dim, dep_embed_dim, pooling='avg'):
        super(GraphConvLayer, self).__init__()
        self.gcn_dim = gcn_dim
        self.edge_dim = edge_dim
        self.dep_embed_dim = dep_embed_dim
        self.device = device
        self.pooling = pooling
        self.layernorm = LayerNorm(self.gcn_dim)
        self.W = nn.Linear(self.gcn_dim, self.gcn_dim)
        self.highway = RefiningStrategy(gcn_dim, self.edge_dim, self.dep_embed_dim, dropout_ratio=0.5)

    # TODO
    def forward(self, weight_prob_softmax, weight_adj, gcn_inputs, self_loop):
        '''
        weight_prob_softmax: [16, 102, 102, 60]
        '''
        batch, seq, dim = gcn_inputs.shape
        weight_prob_softmax = weight_prob_softmax.permute(0, 3, 1, 2)

        gcn_inputs = gcn_inputs.unsqueeze(1).expand(batch, self.edge_dim, seq, dim)

        weight_prob_softmax = self_loop + weight_prob_softmax
        Ax = torch.matmul(weight_prob_softmax, gcn_inputs)
        if self.pooling == 'avg':
            Ax = Ax.mean(dim=1)
        elif self.pooling == 'max':
            Ax, _ = Ax.max(dim=1)
        elif self.pooling == 'sum':
            Ax = Ax.sum(dim=1)
        # Ax: [batch, seq, dim]
        gcn_outputs = self.W(Ax)
        gcn_outputs = self.layernorm(gcn_outputs)
        weights_gcn_outputs = F.relu(gcn_outputs)

        node_outputs = weights_gcn_outputs
        weight_prob_softmax = weight_prob_softmax.permute(0, 2, 3, 1).contiguous()
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)

        # node_outputs: [16, 102, 300]

        return node_outputs, edge_outputs



class Biaffine(nn.Module):
    def __init__(self, args, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.args = args
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = torch.nn.Linear(in_features=self.linear_input_size,
                                    out_features=self.linear_output_size,
                                    bias=False)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(self.args.device)
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(self.args.device)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.linear(input1)
        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)
        return biaffine


class GATLayer(nn.Module):
    """ A GAT module operated on dependency graphs. """

    def __init__(self, device, input_dim, edge_dim, dep_embed_dim, num_heads):
        super(GATLayer, self).__init__()
        self.hidden_dim = input_dim // num_heads
        self.edge_dim = edge_dim
        self.input_dim = input_dim
        self.dep_embed_dim = dep_embed_dim
        self.device = device
        self.layernorm = LayerNorm(self.hidden_dim * num_heads)
        self.W = nn.Linear(self.hidden_dim * num_heads, self.hidden_dim * num_heads)
        self.highway = RefiningStrategy(self.hidden_dim * num_heads, self.edge_dim, self.dep_embed_dim,
                                        dropout_ratio=0.5)
        self.linear = nn.Linear(input_dim, self.hidden_dim * num_heads)
        self.fc_w1 = nn.Parameter(torch.empty(size=(1, 1, num_heads, self.hidden_dim)))
        nn.init.xavier_uniform_(self.fc_w1.data, gain=1.414)
        self.fc_w2 = nn.Parameter(torch.empty(size=(1, 1, num_heads, self.hidden_dim)))
        nn.init.xavier_uniform_(self.fc_w2.data, gain=1.414)
        self.num_heads = num_heads
        self.edge_mlp = nn.Sequential(nn.Linear(edge_dim, 128),
                                      nn.ReLU(),
                                      nn.Dropout(p=0.5),
                                      nn.Linear(128, num_heads),
                                      )

    def forward(self, weight_prob_softmax, weight_adj, gcn_inputs, self_loop):
        batch, seq, dim = gcn_inputs.shape
        weight_prob_softmax = self_loop + weight_prob_softmax
        # 哪个地方有边
        mask = weight_prob_softmax.sum(-1) == 0
        # 维度对齐
        feature = self.linear(gcn_inputs).reshape(batch, seq, self.num_heads, self.hidden_dim)
        #
        attn_src = torch.sum(feature * self.fc_w1, dim=-1).permute(0, 2, 1).unsqueeze(-1)
        attn_dst = torch.sum(feature * self.fc_w2, dim=-1).permute(0, 2, 1).unsqueeze(-2)
        A = self.edge_mlp(weight_prob_softmax).permute(0, 3, 1, 2)
        # print(attn_src)
        # print(attn_dst)
        # print(A)
        attn = F.leaky_relu(attn_src + attn_dst + A)
        # 把没有边的地方设置为 负无穷
        attn = torch.masked_fill(attn, mask.unsqueeze(-3), float("-inf"))
        attn = torch.softmax(attn, axis=-1)
        # 可以加 pooling
        gcn_outputs = torch.matmul(attn, feature.permute(0, 2, 1, 3))
        gcn_outputs = gcn_outputs.permute(0, 2, 1, 3).reshape(batch, seq, -1)
        gcn_outputs = self.W(gcn_outputs)
        gcn_outputs = self.layernorm(gcn_outputs)

        weights_gcn_outputs = F.relu(gcn_outputs)
        # weights_gcn_outputs = gcn_outputs

        node_outputs = weights_gcn_outputs
        weight_prob_softmax = weight_prob_softmax.permute(0, 2, 3, 1).contiguous()
        node_outputs1 = node_outputs.unsqueeze(1).expand(batch, seq, seq, dim)
        node_outputs2 = node_outputs1.permute(0, 2, 1, 3).contiguous()
        edge_outputs = self.highway(weight_adj, node_outputs1, node_outputs2)

        return node_outputs, edge_outputs


class Attention_fusion(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention_fusion, self).__init__()
        self.query_proj = nn.Linear(1, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query, key, value):
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        result = torch.bmm(attn.unsqueeze(1), value)
        return result, attn


class EMCGCN(nn.Module):
    def __init__(self, args):
        super(EMCGCN, self).__init__()
        self.args = args
        self.bert = BertModel.from_pretrained(args.bert_model_path, return_dict=False)
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_model_path)
        self.dropout_output = torch.nn.Dropout(args.emb_dropout)

        self.post_emb = torch.nn.Embedding(args.post_size, args.class_num, padding_idx=0)  # 相对位置距离
        self.deprel_emb = torch.nn.Embedding(args.deprel_size, args.class_num, padding_idx=0)  # 语法解析类型
        self.postag_emb = torch.nn.Embedding(args.postag_size, args.class_num, padding_idx=0)  # 语法标注类型
        self.synpost_emb = torch.nn.Embedding(args.max_sequence_len, args.class_num, padding_idx=0)  # 语法距离

        self.lstm = torch.nn.LSTM(input_size=args.bert_feature_dim, hidden_size=args.bert_feature_dim,
                                  num_layers=1, batch_first=True, bidirectional=True)

        if args.fusion == 'attetion':
            self.attn_fusion = Attention_fusion(args.class_num)
        elif args.fusion == 'add':
            self.pm_dense = nn.Linear(1, 10)
            # 初试权重为 1，等同于复制
            self.pm_dense.weight = nn.Parameter(torch.ones(10).reshape(10, 1))

        self.triplet_biaffine = Biaffine(args, args.gcn_dim, args.gcn_dim, args.class_num, bias=(True, True))
        self.ap_fc = nn.Linear(2 * args.bert_feature_dim, args.gcn_dim)
        self.op_fc = nn.Linear(2 * args.bert_feature_dim, args.gcn_dim)

        self.dense = nn.Linear(2 * args.bert_feature_dim, args.gcn_dim)
        self.num_layers = args.num_layers
        self.gcn_layers = nn.ModuleList()

        self.layernorm = LayerNorm(args.bert_feature_dim)

        for i in range(self.num_layers):
            self.gcn_layers.append(
                # GraphConvLayer(args.device, args.gcn_dim, 6 * args.class_num, args.class_num, args.pooling))
                GATLayer(args.device, args.gcn_dim, 2 * args.class_num, args.class_num, num_heads=2))

    def forward(self, tokens, masks, word_pair_position, word_pair_deprel, word_pair_pos, word_pair_synpost,
                perturbed_matrix=None):
        # perturbed_matrix: [batch_size, max_length, max_length]
        bert_feature, _ = self.bert(tokens, masks, return_dict=False)
        bert_feature = self.dropout_output(bert_feature)
        bert_feature, _ = self.lstm(bert_feature)

        batch, seq = masks.shape
        tensor_masks = masks.unsqueeze(1).expand(batch, seq, seq).unsqueeze(-1)

        # BiAffine
        ap_node = F.relu(self.ap_fc(bert_feature))
        op_node = F.relu(self.op_fc(bert_feature))
        biaffine_edge = self.triplet_biaffine(ap_node, op_node)  # [16, 102, 102, 10]
        gcn_input = F.relu(self.dense(bert_feature))
        gcn_outputs = gcn_input

        # **1** multi-feature cat 方案
        # word_pair_post_emb = self.post_emb(word_pair_position)  # [batch_size, max_length, max_length, k=10]
        # word_pair_deprel_emb = self.deprel_emb(word_pair_deprel)
        # word_pair_postag_emb = self.postag_emb(word_pair_pos)
        # word_pair_synpost_emb = self.synpost_emb(word_pair_synpost)
        #
        # perturbed_matrix = perturbed_matrix.unsqueeze(len(perturbed_matrix.shape))
        # perturbed_matrix_emb = self.pm_dense(perturbed_matrix)
        #
        # # 将 PM 与句法和语法嵌入进行特征融合
        #
        # weight_prob_list = [perturbed_matrix_emb, biaffine_edge, word_pair_post_emb, word_pair_deprel_emb,
        #                     word_pair_postag_emb,
        #                     word_pair_synpost_emb]
        #
        # perturbed_matrix_emb_softmax = F.softmax(perturbed_matrix_emb, dim=-1) * tensor_masks
        # biaffine_edge_softmax = F.softmax(biaffine_edge, dim=-1) * tensor_masks  # 乘以 tensor_masks，只保留有 token 地方的概率
        # word_pair_post_emb_softmax = F.softmax(word_pair_post_emb, dim=-1) * tensor_masks
        # word_pair_deprel_emb_softmax = F.softmax(word_pair_deprel_emb, dim=-1) * tensor_masks
        # word_pair_postag_emb_softmax = F.softmax(word_pair_postag_emb, dim=-1) * tensor_masks
        # word_pair_synpost_emb_softmax = F.softmax(word_pair_synpost_emb, dim=-1) * tensor_masks
        #
        # self_loop = []
        # for _ in range(batch):
        #     self_loop.append(torch.eye(seq))
        # self_loop = torch.stack(self_loop).to(self.args.device).unsqueeze(3) \
        #                 .expand(batch, seq, seq, 6 * self.args.class_num) * tensor_masks.permute(0, 1, 2,
        #                                                                                          3).contiguous()
        #
        # weight_prob = torch.cat([perturbed_matrix_emb, biaffine_edge, word_pair_post_emb, word_pair_deprel_emb, \
        #                          word_pair_postag_emb, word_pair_synpost_emb], dim=-1)
        # weight_prob_softmax = torch.cat(
        #     [perturbed_matrix_emb_softmax, biaffine_edge_softmax, word_pair_post_emb_softmax, \
        #      word_pair_deprel_emb_softmax, word_pair_postag_emb_softmax,
        #      word_pair_synpost_emb_softmax], dim=-1)
        #
        # for _layer in range(self.num_layers):
        #     # 循环之后的 weight_prob 最后一维为 10，因此会报错，所以说要对 PM 用注意力机制与其他 embedding 进行融合
        #     gcn_outputs, weight_prob = self.gcn_layers[_layer](weight_prob_softmax, weight_prob, gcn_outputs,
        #                                                        self_loop)  # [batch, seq, dim]
        #     if _layer == 0:
        #         weight_prob_list.append(weight_prob)
        #     else:
        #         weight_prob_list[-1] = weight_prob
        #     # print(torch.argmax(F.softmax(weight_prob, dim=-1), dim=3))
        #
        # return weight_prob_list

        # **2** TODO 特征融合方案
        word_pair_post_emb = self.post_emb(word_pair_position)  # [batch_size, max_length, max_length, k=10]
        word_pair_deprel_emb = self.deprel_emb(word_pair_deprel)
        word_pair_postag_emb = self.postag_emb(word_pair_pos)
        word_pair_synpost_emb = self.synpost_emb(word_pair_synpost)
        perturbed_matrix = perturbed_matrix.unsqueeze(len(perturbed_matrix.shape))

        # 特征融合（60 -> 10）
        # 方法 1 直接相加
        if self.args.fusion == 'add':
            perturbed_matrix_emb = self.pm_dense(perturbed_matrix)
            edge_emb = word_pair_post_emb + word_pair_deprel_emb + word_pair_postag_emb + \
                       word_pair_synpost_emb + perturbed_matrix_emb
        # 方法 2 特征相乘
        elif self.args.fusion == 'multiply':
            word_pair_post_emb = torch.mul(word_pair_post_emb, perturbed_matrix)
            word_pair_deprel_emb = torch.mul(word_pair_deprel_emb, perturbed_matrix)
            word_pair_postag_emb = torch.mul(word_pair_postag_emb, perturbed_matrix)
            word_pair_synpost_emb = torch.mul(word_pair_synpost_emb, perturbed_matrix)
            # biaffine_edge = torch.mul(biaffine_edge, perturbed_matrix)
            edge_emb = word_pair_post_emb + word_pair_deprel_emb + word_pair_postag_emb + \
                       word_pair_synpost_emb + biaffine_edge
        # elif self.args.fusion == 'attention':
        # 方法 3 注意力机制
        # word_pair_deprel_emb, _ = self.attn_fusion(perturbed_matrix, word_pair_deprel_emb, word_pair_deprel_emb)
        # word_pair_postag_emb, _ = self.attn_fusion(perturbed_matrix, word_pair_postag_emb, word_pair_postag_emb)

        # 自环
        self_loop = []
        for _ in range(batch):
            self_loop.append(torch.eye(seq))
        try:
            self_loop = torch.stack(self_loop).to(self.args.device).unsqueeze(3) \
                            .expand(batch, seq, seq, 2 * self.args.class_num) * tensor_masks.permute(0, 1, 2,
                                                                                                     3).contiguous()
        except Exception as ex:
            print(ex)

        weight_prob_list = [edge_emb, biaffine_edge]
        weight_prob = torch.cat([edge_emb, biaffine_edge], dim=-1)
        edge_emb_softmax = F.softmax(edge_emb, dim=-1) * tensor_masks
        biaffine_edge_softmax = F.softmax(biaffine_edge, dim=-1) * tensor_masks
        weight_prob_softmax = torch.cat([edge_emb_softmax, biaffine_edge_softmax], dim=-1)

        # weight_prob = edge_emb_softmax
        # weight_prob_softmax = edge_emb_softmax
        # weight_prob_list = [edge_emb_softmax]

        for _layer in range(self.num_layers):
            # 循环之后的 weight_prob 最后一维为 10，因此会报错，所以说要对 PM 用注意力机制与其他 embedding 进行融合
            gcn_outputs, weight_prob = self.gcn_layers[_layer](weight_prob_softmax, weight_prob, gcn_outputs,
                                                               self_loop)  # [batch, seq, dim]
            if _layer == 0:
                weight_prob_list.append(weight_prob)
            else:
                weight_prob_list[-1] = weight_prob
            # print(torch.argmax(F.softmax(weight_prob, dim=-1), dim=3))

        return weight_prob_list
