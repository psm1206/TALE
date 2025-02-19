import torch
import numpy as np
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender, SequentialRecommender
from scipy.sparse import coo_matrix


class TALE(SequentialRecommender):

    input_type = InputType.POINTWISE
    
    def __init__(self, config, dataset):
        super().__init__(config, dataset)
        self.dataset = dataset
        self.reg_weight = config["reg_weight"]
        self.tau_train = config['tau_train']
        self.tau_inf = config['tau_inf']

        self.c = config['c']
        self.beta = config['beta']
        self.gamma = config['gamma']

        self.dummy_param = torch.nn.Parameter(torch.zeros(1))
        
        inter_feat = dataset.inter_feat
        
        self.make_slit_preprocess(inter_feat)
        self.make_slit_B()
        
        self.other_parameter_name = ["interaction_matrix", "item_similarity"]
        self.device = config.device
     
    def return_seq(self, seq_info):
        target_item = seq_info.item_id
        target_time = seq_info.timestamp
        target_pop = seq_info.count
        seq_len = seq_info.item_length
        
        item_seq = seq_info.item_id_list[:seq_len].tolist() + [target_item]
        
        time_seq = np.append(seq_info.timestamp_list[:seq_len].numpy(), target_time)
        item_pop_seq = np.append(seq_info.count_list[:seq_len].numpy(), target_pop)
        
        return item_seq, time_seq, item_pop_seq, seq_len
    
    def make_slit_preprocess(self, inter_feat):
        seq_num = len(inter_feat)
        self.idx_count = 0
        
        self.total_source_items_pop = []
        self.total_source_items = []
        self.total_target_items = []
        self.total_source_weights = []
        
        self.source_seq_idx = []
        self.target_idx = []
        
        for i in range(seq_num):
            item_seq, time_seq, item_pop_seq, seq_len = self.return_seq(inter_feat[i])
            
            for j in range(0, seq_len):
                source_item_seq = item_seq[:j+1]
                source_item_pop_seq = item_pop_seq[:j+1]
                target_item = item_seq[j+1]
                source_time_seq = time_seq[:j+1]
                target_time = time_seq[j+1]
                
                time_diff = np.abs(source_time_seq - target_time)
                
                self.total_source_weights.extend(time_diff)
                self.total_source_items_pop.extend(source_item_pop_seq)
                
                self.total_source_items.extend(source_item_seq)
                self.total_target_items.append(target_item)
                
                self.source_seq_idx.extend([self.idx_count]*(j+1))
                self.target_idx.extend([self.idx_count])
                
                self.idx_count += 1
        
    def make_slit_B(self):
        
        total_source_weights = -np.array(self.total_source_weights)
        # from IPython import embed; embed()
        weights = np.exp(total_source_weights/self.tau_train)
        weights += (weights < self.c) * self.c + 1
        
        S = coo_matrix((weights, (self.source_seq_idx, self.total_source_items)), shape=(self.idx_count, self.n_items))
        T = coo_matrix((np.ones([self.idx_count]), (self.target_idx, self.total_target_items)), shape=(self.idx_count, self.n_items))

        G = S.T @ S
        K = S.T @ T

        G = torch.Tensor(G.todense())
        K = torch.Tensor(K.todense())
        
        A = G + self.reg_weight * torch.eye(G.shape[0])
        B = K
        B = torch.linalg.solve(A,B)
        
        self.item_similarity = B
        
    def forward(self):
        pass

    def calculate_loss(self, interaction):
        return torch.nn.Parameter(torch.zeros(1))

    def predict(self, interaction):
        X = self.recon_eval(interaction) 
        return X @ self.item_similarity
    
    def full_sort_predict(self, interaction):
        X = self.recon_eval(interaction) 
        return X @ self.item_similarity
    
    
    def recon_eval(self, interaction):
        item_seq_len = interaction['item_length']
        item_id_list = interaction['item_id_list']
        
        users = []
        items = []
        weights = []
        
        for i in range(len(item_seq_len)):
            seq_len = item_seq_len[i].item()
            item_seq = item_id_list[i][:seq_len].tolist()
            weight = np.arange(seq_len)
            user = [i] * seq_len

            users.extend(user)
            items.extend(item_seq)
            weights.extend(-weight[::-1])
        
        users = torch.LongTensor(users)
        items = torch.LongTensor(items)
        weights = torch.FloatTensor(weights)
        weights = torch.exp(weights/self.tau_inf)
        
        X = torch.zeros([len(interaction), self.n_items])
        X[users, items] = weights

        return X
    