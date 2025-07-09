import torch
import torch.nn.functional as F

def agg_encode(prefix_act, prefix_time, num_act):
    # prefix_act shape: (num_samples, prefix_len)
    # prefix_time shape: (num_samples, prefix_len, 2)

    prefix_act_one_hot = F.one_hot(prefix_act, num_classes=num_act) # shape: (num_samples, prefix_len, num_act)
    prefix_act_agg = prefix_act_one_hot.sum(dim=1) # shape: (num_samples, num_act)
    # remove frequency of padded 0s
    prefix_act_agg = prefix_act_agg[:, 1:]  # shape: (num_samples, num_act-1)
    
    # replace -10000 with 0
    prefix_time[prefix_time == -10000.00] = 0.0
    prefix_time_sum = prefix_time.sum(dim=1)  # shape: (num_samples, 2)
    prefix_lens = (prefix_act != 0).sum(dim=1).unsqueeze(1) # shape: (num_samples, 1)
    assert (prefix_lens > 0).all(), "Found at least one row with all zeros."
    prefix_time_agg = prefix_time_sum / prefix_lens # shape: (num_samples, 2)
    
    prefix_agg = torch.cat((prefix_act_agg, prefix_time_agg), dim=1) # shape: (num_samples, num_act+1)
    prefix_agg = prefix_agg.numpy()

    return prefix_agg

def index_encode(prefix_act, prefix_time, num_act):
    # prefix_act shape: (num_samples, prefix_len)
    # prefix_time shape: (num_samples, prefix_len, 2)

    prefix_act_one_hot = F.one_hot(prefix_act, num_classes=num_act) # shape: (num_samples, prefix_len, num_act)

    prefix = torch.cat((prefix_act_one_hot, prefix_time), dim=-1) # shape: (num_samples, prefix_len, num_act+2)

    prefix_index = prefix.reshape(prefix.shape[0], -1) # shape: (num_samples, prefix_len * (num_act+2) )
    prefix_index = prefix_index.numpy()

    return prefix_index


    

