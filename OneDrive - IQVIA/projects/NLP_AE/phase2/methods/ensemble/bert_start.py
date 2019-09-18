# from bert_serving.server.helper import get_args_parser
# from bert_serving.server import BertServer

# md_dir ='./chinese_L-12_H-768_A-12'
# args = get_args_parser().parse_args(['-model_dir', md_dir,
#                                      '-num_worker','3',
#                                      '-pooling_strategy','REDUCE_MEAN_MAX',
#                                      '-max_seq_len','350',
#                                      '-port', '5555',
#                                      '-port_out', '5556',
#                                     ])

# server = BertServer(args)
# server.start()

import torch
from pytorch_transformers import BertModel, BertTokenizer
import numpy as np

def bc(text, load = True):
    '''
    Using Bert pytorch transformer to encode sentences into vector
    load = True; load pretrained transformer, otherwise, load pretrained 
    model online.
    '''
    if load:
        tokenizer = BertTokenizer.from_pretrained('./BERT_pytorch')
        model = BertModel.from_pretrained('./BERT_pytorch')
    else:
        models = [BertModel, BertTokenizer, 'bert-base-chinese']
        tokenizer = models[1].from_pretrained(models[2])
        model = models[0].from_pretrained(models[2])
    
    for t in text:
        input_ids = torch.tensor([tokenizer.encode(t, add_special_tokens = True)])
        # constrain embedding size to 512
        if input_ids.shape[1]>512:
            input_ids = input_ids[:,:512]
        with torch.no_grad():
            last_hidden_states = model(input_ids)[0]

        l = last_hidden_states.shape[2]
        k_tor = np.zeros((1,l*2))
        k_tor[:,:l] = np.mean(last_hidden_states.tolist(), axis = 1)
        k_tor[:,l:] = np.max(last_hidden_states.tolist(), axis = 1)
        yield k_tor.tolist()[0]