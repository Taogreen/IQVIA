import sys
from bert_serving.server.helper import get_run_args, get_args_parser
from bert_serving.server import BertServer

#model_dir = sys.argv[1]
#pooling_strategy = sys.argv[2]
#num_worker ='8'
#max_seq_len='350'
#args = get_args_parser().parse_args(['-model_dir', model_dir,
#                    '-max_seq_len',max_seq_len,
#                    '-num_worker',num_worker,
#                    '-pooling_strategy', pooling_strategy])
#server = BertServer(args)
#server.start()
#BertServer.shutdown(port=5555)
if __name__ == '__main__':
    args = get_run_args()
    print(args)
    server = BertServer(args)
    server.start()
    server.join()