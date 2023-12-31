# -- coding: utf-8 --**
# @author zjp
# @time 22-04-18_10.44.25

import os
import sys
import time
import torch
import random
import logging
import argparse
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import timedelta
from importlib import import_module

from transformers import BertTokenizer, BertConfig
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

from utils.process_dataset import process_dataset
from utils.build_dataset import build_dataset
from utils.prepare_vec import prepare_vector
from utils.REDataset import REDataset

from sklearn.metrics import accuracy_score, precision_score, recall_score,\
    f1_score, classification_report, confusion_matrix, precision_recall_fscore_support

warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
n_gpu = torch.cuda.device_count()  ## 如果有多块GPU


def get_args():
    '''设置需要的参数'''
    parser = argparse.ArgumentParser(description="Chinese medicine Relation Extraction")
    parser.add_argument('--datadir', type=str, default='./data', help='location of the data corpus')
    parser.add_argument('--dataset', type=str, default='TCM_ET',
                        help='the data corpus(CMeIE,CMeIE_ET, TCM_ET, sample )')  ########
    parser.add_argument('--bert_path', type=str, default='./pretrained_model', help='BERT model path')
    parser.add_argument('--bert_name', type=str, default='bert-base-chinese',  ########
                        help='type of bert ( bert-base-chinese, ) ')
    parser.add_argument('--use_SPECIAL_TOKENS', default=True, help='添加特殊标记')
    parser.add_argument('--model_name', type=str, default='bert_bilstm_gcn',  #######################
                        help='name of network (bert, bilstm,bilstm_ygcn, bert_bilstm_gcn,\
                         bert_bilstm_yygcn, bert_bilstm_ygcn_att_type, bert_bilstm_gcn_att ,\
                         bilstm_gcn_dis, bert_bilstm_gcn_dis, bert_bilstm_gcn_dis_local,bert_bilstm_gcn_dis_local_global)')
    parser.add_argument('--dep_type', type=list, default=["full_graph"],
                        choices=["full_graph", "local_graph", "global_graph", "local_global_graph", "distance_graph",
                                 ",选择后多个时保持距离矩阵在前"])  ######
    parser.add_argument('--num_gcn_layers', type=int, default=2)  ####
    parser.add_argument('--renorm', type=int, default=0, help="重整化参数 =0 为不使用")  ####
    parser.add_argument('--use_vector', default=True, help='Whether to use pretrained word vector')
    parser.add_argument('--vector_file', type=str, default='./vector/cc.zh.300.bin', help='词向量文件的路径')
    parser.add_argument('--vector_dim', type=int, default=300, help='词嵌入的维度')
    parser.add_argument('--grnn_dim', type=int, default=128, help='衔接GCN的BiLSTM层的隐藏维度')
    parser.add_argument('--brnn_dim', type=int, default=128, help='衔接BERT的BiLSTM层的隐藏维度')
    parser.add_argument('--cnn_size', type=int, default=32, help='cnn层的隐藏维度')

    parser.add_argument('--output_dir', type=str, default='../../autodl-tmp/RE_CM_result',
                        help='location of the data corpus')
    parser.add_argument('--log_dir', type=str, default='./log', help='location of the log')

    parser.add_argument('--maxlen', type=float, default=300,  ################
                        help='最大句子长度(CMeIE 170 TCM_ET 300)')
    parser.add_argument('--max_word_len', type=float, default=160,  ################
                        help='最大分词后句子的长度(CMeIE 70 TCM 160)')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='initial learning rate()')
    parser.add_argument('--learning_rate_gcn', type=float, default=5e-3, help='initial learning rate()')
    parser.add_argument('--dropout_prob', type=float, default=0.2, help='dropout applied to layers (0 = no dropout) ')
    parser.add_argument('--epochs', type=int, default=10, help='upper epoch limit')
    parser.add_argument('--train_batch_size', type=int, default=12, help='batch size of train CMeIE 64 TCM 32')  ##
    parser.add_argument('--eval_batch_size', type=int, default=12, help='batch size of evaluation')
    parser.add_argument('--test_batch_size', type=int, default=12, help='batch size of test')
    parser.add_argument('--require_improvement', type=int, default=2000, help='早停')
    parser.add_argument('--warmup_proportion', type=float, default=0.2,
                        help='Proportion of training to perform linear learning rate warmup for. ')
    parser.add_argument('--eval_steps', type=int, default=100, help='number of steps for evaluation')

    parser.add_argument('--do_train', default=True, help='Whether to run training.')
    # parser.add_argument('--do_eval', default=True, help='Whether to run eval on the dev set.')
    parser.add_argument('--do_test', default=False, help='Whether to run eval on the test set.')
    # parser.add_argument('--do_predict', default=False, help='Whether to predict.')
    parser.add_argument('--test_model_idx', type=int, default=10000, help='Whether to predict.')

    parser.add_argument('--seed', type=int, default=2023, help='random seed')
    parser.add_argument('--cuda', default=True, help='use CUDA')

    args = parser.parse_args()
    return args


def set_params(args):
    '''设置一些重要参数
    输出文件夹， 日志文件夹， 日志格式， 随机种子
    '''
    args.output_dir = os.path.join(args.output_dir, args.dataset, args.bert_name, "_".join(args.dep_type))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    args.log_dir = os.path.join(args.log_dir, args.dataset, args.bert_name, "_".join(args.dep_type))
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    logger = logging.getLogger('root')
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%Y/%m/%d %H:%M:%S',
                        level=logging.INFO)
    ## 写入到文件
    formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S')
    fhlr = logging.FileHandler(os.path.join(args.log_dir, args.model_name + "_train.log"), encoding="utf-8")
    fhlr.setFormatter(formatter)
    logger.addHandler(fhlr)
    logger.info("=================begin=================")
    logger.info(sys.argv)
    for i in vars(args).items():
        logger.info(f"{i[0]}: {i[1]}")

    ## 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    return logger


def model_params(prompt_model, logger):
    '''Find total parameters and trainable parameters'''
    total_params = sum(p.numel() for p in prompt_model.parameters())
    logger.info(f'{total_params:,} total parameters.')
    total_trainable_params = sum(
        p.numel() for p in prompt_model.parameters() if p.requires_grad)
    logger.info(f'{total_trainable_params:,} training parameters.')


def train_function(args, logger, model, tokenizer, dataset, results={}):
    '''定义训练模型所需要的数据以及模型、一些重要参数等  训练过程'''
    train_examples = dataset.get_train_examples()  ## 获取训练数据
    train_dataset = build_dataset(train_examples, tokenizer, args.word2index, args)
    train_data = REDataset(train_dataset, args)
    train_sampler = RandomSampler(train_data)
    trian_dataloader = DataLoader(train_data,
                                  sampler=train_sampler,
                                  batch_size=args.train_batch_size,
                                  num_workers=3)

    eval_examples = dataset.get_dev_examples()
    eval_dataset = build_dataset(eval_examples, tokenizer, args.word2index, args)
    eval_data = REDataset(eval_dataset, args)
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data,
                                 sampler=eval_sampler,
                                 batch_size=args.eval_batch_size,
                                 num_workers=3)

    num_train_optimization_steps = int(
        len(train_examples) / args.train_batch_size) * args.epochs
    num_warmup_steps = int(args.warmup_proportion * num_train_optimization_steps)
    max_grad_norm = 1.0

    model.to(device)
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    other = ['lstm_word', 'gcn_layer', 'words_embedding', 'dep_type_embedding']  ##gcn相关的设置较高的学习率 weight_deacy 0.01 lrg
    ## weight_deacy 0.01 lr         not no_decay, not other
    ## weight_deacy 0.0 lr          no_decay, not other
    ## weight_deacy 0.01 lrg        not no_decay, other
    ## weight_deacy 0.01 lrg        other, no_decay
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in other)], 'weight_decay': 0.01,
         'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in other)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in other)], 'weight_decay': 0.01,
         'lr': args.learning_rate_gcn},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': args.learning_rate_gcn}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_train_optimization_steps)
    logger.info("lr: {} warm_step: {} total_step: {}".format(args.learning_rate, num_warmup_steps,
                                                             num_train_optimization_steps))
    model_params(model, logger)
    torch.cuda.empty_cache()
    model.train()
    global_step = 0
    best_macro_f1 = 0
    last_improve = 0  # 记录上次验证集loss下降的batch数
    flag = False  # 记录是否很久没有效果提升
    start_time = time.time()
    for epoch_num in range(args.epochs):
        logger.info('Epoch [{}/{}]'.format(epoch_num + 1, args.epochs))
        tr_loss = 0
        nb_tr_steps = 0
        for batch in trian_dataloader:
            batch = tuple(t.to(device) for t in batch)
            input_ids, attention_masks, segment_ids, entity_token_ids, label_id, word_feature, e1_mask, e2_mask, dep_type_matrix, dep_distence_matrix = batch

            loss = model(input_ids=input_ids,
                         attention_masks=attention_masks,
                         segment_ids=segment_ids,
                         entity_token_ids=entity_token_ids,
                         label_id=label_id,
                         word_feature=word_feature,
                         e1_mask=e1_mask,
                         e2_mask=e2_mask,
                         dep_type_matrix=dep_type_matrix,
                         dep_distence_matrix=dep_distence_matrix)
            loss.backward()
            perplexity = torch.exp(torch.tensor(loss))
            tr_loss += loss.item()
            nb_tr_steps += 1
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)  # 梯度裁剪不再在AdamW中了(因此你可以毫无问题地使用放大器)
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

            if global_step % args.eval_steps == 0:
                trues, preds = evaluate(args, model, eval_dataloader)
                macro_f1 = f1_score(y_true=trues, y_pred=preds, average='macro')
                if macro_f1 > best_macro_f1:
                    checkpoint = {'epoch': epoch_num, 'loss': tr_loss / nb_tr_steps, 'state_dict': model.state_dict(),
                                  'optimizer': optimizer.state_dict(), }
                    # save_path_name = os.path.join(args.output_dir,args.model_name+"_best.pt")
                    save_path_name = os.path.join(args.output_dir, args.model_name + "_{}_best.pt".format(global_step))
                    logger.info("model save path: " + str(save_path_name))
                    torch.save(checkpoint, save_path_name)
                    best_macro_f1 = macro_f1
                    improve = '*'
                    last_improve = global_step
                else:
                    improve = ''
                time_dif = timedelta(seconds=int(round(time.time() - start_time)))
                msg = 'Iter: {0:},T Loss: {1:.5},T perplexity: {2:.5}, V F1: {3:.5},Time: {4} {5}'
                logger.info(msg.format(global_step, tr_loss / nb_tr_steps, perplexity, macro_f1, time_dif, improve))
                model.train()
            if global_step - last_improve > args.require_improvement:
                # 验证集loss超过指定batch没下降，结束训练
                logger.info("No optimization for a long time, auto-stopping...")
                flag = True
                break
        if flag:
            break


def evaluate(args, model, dataloader):
    # examples = dataset.get_dev_examples()
    # eval_dataset = build_dataset(examples, tokenizer, word2index, args)
    # eval_data = REDataset(eval_dataset, args)
    # eval_sampler = SequentialSampler(eval_data)
    # eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=6)

    model.eval()
    pred_scores = None
    out_label_ids = None
    if args.do_test:
        dataloader = tqdm(dataloader, desc="test model")
    for batch in dataloader:
        batch = tuple(t.to(device) for t in batch)
        input_ids, attention_masks, segment_ids, entity_token_ids, label_id, word_feature, e1_mask, e2_mask, dep_type_matrix, dep_distence_matrix = batch
        with torch.no_grad():
            logits = model(input_ids=input_ids,
                           attention_masks=attention_masks,
                           segment_ids=segment_ids,
                           entity_token_ids=entity_token_ids,
                           label_id=None,
                           word_feature=word_feature,
                           e1_mask=e1_mask,
                           e2_mask=e2_mask,
                           dep_type_matrix=dep_type_matrix,
                           dep_distence_matrix=dep_distence_matrix)

        if pred_scores is None:
            pred_scores = logits.detach().cpu().numpy()
            out_label_ids = label_id.detach().cpu().numpy()
        else:
            pred_scores = np.append(pred_scores, logits.detach().cpu().numpy(), axis=0)
            out_label_ids = np.append(out_label_ids, label_id.detach().cpu().numpy(), axis=0)
    preds = np.argmax(pred_scores, axis=1)
    return out_label_ids, preds


def base_function(args, logger):
    '''训练模型, 测试模型共同需要的一些功能'''
    ## 处理数据
    dataset = process_dataset(args=args)
    # dep_data = dataset.get_train_examples()
    # dep_data = dataset.get_test_examples()
    # dep_data = dataset.get_dev_examples()
    args.types_dict = dataset.prepare_type_dict()  ## 获取依存类型字典
    args.labels_dict = dataset.prepare_labels_dict()  ## 获取标签字典
    dep_type_list = args.types_dict.keys()  ## 获取依存类型列表
    label_list = args.labels_dict.keys()  ## 获取标签列表
    type_num = len(dep_type_list)
    num_labels = len(label_list)

    ## 准备词向量嵌入数据
    vector = prepare_vector(args=args)
    args.word2index = vector.word_index()
    if args.use_vector:
        args.embedding_matrix = vector.get_embedding_matrix()
    else:
        logger.info("不使用词向量文件")
        args.embedding_matrix = []

    bert_vocab_file = os.path.join(args.bert_path, args.bert_name, 'vocab.txt')
    logger.info("加载 bert vocab.txt, 文件路径: {}".format(bert_vocab_file))

    if args.use_SPECIAL_TOKENS:
        args.ADDITIONAL_SPECIAL_TOKENS = []
        entity_type_file = os.path.join(args.datadir, args.dataset, 'entity_type.txt')
        with open(entity_type_file, 'r', encoding="utf-8") as f:
            for each in f:
                args.ADDITIONAL_SPECIAL_TOKENS.append(each.strip())
        tokenizer = BertTokenizer.from_pretrained(bert_vocab_file, do_lower_case=True)
        tokenizer.add_special_tokens({"additional_special_tokens": args.ADDITIONAL_SPECIAL_TOKENS})
    else:
        args.ADDITIONAL_SPECIAL_TOKENS = []
        tokenizer = BertTokenizer.from_pretrained(bert_vocab_file, do_lower_case=True)

    config_file = os.path.join(args.bert_path, args.bert_name, 'config.json')
    config = BertConfig.from_json_file(config_file)
    config.__dict__['num_gcn_layers'] = args.num_gcn_layers
    config.__dict__['type_num'] = type_num
    config.__dict__['num_relation_labels'] = num_labels
    config.__dict__['dep_type'] = args.dep_type
    args.tokenizer_len = len(tokenizer)

    return config, tokenizer, dataset


def load_model_from_file(model, optimizer, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    return model, optimizer, epoch, loss


# model_params(prompt_model, logger):

def test_func(args, logger, model, tokenizer, dataset, model_path):
    '''测试模型'''
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    other = ['lstm_word', 'gcn_layer', 'words_embedding', 'dep_type_embedding']  ##gcn相关的设置较高的学习率 weight_deacy 0.01 lrg
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay) and not any(nd in n for nd in other)], 'weight_decay': 0.01,
         'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if
                    any(nd in n for nd in no_decay) and not any(nd in n for nd in other)], 'weight_decay': 0.0,
         'lr': args.learning_rate},
        {'params': [p for n, p in param_optimizer if
                    not any(nd in n for nd in no_decay) and any(nd in n for nd in other)], 'weight_decay': 0.01,
         'lr': args.learning_rate_gcn},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in other) and any(nd in n for nd in no_decay)],
         'weight_decay': 0.01, 'lr': args.learning_rate_gcn}
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=1e-8)

    model, optimizer, epoch, loss = load_model_from_file(model, optimizer, model_path)

    model_params(model, logger)
    test_examples = dataset.get_test_examples()
    test_dataset = build_dataset(test_examples, tokenizer, args.word2index, args)
    test_data = REDataset(test_dataset, args)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data,
                                 sampler=test_sampler,
                                 batch_size=args.test_batch_size)

    model.to(device)
    # result = evaluate(args, model, tokenizer, processor, device, mode="test")
    trues, preds = evaluate(args, model, test_dataloader)

    accuracy = accuracy_score(trues, preds)
    micro_p = precision_score(trues, preds, average='micro')
    micro_r = recall_score(trues, preds, average='micro')
    micro_f = f1_score(trues, preds, average='micro')

    macro_p = precision_score(trues, preds, average='macro')
    macro_r = recall_score(trues, preds, average='macro')
    macro_f = f1_score(trues, preds, average='macro')

    weighted_p = precision_score(trues, preds, average='weighted')
    weighted_r = recall_score(trues, preds, average='weighted')
    weighted_f = f1_score(trues, preds, average='weighted')

    logger.info("accuracy: {:.6f} ".format(accuracy))
    logger.info("micro_precision: {:.6f}, macro_precision: {:.6f}, weighted_precision: {:.6f}".format(micro_p, macro_p,
                                                                                                      weighted_p))
    logger.info(
        "micro_recall: {:.6f}, macro_recall: {:.6f}, weighted_recall: {:.6f}".format(micro_r, macro_r, weighted_r))
    logger.info("micro_f1: {:.6f}, macro_f1: {:.6f}, weighted_f1: {:.6f}".format(micro_f, macro_f, weighted_f))
    label_list = args.labels_dict.keys()  ## 获取标签列表

    report = classification_report(trues, preds, target_names=label_list)
    logger.info(report)
    cm = confusion_matrix(trues, preds)
    cm = pd.DataFrame(cm, columns=[i for i, v in enumerate(label_list)], index=label_list)
    # logger.info(cm)
    for row in cm.itertuples():
        print(" ".join([str(i) for i in row]))

    p, r, f, s = precision_recall_fscore_support(y_true=trues, y_pred=preds, average=None)
    logger.info("p,r,f,s: \n{:},\n{:},\n{:},\n{:}\n ".format(str(p), str(r), str(f), str(s)))


if __name__ == "__main__":
    args = get_args()
    logger = set_params(args=args)
    config, tokenizer, dataset = base_function(args, logger)

    x = import_module('models.' + args.model_name)  ## 通过文件名称引入模型
    model = x.Model(args, config)

    if args.do_train:
        train_function(args, logger, model, tokenizer, dataset)
    elif args.do_test:
        model_path = os.path.join(args.output_dir, args.model_name + "_best.pt")
        # model_path = os.path.join(args.output_dir,args.model_name+"_{}_best.pt".format(args.test_model_idx))
        logger.info("model path:{}".format(model_path))
        test_func(args, logger, model, tokenizer, dataset, model_path)
