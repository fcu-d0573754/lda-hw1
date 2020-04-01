import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

import random
import time
import math
import argparse
import utils
import pickle
from torch.utils import data
from dataset import Seq2SeqDataset
import json


parser = argparse.ArgumentParser()
# parser.add_argument('--data_path', type=str, default='./data/train.jsonl')
parser.add_argument('--attention', type=str, default=None)
parser.add_argument('--output_dir', type=str, default='./datasets/seq2seq')#train.jsonl
# parser.add_argument('--eval', action='store_true', default=False)
parser.add_argument('--output_path',
                    type=str,
                    default='./seq2seq_attention_ans.txt')#predict end

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


learning_rate = 0.0001
decoder_learning_rate_ratio = 5
l2 = 0
batch_size = 8
n_iters = 20
clip = 50.0
hidden_size = 512



def sequence_mask(sequence_length, max_len=None):
    if max_len is None:
        max_len = sequence_length.max()
    batch_size = sequence_length.size(0)
    seq_range = torch.arange(0, max_len).long()
    seq_range_expand = seq_range.unsqueeze(0).expand(batch_size, max_len)
    seq_range_expand = seq_range_expand.to(device=device)
    seq_length_expand = (
        sequence_length.unsqueeze(1).expand_as(seq_range_expand))
    return seq_range_expand < seq_length_expand


def masked_cross_entropy(logits, target, length):
    length = torch.LongTensor(length).to(device=device)
    """
    Args:
        logits: A Variable containing a FloatTensor of size
            (batch, max_len, num_classes) which contains the
            unnormalized probability for each class.
        target: A Variable containing a LongTensor of size
            (batch, max_len) which contains the index of the true
            class for each corresponding step.
        length: A Variable containing a LongTensor of size (batch,)
            which contains the length of each data in a batch.
    Returns:
        loss: An average loss value masked by the length.
    """

    # logits_flat: (batch * max_len, num_classes)
    logits_flat = logits.view(-1, logits.size(-1))
    # log_probs_flat: (batch * max_len, num_classes)
    log_probs_flat = F.log_softmax(logits_flat, dim=1)
    # target_flat: (batch * max_len, 1)
    target_flat = target.view(-1, 1)
    # losses_flat: (batch * max_len, 1)
    losses_flat = -torch.gather(log_probs_flat, dim=1, index=target_flat)
    # losses: (batch, max_len)
    losses = losses_flat.view(*target.size())
    # mask: (batch, max_len)
    mask = sequence_mask(sequence_length=length, max_len=target.size(1))
    losses = losses * mask.float()
    loss = losses.sum(dim=(0, 1)) / length.float().sum(dim=0)
    return loss



class Encoder(nn.Module):
    def __init__(self, input_size, embed_size, hidden_size, n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.embed = nn.Embedding(input_size, embed_size)
        self.gru = nn.GRU(embed_size, 
                          hidden_size, 
                          n_layers, 
                          dropout=dropout, 
                          bidirectional=True)
    
    def forward(self, input_seqs, input_lengths, hidden=None):
        # for b in range(input_seqs.transpose(0, 1).size(0)):
        #     count = 0
        #     for i in range(input_seqs.transpose(0,1).size(1)):
        #         if input_seqs.transpose(0,1)[b][i] != 0:
        #             count += 1
        #         else:
        #             break
        #     print(count)
        embedded = self.embed(input_seqs)
        # print(input_lengths)
        # print('encoder input{} embed{}'.format(input_seqs.size(), embedded.size()))
        packed = torch.nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths)
        outputs, hidden = self.gru(packed, hidden)
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        # outputs, hidden = self.gru(embedded, hidden)
        outputs = outputs[:, :, :
                          self.hidden_size] + outputs[:, :,
                                                 self.hidden_size:]  # Sum bidirectional outputs
        return outputs, hidden
    
    
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies, dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.relu(self.attn(torch.cat([hidden, encoder_outputs], 2)))
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1) 


class Decoder(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size, n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        
        self.embed = nn.Embedding(output_size, embed_size)
        self.dropout = nn.Dropout(dropout, inplace=True)
        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size+embed_size, 
                          hidden_size,
                          n_layers,
                          dropout=dropout)
        self.out = nn.Linear(hidden_size*2, output_size)
    
    def forward(self, input_seq, last_hidden, encoder_outputs):
        if input_seq.size(0) != 1:
            raise Exception('decoder input should have 1 step') 
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(1)
        embedded = self.embed(input_seq)#(1,B,N)
        # print('decoder input{} embed{}'.format(input_seq.size(), embedded.size()), end='\r')
        # print(embedded.size())
        embedded = self.dropout(embedded)
        # print(embedded.size())
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs) 
        context = attn_weights.bmm(encoder_outputs.transpose(0,1))#(B,1,N)
        # print(embedded.size())
        context = context.transpose(0,1)#(1,b,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden= self.gru(rnn_input, last_hidden)
        output = output.squeeze(0) #(1,B,N) -> (B.N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1) )# B x E
        return output.unsqueeze(0), hidden, attn_weights


class Seq2seq_new(nn.Module):
    def __init__(self, args):
        super(Seq2seq_new, self).__init__()

        with open(args.output_dir+'/embedding.pkl', 'rb') as f:
            self.pre_embedding = pickle.load(f)
        with open( './datasets/seq2seq/'+ 'vocab.pkl', 'rb') as f:
            self.dict = pickle.load(f)
        self.input_size = self.pre_embedding.vectors.size(0)
        self.embed_size = self.pre_embedding.vectors.size(1)
        self.encoder = Encoder(self.input_size, self.embed_size, hidden_size)
        self.decoder = Decoder(self.embed_size, hidden_size, self.input_size)
        
    def forward(self, input_seqs, target_seqs, len_text, len_summary, teacher_forcing_ratio=1):
        # input_batches, input_lengths, target_batches, target_lengths = pair_data
        batch_size = input_seqs.size(1)
        # print('input size {}'.format(input_seqs.size()))
        # print(len_summary)
        max_target_length = max(len_summary)
        encoder_output, hidden = self.encoder(input_seqs, len_text)
        hidden = hidden[:self.decoder.n_layers]
        
        decoder_input = torch.LongTensor(
            [self.dict['<s>']]*batch_size).unsqueeze(0).to(device)
        all_decoder_outputs = torch.zeros(
            max_target_length, batch_size, 
            self.decoder.output_size).to(device)
        all_decoder_idx = torch.zeros(#[len_suammry * batch]
            max_target_length, batch_size
        ).to(device)

        for i in range(max_target_length):
            # print(decoder_input.size())
            output, hidden, attn_weight = self.decoder(decoder_input, hidden, encoder_output)
            all_decoder_outputs[i] = output
            is_teacher = random.random() < teacher_forcing_ratio
            if is_teacher :
                decoder_input = target_seqs[i].unsqueeze(0)
                # print('is_teacher :{}'.format(decoder_input.size()))
            else:
                _, topi = output.squeeze(0).topk(1)  
                decoder_input = torch.LongTensor([[
                    topi[i][0] for i in range(batch_size)
                ]]).to(device=device)
                if is_teacher == 0:
                    all_decoder_idx[i] = decoder_input
        return all_decoder_outputs, all_decoder_idx


def train(args, train, valid):
    start = time.time()
    t_batch =  len(train)
    total_loss = 0
    model = Seq2seq_new(args).to(device)
    model.train()
    with open(args.output_dir+'/embedding.pkl', 'rb') as f:
        pre_embedding = pickle.load(f)  
    #load embed layer parameters
    model.encoder.embed.weight.data.copy_(
        torch.FloatTensor(pre_embedding.vectors)
    )
    model.decoder.embed.weight.data.copy_(
        torch.FloatTensor(pre_embedding.vectors)
    )
    
    #optimizer 
    encoder_optimizer = optim.Adam(model.encoder.parameters(),
                                    lr=learning_rate,
                                    weight_decay=l2,
                                    )
    decoder_optimizer = optim.Adam(model.decoder.parameters(),
                                    lr=learning_rate *
                                    decoder_learning_rate_ratio,
                                    weight_decay=l2,
                                    )
    
    best_val_loss = None
    for epoch in range(n_iters):
        total_loss = 0
        count = 0

        #train
        for i, train_batch in enumerate(train):
            input_seqs = torch.LongTensor(train_batch['text']).transpose(0, 1).to(device)
            target_seqs = torch.LongTensor(train_batch['summary']).transpose(0, 1).to(device)
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            outputs, outputs_idx = model(input_seqs, target_seqs, train_batch['len_text'], train_batch['len_summary'], 1)
            loss = masked_cross_entropy(
                outputs.transpose(0, 1).contiguous(),
                target_seqs.transpose(0,1).contiguous(),
                train_batch['len_summary']
                )
            
            loss.backward()
            encoder_clip = torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), clip)
            decoder_clip = torch.nn.utils.clip_grad_norm_(model.decoder.parameters(), clip)
            encoder_optimizer.step()
            decoder_optimizer.step()
            
            total_loss += loss.item()
            count += 1
            print('[ Epoch{}: {}/{} ] loss:{:.3f}  '.format(
            	epoch+1, i+1, t_batch, loss.item()), end='\r')
        print('\nTrain |time:{} |Loss:{:.5f} '.format(time.strftime("%H:%M:%S",time.localtime()), total_loss/count))
        

        #evaluation 
        model.eval()
        with torch.no_grad():
            total_loss = 0
            count = 0
            for i, val_batch in enumerate(valid):
                input_seqs = torch.LongTensor(train_batch['text']).transpose(0, 1).to(device)
                target_seqs = torch.LongTensor(train_batch['summary']).transpose(0, 1).to(device)  
                outputs, outputs_idx = model(input_seqs, target_seqs, train_batch['len_text'], train_batch['len_summary'], 1)
                loss = masked_cross_entropy(
                    outputs.transpose(0, 1).contiguous(),
                    target_seqs.transpose(0,1).contiguous(),
                    train_batch['len_summary']
                    )
                total_loss += loss.item()
                count += 1
                avg_val_loss = total_loss / count
            print('\nValid |time:{} |Loss:{:.5f} '.format(time.strftime("%H:%M:%S",time.localtime()), total_loss/count))
            
            #save model
            if not best_val_loss or avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model, './model/seq2seq_attention.pt')
        print('-----------------------------------------------')
        model.train() #new epoch start


#########  output  format   ######################################
##{"id": "2000000", "predict": "Your summary of the articles"}
##{"id": "2000001", "predict": "should be here."}
##################################################################
def predict(args, test, model):
    unk_token = 3
    with open(args.output_dir+'/embedding.pkl', 'rb') as f:
        embedding = pickle.load(f)
    # print('unk_token:{}'.format(embedding.vocab[3]))
    model.eval()
    t_batch = len(test)
    count = 0
    with torch.no_grad():
        with open(args.output_path, 'w') as f:
            for i, test_batch in enumerate(test):
                summaries = []
                ids = []
                input_seqs = torch.LongTensor(test_batch['text']).transpose(0, 1).to(device)
                input_tgr = torch.LongTensor(test_batch['summary']).transpose(0, 1).to(device)
                input_id = test_batch['id']
                ids = input_id
                # print('input_id:{}'.format(input_id))
                outputs, output_idx = model(input_seqs, input_tgr, test_batch['len_text'], test_batch['len_summary'], 0)
                output_idx = output_idx.transpose(0, 1) #[batch * len]
                for batch in range(output_idx.size(0)):
                    summary = []
                    # print('idx:{}'.format(int(output_idx[batch][i].item())))
                    for i in range(output_idx.size(1)):
                        if embedding.vocab[int(output_idx[batch][i].item())] == '</s>':
                            # summary.append(embedding.vocab[unk_token])
                            break
                        else:
                            summary.append(embedding.vocab[int(output_idx[batch][i].item())])
                    summaries.append(summary)
                
                objs = [{
                    'id': idx,
                    'predict': ' '.join(summary)
                } for idx, summary in zip(ids, summaries)]

                objs = sorted(objs, key=lambda x: x['id'])
                for obj in objs:
                    f.write(json.dumps(obj) + '\n')
                count += 1
                print('load {}/{} batch prdiction into josn'.format(count , t_batch))
    
      
         
    
if __name__ == '__main__':
    args = parser.parse_args()
    # # seq2seq = Seq2seq(args)
    # train(args)
    
    
    train_flag = False
    if train_flag:
        print('loading training data...')
        with open(args.output_dir+'/train.pkl', 'rb') as f:
            train_dataset = pickle.load(f)
        with open(args.output_dir+'/valid.pkl', 'rb') as f:
            val_dataset = pickle.load(f)
        # 把data 轉成 batch of tensors

        train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                                    batch_size = batch_size,
                                                    shuffle = True,
                                                    num_workers = 8,
                                                    collate_fn = train_dataset.collate_fn
                                                    )
        val_loader = torch.utils.data.DataLoader(dataset = val_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8,
                                            collate_fn = train_dataset.collate_fn
                                            )
        train(args, train_loader, val_loader)

    #predict 
    else:
        print('loading testing data...')
        with open(args.output_dir+'/valid.pkl', 'rb') as f:
            test_dataset = pickle.load(f)
        test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                            batch_size = batch_size,
                                            shuffle = False,
                                            num_workers = 8,
                                            collate_fn = test_dataset.collate_fn
                                            )
        print('load model...')
        model = torch.load('./model/seq2seq_attention.pt').to(device)

        outputs = predict(args, test_loader, model)




   
    
    
                
                
                
                     
                       
            
        
        
        
        
        
        
        
        
        
        
