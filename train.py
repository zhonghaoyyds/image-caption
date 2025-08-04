import argparse, json
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence
from torchvision import transforms

from dataset import ImageCaptionDataset, collate_fn
from decoder_lstm import Decoder
from encoder import Encoder
from utils import AverageMeter, accuracy,save_model,log_train
from eval import eval_captions,eval1

data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def main(args):
  

  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    word_dict = json.load(open(args.data + '/word_dict.json', 'r'))
    vocabulary_size = len(word_dict)

    encoder = Encoder(args.network, args.is_finetune)
    decoder = Decoder(vocabulary_size, encoder.dim, args.hidden_dim,args.tf)
    print(encoder.get_finetune_state())
    for param in encoder.parameters():
        param.requires_grad = False
    print(encoder.dim)
    print(decoder.parameters())

    if args.model:
        if encoder.get_finetune_state():
            checkpoint = torch.load(args.model, map_location=device)
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
        else:
           decoder.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))

    encoder.to(device)
    decoder.to(device)
    if encoder.get_finetune_state():

        encoder_params = list(filter(lambda p: p.requires_grad, encoder.parameters()))
        decoder_params = list(decoder.parameters())
  
        optimizer = optim.Adam([
             {'params': encoder_params, 'lr': args.lr * 0.1},  
         {'params': decoder_params, 'lr': args.lr},        
          ])
    else:
        optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    
    scheduler = optim.lr_scheduler.StepLR(optimizer, args.step_size)
    cross_entropy_loss = nn.CrossEntropyLoss().to(device)

    train_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, args.data),
        batch_size=args.batch_size, shuffle=True, num_workers=32 ,collate_fn=collate_fn)

    val_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, args.data, split_type='val'),
        batch_size=args.batch_size, shuffle=False , num_workers=32, collate_fn=collate_fn)

        

    print('Starting training with {}'.format(args))

    best_cider = 0.0
    epochs_no_improve = 0
    patience = args.patience
    
    for epoch in range(1, args.epochs + 1):
        
        train(epoch, encoder, decoder, optimizer, cross_entropy_loss,
                train_loader, args.alpha_c, args.log_interval, device)
        
        score = validate(epoch, encoder, decoder, cross_entropy_loss, val_loader,
                            word_dict, args.alpha_c, args.log_interval, device) 

        scheduler.step()
        if encoder.get_finetune_state():
            model_file = 'model_new/' + args.network + '_'+str(args.hidden_dim)+'_finetune_' + str(epoch) + '.pth'
        else:
            model_file = 'model_new/' + args.network + '_'+str(args.hidden_dim)+'_frozen_' + str(epoch) + '.pth'

        if score['BLEU-4']>best_cider:
            model_path =model_file
        best_cider, epochs_no_improve, is_stop = save_model(score['BLEU-4'], best_cider, epochs_no_improve, patience,
                                                                               encoder, decoder, optimizer, epoch, model_file, args)

        if is_stop:
            print(f'提前终止训练，连续 {patience} 轮无 CIDEr 提升,最高得分是: {best_cider}')
            break
    log_train(args,best_cider,model_path)



def train(epoch, encoder, decoder, optimizer, cross_entropy_loss, data_loader, alpha_c, log_interval,  device):
    encoder.eval()
    decoder.train()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    
    for batch_idx, (imgs, captions, caption_lens, all_captions) in enumerate(tqdm(data_loader)):
        imgs, captions = Variable(imgs).to(device), Variable(captions).to(device)
        img_features = encoder(imgs)
        optimizer.zero_grad()

        preds, alphas, decoder_lens = decoder(img_features, captions, caption_lens)
        targets = captions[:, 1:]

        targets = pack_padded_sequence(targets, decoder_lens, batch_first=True)[0]
        preds = pack_padded_sequence(preds, decoder_lens, batch_first=True)[0]

        att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()

        loss = cross_entropy_loss(preds, targets)
        loss += att_regularization
        loss.backward()
        optimizer.step()

        total_caption_length = sum(decoder_lens)
        acc1 = accuracy(preds, targets, 1)
        acc5 = accuracy(preds, targets, 5)
        losses.update(loss.item(), total_caption_length)
        top1.update(acc1, total_caption_length)
        top5.update(acc5, total_caption_length)

        if batch_idx % log_interval == 0:
            print('Train Batch: [{0}/{1}]\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                      batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))



def validate(epoch, encoder, decoder, cross_entropy_loss, data_loader, word_dict,alpha_c, log_interval, device):
    encoder.eval()
    decoder.eval()

    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    references = {}
    hypotheses = {}
    sample_id = 0
    ignore_tokens = {'<start>', '<pad>', '<eos>'}
    idx_to_word = {index: word for word, index in word_dict.items()}
    with torch.no_grad():
        for batch_idx, (imgs, captions, caption_lens, all_captions) in enumerate(tqdm(data_loader)):
            imgs, captions = Variable(imgs).to(device), Variable(captions).to(device)

            img_features = encoder(imgs)
            preds, alphas, decoder_lens = decoder(img_features, captions, caption_lens)
            targets = captions[:, 1:]

            targets = pack_padded_sequence(targets, decoder_lens, batch_first=True)[0]
            packed_preds = pack_padded_sequence(preds, decoder_lens, batch_first=True)[0]

            att_regularization = alpha_c * ((1 - alphas.sum(1))**2).mean()

            loss = cross_entropy_loss(packed_preds, targets)
            loss += att_regularization

            total_caption_length = sum(decoder_lens)
            acc1 = accuracy(packed_preds, targets, 1)
            acc5 = accuracy(packed_preds, targets, 5)
            losses.update(loss.item(), total_caption_length)
            top1.update(acc1, total_caption_length)
            top5.update(acc5, total_caption_length)

            for i, cap_set in enumerate(all_captions):
                reference = []
                for caption_indices in cap_set:
                    words = []
                    for idx in caption_indices:
#                        idx = idx.item()
                        word = idx_to_word[idx]
                        if word and word not in ignore_tokens:
                            words.append(word)
                    sentence = ' '.join(words)
                    reference.append(sentence)
                references[sample_id + i] = reference


            pred_caps= torch.max(preds,dim=2)[1]
            for i,pred_cap in enumerate(pred_caps):
                words = []
                for idx in pred_cap:
                    idx = idx.item()
                    word= idx_to_word[idx]
                    if word and word not in ignore_tokens:
                        words.append(word)
                    if word == '<eos>':
                        break
                sentence = ' '.join(words)
                hypotheses[sample_id + i] = [sentence]

            sample_id += len(pred_caps)

            if batch_idx % 20 == 0:
                print('Validation Batch: [{0}/{1}]\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Top 1 Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Top 5 Accuracy {top5.val:.3f} ({top5.avg:.3f})'.format(
                          batch_idx, len(data_loader), loss=losses, top1=top1, top5=top5))

        print(f'epoch : {epoch} ')
        score = eval1(references,hypotheses)

    
        return score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练参数设置')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--epochs', type=int, default=20, metavar='E')
    parser.add_argument('--patience',type = int, default = 5 )
    parser.add_argument('--lr', type=float, default=2e-4, metavar='LR')
    parser.add_argument('--step-size', type=int, default=5,)
    parser.add_argument('--alpha-c', type=float, default=1, metavar='A')
    parser.add_argument('--log-interval', type=int, default=100, metavar='L')
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--model', type=str,default=None )# 'model_new/clip_rn50_2048_frozen_9.pth'
    parser.add_argument('--network', choices=['resnet152',  'clip_rn50','clip_vit','vit_base'], default='clip_vit')

    parser.add_argument('--tf', action='store_true', default=True)
    parser.add_argument('--hidden_dim',type = int , default =1024)
    parser.add_argument('--is_finetune',type= str ,default='False')
    main(parser.parse_args())
 