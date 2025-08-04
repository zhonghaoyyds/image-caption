
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

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

from dataset import ImageCaptionDataset, collate_fn
from decoder_lstm import Decoder
from encoder import Encoder



data_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

def main(args):

  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    word_dict = json.load(open(args.data + '/word_dict.json', 'r'))
    test_word_dict = json.load(open(args.data + '/test_word_dict.json','r'))
    vocabulary_size = len(word_dict)

    encoder = Encoder(args.network,args.is_finetune)
    decoder = Decoder(vocabulary_size, encoder.dim, args.hidden_dim,args.tf)
    print(encoder.get_finetune_state())
    if args.model:
        if encoder.get_finetune_state():
            checkpoint = torch.load(args.model, map_location=device)
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
        else:
           decoder.load_state_dict(torch.load(args.model, map_location=device, weights_only=True))

    encoder.to(device)
    decoder.to(device)


    test_loader = torch.utils.data.DataLoader(
        ImageCaptionDataset(data_transforms, args.data, split_type='test'),
        batch_size=args.batch_size, shuffle=False, num_workers=32, collate_fn=collate_fn)
    test( encoder, decoder, test_loader, word_dict,test_word_dict, device)

def test(encoder, decoder, data_loader, word_dict, test_word_dict, device):
    encoder.eval()
    decoder.eval()

    references = {}
    hypotheses = {}
    sample_id = 0
    ignore_tokens = {'<start>', '<pad>', '<eos>'}
    idx_to_word = {index: word for word, index in word_dict.items()}
    test_idx_to_word = {index: word for word, index in test_word_dict.items()}

    with torch.no_grad():
        for batch_idx, (imgs, _, _, all_captions) in enumerate(tqdm(data_loader)):
            imgs = imgs.to(device)
            img_features = encoder(imgs)  # [batch, N, D] 


            for i, cap_set in enumerate(all_captions):
                reference = []
                for caption_indices in cap_set:
                    words = []
                    for word_idx in caption_indices:
#                        word_idx = word_idx.item() if isinstance(word_idx, torch.Tensor) else word_idx
                        word = test_idx_to_word[word_idx]
                        if word and word not in ignore_tokens:
                            words.append(word)
                    sentence = ' '.join(words)
                    reference.append(sentence)
                references[sample_id + i] = reference

 
            batch_size = imgs.size(0)
            beam_size = 5
            for i in range(batch_size):
 
                single_img_feat = img_features[i].unsqueeze(0) 
                single_img_feat = single_img_feat.expand(beam_size, -1, -1).contiguous().to(device)  

                pred_cap, _ = decoder.inference(single_img_feat, beam_size=beam_size)  


                words = []
                for idx in pred_cap:
                    word= idx_to_word[idx]
                    if word and word not in ignore_tokens:
                        words.append(word)
                    if word == '<eos>':
                        break
                sentence = ' '.join(words)
                hypotheses[sample_id + i] = [sentence]

            sample_id += batch_size

    eval_captions(references, hypotheses)


def eval_captions(references, hypotheses):
    
    # BLEU
    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(references, hypotheses)
    print(f"BLEU-1: {bleu_score[0]:.4f}")
    print(f"BLEU-2: {bleu_score[1]:.4f}")
    print(f"BLEU-3: {bleu_score[2]:.4f}")
    print(f"BLEU-4: {bleu_score[3]:.4f}")

    # ROUGE
    rouge_scorer = Rouge()
    rouge_score, _ = rouge_scorer.compute_score(references, hypotheses)
    print(f"ROUGE-L: {rouge_score:.4f}")

    # METEOR
    meteor_scorer = Meteor()
    meteor_score, _ = meteor_scorer.compute_score(references, hypotheses)
    print(f"METEOR: {meteor_score:.4f}")

    # CIDEr
    cider_scorer = Cider()
    cider_score, _ = cider_scorer.compute_score(references, hypotheses)
    print(f"CIDEr: {cider_score:.4f}")

    scores = {
        "BLEU-1": bleu_score[0],
        "BLEU-2": bleu_score[1],
        "BLEU-3": bleu_score[2],
        "BLEU-4": bleu_score[3],
        "ROUGE-L": rouge_score,
        "METEOR": meteor_score,
        "CIDEr": cider_score
    }

    return scores
def eval1(references, hypotheses):
    
    # BLEU
    bleu_scorer = Bleu(4)
    bleu_score, _ = bleu_scorer.compute_score(references, hypotheses)
    print(f"BLEU-1: {bleu_score[0]:.4f}")
    print(f"BLEU-2: {bleu_score[1]:.4f}")
    print(f"BLEU-3: {bleu_score[2]:.4f}")
    print(f"BLEU-4: {bleu_score[3]:.4f}")



    scores = {
        "BLEU-1": bleu_score[0],
        "BLEU-2": bleu_score[1],
        "BLEU-3": bleu_score[2],
        "BLEU-4": bleu_score[3],

    }

    return scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练参数设置')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N')
    parser.add_argument('--data', type=str, default='data')
    parser.add_argument('--model', type=str,default='model_new/clip_rn50_1024_finetune_15.pth' )# 'model_new/clip_rn50_2048_frozen_9.pth'
    parser.add_argument('--network', choices=['resnet152', 'clip_rn50','clip_vit','vit_base'], default='clip_rn50')
    parser.add_argument('--tf', action='store_true', default=False)
    parser.add_argument('--hidden_dim',type = int , default =1024)
    parser.add_argument('--is_finetune',type=str,default=False)
    main(parser.parse_args())





   