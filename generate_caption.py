import argparse, json, os
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import skimage
import skimage.transform
import torch
import torchvision.transforms as transforms
from math import ceil
from PIL import Image

from dataset import pil_loader
from decoder_lstm import Decoder
from encoder import Encoder
from train import data_transforms


def generate_caption_visualization(model_path, encoder, decoder, img_path, word_dict, beam_size=3, smooth=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = Image.open(img_path)
    img = data_transforms(img)
    img = img.to(device)
    img = img.unsqueeze(0)

    img_for_vis = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_for_vis = img_for_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_for_vis = np.clip(img_for_vis, 0, 1)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    with torch.no_grad():
        img_features = encoder(img)
        img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
        img_features = img_features.to(device) 
        sentence, alpha = decoder.inference(img_features, beam_size)

    token_dict = {idx: word for word, idx in word_dict.items()}
    sentence_tokens = []
    for word_idx in sentence:
        sentence_tokens.append(token_dict[word_idx])
        if word_idx == word_dict['<eos>']:
            break

    img = Image.open(img_path)
    w, h = img.size
    if w > h:
        w = w * 256 / h
        h = 256
    else:
        h = h * 256 / w
        w = 256
    left = (w - 224) / 2
    top = (h - 224) / 2
    resized_img = img.resize((int(w), int(h)), Image.BICUBIC).crop((left, top, left + 224, top + 224))
    img = np.array(resized_img.convert('RGB').getdata()).reshape(224, 224, 3)
    img = img.astype('float32') / 255

    num_words = len(sentence_tokens)
    w = np.round(np.sqrt(num_words))
    h = np.ceil(np.float32(num_words) / w)
    alpha = torch.tensor(alpha)

    plot_height = ceil((num_words + 3) / 4.0)
    ax1 = plt.subplot(4, plot_height, 1)
    plt.imshow(img_for_vis)
    plt.axis('off')
    for idx in range(num_words):
        ax2 = plt.subplot(4, plot_height, idx + 2)
        label = sentence_tokens[idx]
        plt.text(0, 1, label, backgroundcolor='white', fontsize=13)
        plt.text(0, 1, label, color='black', fontsize=13)
        plt.imshow(img_for_vis)

        if encoder.network == 'vgg19':
            shape_size = 14
        else:
            shape_size = 7

        if smooth:
            alpha_img = skimage.transform.pyramid_expand(alpha[idx, :].reshape(shape_size, shape_size), upscale=16, sigma=20)
        else:
            alpha_img = skimage.transform.resize(alpha[idx, :].reshape(shape_size,shape_size), [img_for_vis.shape[0], img_for_vis.shape[1]])
        
        if alpha_img.shape[:2] != img_for_vis.shape[:2]:
            alpha_img = skimage.transform.resize(alpha_img, (img_for_vis.shape[0], img_for_vis.shape[1]))
        
        plt.imshow(alpha_img, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')
    plt.savefig(model_path + '_caption_result.png')
    print("图片已保存为：", model_path + '_caption_result.png')

def generate_caption_visualization_vit(model_path, encoder, decoder, img_path, word_dict, beam_size=3, smooth=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    img = Image.open(img_path)
    img = data_transforms(img)
    img = img.to(device)
    img = img.unsqueeze(0)

    img_for_vis = img.squeeze(0).permute(1, 2, 0).cpu().numpy()
    img_for_vis = img_for_vis * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
    img_for_vis = np.clip(img_for_vis, 0, 1)

    encoder = encoder.to(device)
    decoder = decoder.to(device)

    with torch.no_grad():
        img_features = encoder(img)
        img_features = img_features.expand(beam_size, img_features.size(1), img_features.size(2))
        img_features = img_features.to(device)
        sentence, alpha = decoder.inference(img_features, beam_size)


    alpha = torch.tensor(alpha)


    seq_len = alpha.shape[1]
    shape_size = int(seq_len ** 0.5)

    token_dict = {idx: word for word, idx in word_dict.items()}
    sentence_tokens = []
    for word_idx in sentence:
        sentence_tokens.append(token_dict[word_idx])
        if word_idx == word_dict['<eos>']:
            break

    img = Image.open(img_path)
    w, h = img.size
    if w > h:
        w = w * 256 / h
        h = 256
    else:
        h = h * 256 / w
        w = 256
    left = (w - 224) / 2
    top = (h - 224) / 2
    resized_img = img.resize((int(w), int(h)), Image.BICUBIC).crop((left, top, left + 224, top + 224))
    img = np.array(resized_img.convert('RGB').getdata()).reshape(224, 224, 3)
    img = img.astype('float32') / 255

    num_words = len(sentence_tokens)
    plot_height = ceil((num_words + 3) / 4.0)

    ax1 = plt.subplot(4, plot_height, 1)
    plt.imshow(img_for_vis)
    plt.axis('off')

    for idx in range(num_words):
        ax2 = plt.subplot(4, plot_height, idx + 2)
        label = sentence_tokens[idx]
        plt.text(0, 1, label, backgroundcolor='white', fontsize=13)
        plt.text(0, 1, label, color='black', fontsize=13)
        plt.imshow(img_for_vis)

        if smooth:
            alpha_img = skimage.transform.pyramid_expand(alpha[idx, :].reshape(shape_size, shape_size),
                                                         upscale=16, sigma=20)
        else:
            alpha_img = skimage.transform.resize(alpha[idx, :].reshape(shape_size, shape_size),
                                                 [img_for_vis.shape[0], img_for_vis.shape[1]])

        if alpha_img.shape[:2] != img_for_vis.shape[:2]:
            alpha_img = skimage.transform.resize(alpha_img, (img_for_vis.shape[0], img_for_vis.shape[1]))

        plt.imshow(alpha_img, alpha=0.8)
        plt.set_cmap(cm.Greys_r)
        plt.axis('off')

    plt.savefig(model_path + '_caption_result_vit.png')
    print("图片已保存为：", model_path + '_caption_result_vit.png')





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='训练参数设置')
    parser.add_argument('--img-path',default='flickr8k_aim3/images/3385593926_d3e9c21170.jpg')
    parser.add_argument('--model',  default='model_new/clip_rn50_1024_finetune_9.pth')
    parser.add_argument('--network', choices=[ 'resnet152','clip_rn50','clip_vit','vit'], default='clip_rn50')
    parser.add_argument('--hidden_dim',type= int ,default=1024)
    parser.add_argument('--data-path',  default='data')
    parser.add_argument('--is_finetune',default='True')
   
    args = parser.parse_args()

    word_dict = json.load(open(args.data_path + '/word_dict.json', 'r'))
    vocabulary_size = len(word_dict)

    encoder = Encoder(network=args.network,is_finetune=args.is_finetune)
    decoder = Decoder(vocabulary_size, encoder.dim,args.hidden_dim)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.model:
        if encoder.get_finetune_state():
            checkpoint = torch.load(args.model, map_location=device)
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
        else:
           decoder.load_state_dict(torch.load(args.model, map_location=device))

    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()
    if args.network =='clip_vit' or args.network=='vit_base':
        generate_caption_visualization_vit(args.model,encoder, decoder, args.img_path, word_dict)
    else:
        generate_caption_visualization(args.model,encoder, decoder, args.img_path, word_dict)
