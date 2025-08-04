import argparse, json
from collections import Counter
import os


def generate_json_data(split_path, data_path, max_captions_per_image, min_word_count):
    split = json.load(open(split_path, 'r'))
    word_count = Counter()
    test_word_count =Counter()

    train_img_paths = []
    train_caption_tokens = []
    validation_img_paths = []
    validation_caption_tokens = []
    test_img_paths = []
    test_caption_tokens =[]

 
    for img in split['images']:
        caption_count = 0
        for sentence in img['sentences']:
            if caption_count < max_captions_per_image:
                caption_count += 1
            else:
                break

            if img['split'] == 'train':
                train_img_paths.append('flickr8k_aim3/images/' + img['filename'])
                train_caption_tokens.append(sentence['tokens'])
                word_count.update(sentence['tokens'])
            elif img['split'] == 'val':
                validation_img_paths.append('flickr8k_aim3/images/' + img['filename'])
                validation_caption_tokens.append(sentence['tokens'])
                word_count.update(sentence['tokens'])
            elif img['split'] == 'test':
                test_img_paths.append('flickr8k_aim3/images/' + img['filename'])
                test_caption_tokens.append(sentence['tokens'])
                test_word_count.update(sentence['tokens'])




    words = [word for word in word_count.keys() if word_count[word] >= args.min_word_count]
    word_dict = {word: idx + 4 for idx, word in enumerate(words)}
    word_dict['<start>'] = 0
    word_dict['<eos>'] = 1
    word_dict['<unk>'] = 2
    word_dict['<pad>'] = 3
    
    test_words = [word for word in test_word_count.keys() ]
    test_word_dict = {word: idx + 4 for idx, word in enumerate(test_words)}
    test_word_dict['<start>'] = 0
    test_word_dict['<eos>'] = 1
    test_word_dict['<unk>'] = 2
    test_word_dict['<pad>'] = 3



    with open(data_path + '/word_dict.json', 'w') as f:
        json.dump(word_dict, f)

    with open(data_path + '/test_word_dict.json', 'w') as f:
        json.dump(test_word_dict, f)

    

    train_captions = process_caption_tokens(train_caption_tokens, word_dict)
    validation_captions = process_caption_tokens(validation_caption_tokens, word_dict)
    test_captions =process_caption_tokens(test_caption_tokens,test_word_dict)

    with open(data_path + '/train_img_paths.json', 'w') as f:
        json.dump(train_img_paths, f)
    with open(data_path + '/val_img_paths.json', 'w') as f:
        json.dump(validation_img_paths, f)
    with open(data_path + '/train_captions.json', 'w') as f:
        json.dump(train_captions, f)
    with open(data_path + '/val_captions.json', 'w') as f:
        json.dump(validation_captions, f)
    with open(data_path + '/test_img_paths.json', 'w') as f:
        json.dump(test_img_paths , f)
    with open(data_path+'/test_captions.json', 'w') as f:
        json.dump(test_captions,f)


def process_caption_tokens(caption_tokens, word_dict):
    captions = []
    for tokens in caption_tokens:
        token_idxs = [word_dict[token] if token in word_dict else word_dict['<unk>'] for token in tokens]
        captions.append(
            [word_dict['<start>']] + token_idxs + [word_dict['<eos>']])

    return captions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='参数管理')
    parser.add_argument('--split-path', type=str, default='flickr8k_aim3/dataset_flickr8k.json')
    parser.add_argument('--data-path', type=str, default='data')
    parser.add_argument('--max-captions', type=int, default=5)
    parser.add_argument('--min-word-count', type=int, default=5)
    args = parser.parse_args()

    generate_json_data(args.split_path, args.data_path, args.max_captions, args.min_word_count)
