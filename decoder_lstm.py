import torch
import torch.nn as nn
from attention import Attention


class Decoder(nn.Module):
    def __init__(self, vocabulary_size, encoder_dim, hidden_dim,  tf=False):
        super(Decoder, self).__init__()
        self.use_tf = tf
        self.hidden_dim =hidden_dim
        self.vocabulary_size = vocabulary_size
        self.encoder_dim = encoder_dim

        self.init_h = nn.Linear(encoder_dim, hidden_dim)
        self.init_c = nn.Linear(encoder_dim, hidden_dim)   
        self.tanh = nn.Tanh()

        self.f_beta = nn.Linear(hidden_dim, encoder_dim)
        self.sigmoid = nn.Sigmoid()

        self.deep_output = nn.Linear(hidden_dim, vocabulary_size)
        self.dropout = nn.Dropout(0.5)

        self.attention = Attention(encoder_dim,hidden_dim)
        self.embedding = nn.Embedding(vocabulary_size, hidden_dim)
        self.lstm = nn.LSTMCell(hidden_dim + encoder_dim, hidden_dim,)

    def forward(self, img_features, captions, caption_lens):
        batch_size = img_features.size(0)
        device = img_features.device
        hidden_dim =self.hidden_dim
        h, c = self.get_init_lstm_state(img_features)
        decoder_lens = [c-1 for c in caption_lens]
        max_timespan = max(decoder_lens)
        prev_words = torch.zeros(batch_size, 1).long().to(device)

        if self.use_tf:
            embedding = self.embedding(captions) if self.training else self.embedding(prev_words)
        else:
            embedding = self.embedding(prev_words) # prev_words is the previous word 

        preds = torch.zeros(batch_size, max_timespan, self.vocabulary_size).to(device)
        alphas = torch.zeros(batch_size, max_timespan, img_features.size(1)).to(device)
        for t in range(max(decoder_lens)):
            batch_size_t = sum([l>t for l in decoder_lens ])
            context, alpha = self.attention(img_features[:batch_size_t], h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))
            gated_context = gate * context

            if self.use_tf and self.training:
                lstm_input = torch.cat((embedding[:batch_size_t, t], gated_context), dim=1)
            else:   
                embedding = embedding.squeeze(1) if embedding.dim() == 3 else embedding
                lstm_input = torch.cat((embedding[:batch_size_t], gated_context), dim=1)

            h, c = self.lstm(lstm_input, (h[:batch_size_t], c[:batch_size_t]))
            output = self.deep_output(self.dropout(h))

            preds[:batch_size_t, t] = output
            alphas[:batch_size_t, t] = alpha

            if not self.training or not self.use_tf:
                embedding = self.embedding(output.max(1)[1].reshape(batch_size_t, 1))

        return preds, alphas, decoder_lens

    def get_init_lstm_state(self, img_features):
        avg_features = img_features.mean(dim=1)

        c = self.init_c(avg_features)
        c = self.tanh(c)

        h = self.init_h(avg_features)
        h = self.tanh(h)

        return h, c

    def change_tf(self, tf):
       self.tf = tf


       
    def inference(self, img_features, beam_size):
        device = img_features.device
        prev_words = torch.zeros(beam_size, 1).long().to(device)
        hidden_dim = self.hidden_dim
        sentences = prev_words
        top_preds = torch.zeros(beam_size, 1).to(device)
        alphas = torch.ones(beam_size, 1, img_features.size(1)).to(device)

        completed_sentences = []
        completed_sentences_alphas = []
        completed_sentences_preds = []

        step = 1
        h, c = self.get_init_lstm_state(img_features)

        while True:
            embedding = self.embedding(prev_words).squeeze(1)
            context, alpha = self.attention(img_features, h)
            gate = self.sigmoid(self.f_beta(h))
            gated_context = gate * context

            lstm_input = torch.cat((embedding, gated_context), dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            output = self.deep_output(h)
            output = top_preds.expand_as(output) + output

            if step == 1:
                top_preds, top_words = output[0].topk(beam_size, 0, True, True)
            else:
                top_preds, top_words = output.view(-1).topk(beam_size, 0, True, True)
            prev_word_idxs = top_words // output.size(1)
            next_word_idxs = top_words % output.size(1)
            sentences = torch.cat((sentences[prev_word_idxs], next_word_idxs.unsqueeze(1)), dim=1)
            alphas = torch.cat((alphas[prev_word_idxs], alpha[prev_word_idxs].unsqueeze(1)), dim=1)

            incomplete = [idx for idx, next_word in enumerate(next_word_idxs) if next_word != 1]
            complete = list(set(range(len(next_word_idxs))) - set(incomplete))

            if len(complete) > 0:
                completed_sentences.extend(sentences[complete].tolist())
                completed_sentences_alphas.extend(alphas[complete].tolist())
                completed_sentences_preds.extend(top_preds[complete])
            beam_size -= len(complete)

            if beam_size == 0:
                break
            sentences = sentences[incomplete]
            alphas = alphas[incomplete]
            h = h[prev_word_idxs[incomplete]]
            c = c[prev_word_idxs[incomplete]]
            img_features = img_features[prev_word_idxs[incomplete]]
            top_preds = top_preds[incomplete].unsqueeze(1)
            prev_words = next_word_idxs[incomplete].unsqueeze(1)

            if step > 50:
                break
            step += 1

        if len(completed_sentences_preds) == 0:

            top_pred = top_preds.max().item()
            idx = top_preds.argmax().item()
            completed_sentences = sentences.tolist()
            completed_sentences_alphas = alphas.tolist()
            completed_sentences_preds = [top_pred]
            idx = 0  

        else:
            idx = completed_sentences_preds.index(max(completed_sentences_preds))

        sentence = completed_sentences[idx]
        alpha = completed_sentences_alphas[idx]

        return sentence, alpha
