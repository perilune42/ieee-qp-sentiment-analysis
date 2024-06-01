import torch
import torch.nn as nn
import re
import numpy as np

class SentimentAnalysisModel(nn.Module):
    """
    Model for sentiment analysist.
    """
    def __init__(self, vocab_size, output_dim, embedding_dim, hidden_dim, n_layers, drop_prob, large):
        """
        Initialize the model by setting up the layers
        """
        super().__init__()
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim

        #Embedding and LSTM layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=drop_prob, batch_first=True)

        #dropout layer
        self.dropout = nn.Dropout(0.3)

        #Linear and sigmoid layer
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16,output_dim)
        self.sigmoid = nn.Sigmoid()

        if large:
            self.load_state_dict(torch.load('model_weights.pth', map_location=torch.device('cpu')))
        else:
            self.load_state_dict(torch.load('model_weights_smaller.pth', map_location=torch.device('cpu')))
        self.eval()
        if large:
            self.word_freq = np.load('word_freq.npy',allow_pickle='TRUE').item()
        else:
            self.word_freq = np.load('word_freq_1.npy',allow_pickle='TRUE').item()

    def forward(self, x, hidden):
        """
        Perform a forward pass of our model on some input and hidden state.
        """
        batch_size = x.size()

        #Embadding and LSTM output
        embeds = self.embedding(x)
        lstm_out, hidden = self.lstm(embeds, hidden)

        #stack up the lstm output
        lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)

        #dropout and fully connected layers
        out = self.dropout(lstm_out)
        out = self.fc1(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.dropout(out)
        out = self.fc3(out)
        out = self.dropout(out)
        out = self.fc4(out)
        sig_out = self.sigmoid(out)

        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]

        return sig_out, hidden

    def init_hidden(self, batch_size):
        """Initialize Hidden STATE"""
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters()).data

        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())

        return hidden
    
    def predict_text(self, text):
        word_seq = np.array([self.word_freq[preprocess_string(word)] for word in text.split()
                         if preprocess_string(word) in self.word_freq.keys()])
        word_seq = np.expand_dims(word_seq,axis=0)
        inputs =  torch.from_numpy(padding_(word_seq,50))
        batch_size = 1
        h = self.init_hidden(batch_size)
        h = tuple([each.data for each in h])
        output, h = self(inputs, h)
        return(output.item())



def preprocess_string(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)
    return s

def padding_(sentences, seq_len):
    features = np.zeros((len(sentences), seq_len),dtype=int)
    for ii, review in enumerate(sentences):
        if len(review) != 0:
            features[ii, -len(review):] = np.array(review)[:seq_len]
    return features

if __name__ == "__main__":
    vocab_size = 66123 # +1 for the 0 padding
    output_size = 1
    embedding_dim = 400
    hidden_dim = 128
    n_layers = 2
    model = SentimentAnalysisModel(vocab_size, output_size, embedding_dim, hidden_dim, n_layers)



    
    while (1):

        input_text = input("input your text: ")
        print(input_text)
        print('='*70)
        pro = model.predict_text(input_text)
        status = "positive" if pro > 0.5 else "negative"
        pro = (1 - pro) if status == "negative" else pro
        print(f'Predicted sentiment is {status} with a probability of {pro}')