import torch.nn as nn
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class LSTMNet(nn.Module):
    
    def __init__(self, vocab_size=6972, embed_dim=128, hidden_dim=32, nb_classes=2, n_layers=2,
                 **kwargs):
        super(LSTMNet, self).__init__()
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.dp_mode = kwargs['dp_mode']

        print(self.dp_mode)
        # Embedding layer converts integer sequences to vector sequences
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=self.embed_dim, padding_idx=0)
        
        # LSTM layer process the vector sequences 
        self.lstm_1 = nn.LSTM(input_size=self.embed_dim, hidden_size=self.hidden_dim,
                              num_layers=self.n_layers, bidirectional=True,
                              dropout=0.2, batch_first=True)
        self.tanh = nn.Tanh()

        self.lstm_2 = nn.LSTM(input_size=self.hidden_dim*2, hidden_size=self.hidden_dim,
                              num_layers=self.n_layers, bidirectional=True, batch_first=True)

        # 2 for bidirectional
        self.fc = nn.Linear(in_features=self.hidden_dim*2, out_features=nb_classes)
        
        # Prediction activation function
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x_batch):
        embedded = self.embedding(x_batch)

        # 2 for bidirectional
        hidden_0 = torch.randn(2*self.n_layers, len(x_batch), self.hidden_dim).to(device)
        carry_0 = torch.randn(2*self.n_layers, len(x_batch), self.hidden_dim).to(device) 

        if not self.dp_mode:
            self.lstm_1.flatten_parameters()  # Add this line

        output, (hidden_1, carry_1) = self.lstm_1(embedded, (hidden_0, carry_0))
    
        output = self.tanh(output) 

        if not self.dp_mode:
            self.lstm_2.flatten_parameters()

        self.lstm_2.flatten_parameters()  # Add this line
        output, _ = self.lstm_2(output, (hidden_1, carry_1)) 
    
        output = self.fc(output[:, -1, :]) 
        output = self.softmax(output) 
    
        return output

