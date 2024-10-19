import torch
import torch.nn as nn

class RNN(nn.Module):
    
    def __init__(self, embedding_matrix, hidden_dim, output_dim, num_layers):
        super(RNN, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        # embedding layer
        vocab_size, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(embedding_matrix), freeze=True)
        
        # rnn layer
        self.rnn = nn.RNN(embedding_dim, hidden_dim, num_layers, batch_first=True)
        
        # fc layer
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    
    def forward(self, x):
        
        x = self.embedding(x)
        
        output, hidden = self.rnn(x)
        
        # use last hiddent state as sentence representation
        sentence_representation = hidden[-1]
        
        # output layer
        logits = self.fc(sentence_representation)
        
        return logits
        
        
        
        
        
        
        


        
        
        
        
        
