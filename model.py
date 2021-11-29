import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.vocab_size = vocab_size
        
        self.embed_size = embed_size
        
        self.word_embedding = nn.Embedding(self.vocab_size,self.embed_size)
        
        self.lstm = nn.LSTM(self.embed_size, self.hidden_size, num_layers = 1, batch_first = True)
        
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)
        
  
    def forward(self, features, captions):
        
        features = features.view(features.shape[0], 1, -1)  #features_shape = [batch_size, 1, embed_size] = [1, 1, 512]
        
        captions_without_end = captions[:,:-1]              # Remove the end word from captions.  
                    
        embeds = self.word_embedding(captions_without_end)  #to convert captions of same dim captions=[batch_size,seq_len-1,embed_size)
        
        inputs = torch.cat((features, embeds), dim =1)      #Concat input img vector and embeddings inputs=[batch_size,seq_len, embed_size)
                         
        lstm_out,_ = self.lstm(inputs, None)                # lstm_out_shape = [ batch_size, seq_len, hidden_size]
            
        lstm_out = lstm_out.reshape(-1, self.hidden_size)   #lstm_out_reshape = [ batch_size*seq_len, hidden_size]
            
        output = self.linear(lstm_out)                      #output_shape = [ batch_size*seq_len, vocab_size]
                
        output = output.reshape(captions.shape[0],captions.shape[1], self.vocab_size) 
       
          
        return output
        

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        outputs  = []
        
        hidden = (torch.randn(1, 1, 512).to(inputs.device),
                  torch.randn(1, 1, 512).to(inputs.device))
        
        while True:
        
            lstm_out, hidden = self.lstm(inputs, hidden)

            output = self.linear(lstm_out.reshape(-1, self.hidden_size))
        
            _, max_predict_index = torch.max(output, dim = 1)
            
            word_idx = max_predict_index.cpu().numpy()[0].item()
            
            outputs.append(word_idx)
            
            if len(outputs) == max_len:
                
                return outputs
                
            elif (word_idx == 1):
                     
                return outputs
            
            else:
                
                inputs = self.word_embedding(max_predict_index)
            
                inputs = torch.unsqueeze(inputs, dim = 1)

           
    