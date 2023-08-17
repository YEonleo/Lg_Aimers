import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class BaseModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=512, output_size=21):
        super(BaseModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=3, batch_first=True) # here we changed num_layers from default (1) to 3
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size//2),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(hidden_size//2, output_size)
        )
            
        self.actv = nn.ReLU()
    
    def forward(self, x):
        # x shape: (B, TRAIN_WINDOW_SIZE, 5)
        batch_size = x.size(0)
        hidden = self.init_hidden(batch_size, x.device)
        
        # LSTM layer
        lstm_out, hidden = self.lstm(x, hidden)
        
        # Only use the last output sequence
        last_output = lstm_out[:, -1, :]
        
        # Fully connected layer
        output = self.actv(self.fc(last_output))
        
        return output.squeeze(1)
    
    def init_hidden(self, batch_size, device):
        # Initialize hidden state and cell state
        # Now with num_layers=3
        return (torch.zeros(3, batch_size, self.hidden_size, device=device), 
                torch.zeros(3, batch_size, self.hidden_size, device=device))
