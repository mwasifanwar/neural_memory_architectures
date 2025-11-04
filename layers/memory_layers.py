# layers/memory_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MemoryLayer(nn.Module):
    def __init__(self, input_size, memory_size, memory_dim, num_heads=4):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        
        self.memory = nn.Parameter(torch.zeros(memory_size, memory_dim))
        self.input_proj = nn.Linear(input_size, memory_dim)
        self.output_proj = nn.Linear(memory_dim, input_size)
        self.attention = nn.MultiheadAttention(memory_dim, num_heads, batch_first=True)
        
        nn.init.xavier_uniform_(self.memory)
        
    def forward(self, x, prev_memory=None):
        batch_size = x.size(0)
        
        if prev_memory is None:
            memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            memory = prev_memory
            
        projected_input = self.input_proj(x).unsqueeze(1)
        
        attended_output, attention_weights = self.attention(
            projected_input, memory, memory
        )
        
        output = self.output_proj(attended_output.squeeze(1))
        
        memory_update = projected_input
        updated_memory = memory + memory_update
        
        return output, updated_memory

class RecurrentMemoryLayer(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, memory_dim):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        self.rnn = nn.GRUCell(input_size, hidden_size)
        self.memory_proj = nn.Linear(hidden_size, memory_dim)
        self.read_proj = nn.Linear(hidden_size, memory_dim)
        self.output_proj = nn.Linear(hidden_size + memory_dim, input_size)
        
        self.memory = nn.Parameter(torch.zeros(memory_size, memory_dim))
        nn.init.xavier_uniform_(self.memory)
        
    def forward(self, x, prev_state=None):
        batch_size = x.size(0)
        
        if prev_state is None:
            hidden = torch.zeros(batch_size, self.hidden_size, device=x.device)
            memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            hidden, memory = prev_state
            
        hidden_next = self.rnn(x, hidden)
        
        memory_query = self.read_proj(hidden_next).unsqueeze(1)
        similarity = F.cosine_similarity(memory_query, memory, dim=-1)
        read_weights = F.softmax(similarity, dim=-1)
        
        retrieved = torch.bmm(read_weights.unsqueeze(1), memory).squeeze(1)
        
        memory_update = self.memory_proj(hidden_next).unsqueeze(1)
        updated_memory = memory + memory_update
        
        output = self.output_proj(torch.cat([hidden_next, retrieved], dim=-1))
        
        return output, (hidden_next, updated_memory)

class TransformerMemoryLayer(nn.Module):
    def __init__(self, input_size, memory_size, memory_dim, num_heads=8, num_layers=2):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        
        self.input_proj = nn.Linear(input_size, memory_dim)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=memory_dim,
            nhead=num_heads,
            dim_feedforward=memory_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.memory_proj = nn.Linear(memory_dim, input_size)
        self.memory = nn.Parameter(torch.zeros(memory_size, memory_dim))
        
        nn.init.xavier_uniform_(self.memory)
        
    def forward(self, x, prev_memory=None):
        batch_size = x.size(0)
        
        if prev_memory is None:
            memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            memory = prev_memory
            
        projected_input = self.input_proj(x).unsqueeze(1)
        
        transformer_input = torch.cat([memory, projected_input], dim=1)
        
        transformer_output = self.transformer(transformer_input)
        
        memory_output = transformer_output[:, :self.memory_size]
        input_output = transformer_output[:, self.memory_size:]
        
        output = self.memory_proj(input_output.squeeze(1))
        
        return output, memory_output