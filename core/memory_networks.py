# core/memory_networks.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class NeuralTuringMachine(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, memory_dim, num_heads=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        
        self.controller = nn.LSTMCell(input_size, hidden_size)
        
        self.read_heads = nn.ModuleList([
            nn.Linear(hidden_size, memory_dim) for _ in range(num_heads)
        ])
        self.write_heads = nn.ModuleList([
            nn.Linear(hidden_size, memory_dim) for _ in range(num_heads)
        ])
        
        self.memory = nn.Parameter(torch.zeros(memory_size, memory_dim))
        self.usage_vector = nn.Parameter(torch.zeros(memory_size))
        
        self.output_proj = nn.Linear(hidden_size + num_heads * memory_dim, input_size)
        
        nn.init.xavier_uniform_(self.memory)
        
    def forward(self, x, prev_state=None):
        batch_size = x.size(0)
        
        if prev_state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
            memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
            usage = self.usage_vector.unsqueeze(0).expand(batch_size, -1)
        else:
            h, c, memory, usage = prev_state
            
        h_next, c_next = self.controller(x, (h, c))
        
        read_vectors = []
        for head in self.read_heads:
            read_weights = F.softmax(head(h_next), dim=-1)
            read_vec = torch.bmm(read_weights.unsqueeze(1), memory).squeeze(1)
            read_vectors.append(read_vec)
            
        write_vectors = []
        for head in self.write_heads:
            write_weights = F.softmax(head(h_next), dim=-1)
            write_vectors.append(write_weights)
            
        for write_weights in write_vectors:
            memory = memory + torch.bmm(write_weights.unsqueeze(-1), x.unsqueeze(1)).transpose(1, 2)
            
        read_output = torch.cat(read_vectors, dim=-1)
        controller_output = torch.cat([h_next, read_output], dim=-1)
        output = self.output_proj(controller_output)
        
        next_state = (h_next, c_next, memory, usage)
        
        return output, next_state

class DifferentiableNeuralComputer(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, memory_dim, num_read_heads=4, num_write_heads=1):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_read_heads = num_read_heads
        self.num_write_heads = num_write_heads
        
        self.controller = nn.LSTMCell(input_size + num_read_heads * memory_dim, hidden_size)
        
        self.read_weights_proj = nn.Linear(hidden_size, num_read_heads * memory_size)
        self.write_weights_proj = nn.Linear(hidden_size, num_write_heads * memory_size)
        self.erase_vectors_proj = nn.Linear(hidden_size, num_write_heads * memory_dim)
        self.add_vectors_proj = nn.Linear(hidden_size, num_write_heads * memory_dim)
        
        self.memory = nn.Parameter(torch.zeros(memory_size, memory_dim))
        self.link_matrix = nn.Parameter(torch.zeros(memory_size, memory_size))
        
        self.output_proj = nn.Linear(hidden_size, input_size)
        
        nn.init.xavier_uniform_(self.memory)
        nn.init.xavier_uniform_(self.link_matrix)
        
    def _content_based_addressing(self, key, memory, beta):
        similarity = F.cosine_similarity(key.unsqueeze(1), memory, dim=-1)
        weights = F.softmax(beta * similarity, dim=-1)
        return weights
        
    def forward(self, x, prev_state=None):
        batch_size = x.size(0)
        
        if prev_state is None:
            h = torch.zeros(batch_size, self.hidden_size, device=x.device)
            c = torch.zeros(batch_size, self.hidden_size, device=x.device)
            memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
            read_vectors = torch.zeros(batch_size, self.num_read_heads * self.memory_dim, device=x.device)
            link_matrix = self.link_matrix.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            h, c, memory, read_vectors, link_matrix = prev_state
            
        controller_input = torch.cat([x, read_vectors], dim=-1)
        h_next, c_next = self.controller(controller_input, (h, c))
        
        read_weights = F.softmax(self.read_weights_proj(h_next).view(batch_size, self.num_read_heads, self.memory_size), dim=-1)
        read_vectors = torch.bmm(read_weights, memory).view(batch_size, -1)
        
        write_weights = F.softmax(self.write_weights_proj(h_next).view(batch_size, self.num_write_heads, self.memory_size), dim=-1)
        erase_vectors = torch.sigmoid(self.erase_vectors_proj(h_next).view(batch_size, self.num_write_heads, self.memory_dim))
        add_vectors = torch.tanh(self.add_vectors_proj(h_next).view(batch_size, self.num_write_heads, self.memory_dim))
        
        for i in range(self.num_write_heads):
            erase_update = torch.bmm(write_weights[:, i:i+1].transpose(1, 2), erase_vectors[:, i:i+1])
            add_update = torch.bmm(write_weights[:, i:i+1].transpose(1, 2), add_vectors[:, i:i+1])
            memory = memory * (1 - erase_update) + add_update
            
        output = self.output_proj(h_next)
        
        next_state = (h_next, c_next, memory, read_vectors, link_matrix)
        
        return output, next_state

class MemoryAugmentedNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, memory_dim, num_blocks=8):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_blocks = num_blocks
        
        self.encoder = nn.Linear(input_size, hidden_size)
        self.memory_blocks = nn.ModuleList([
            nn.Linear(hidden_size, memory_dim) for _ in range(num_blocks)
        ])
        self.attention = nn.MultiheadAttention(memory_dim, num_heads=8, batch_first=True)
        self.decoder = nn.Linear(memory_dim, input_size)
        
        self.memory = nn.Parameter(torch.zeros(memory_size, memory_dim))
        nn.init.xavier_uniform_(self.memory)
        
    def forward(self, x, prev_memory=None):
        batch_size = x.size(0)
        
        encoded = torch.tanh(self.encoder(x))
        
        if prev_memory is None:
            memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            memory = prev_memory
            
        memory_updates = []
        for block in self.memory_blocks:
            update = block(encoded).unsqueeze(1)
            memory_updates.append(update)
            
        memory_updates = torch.cat(memory_updates, dim=1)
        
        attended_updates, _ = self.attention(memory_updates, memory, memory)
        
        updated_memory = memory + attended_updates.mean(dim=1, keepdim=True)
        
        retrieved, attention_weights = self.attention(
            encoded.unsqueeze(1), updated_memory, updated_memory
        )
        
        output = self.decoder(retrieved.squeeze(1))
        
        return output, updated_memory