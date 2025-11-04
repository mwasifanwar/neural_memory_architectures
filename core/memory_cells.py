# core/memory_cells.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class MemoryCell(nn.Module):
    def __init__(self, input_size, memory_size, memory_dim, num_heads=4):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_heads = num_heads
        
        self.memory = nn.Parameter(torch.zeros(memory_size, memory_dim))
        self.write_heads = nn.Linear(input_size, num_heads * memory_dim)
        self.read_heads = nn.Linear(input_size, num_heads * memory_dim)
        self.output_proj = nn.Linear(num_heads * memory_dim, input_size)
        
        nn.init.xavier_uniform_(self.memory)
        
    def forward(self, x, prev_state=None):
        batch_size = x.size(0)
        
        write_weights = torch.softmax(self.write_heads(x).view(batch_size, self.num_heads, self.memory_dim), dim=-1)
        read_weights = torch.softmax(self.read_heads(x).view(batch_size, self.num_heads, self.memory_dim), dim=-1)
        
        memory_expanded = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        
        write_updates = torch.bmm(write_weights.transpose(1, 2), x.unsqueeze(-1)).squeeze(-1)
        updated_memory = memory_expanded + write_updates.unsqueeze(1)
        
        read_output = torch.bmm(read_weights, updated_memory)
        read_output = read_output.view(batch_size, -1)
        
        output = self.output_proj(read_output)
        
        return output, updated_memory

class DynamicMemory(nn.Module):
    def __init__(self, input_size, memory_size, memory_dim, num_slots=8):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_slots = num_slots
        
        self.slot_generator = nn.Linear(input_size, num_slots * memory_dim)
        self.slot_attention = nn.MultiheadAttention(memory_dim, num_heads=8, batch_first=True)
        self.memory_proj = nn.Linear(memory_dim, memory_dim)
        self.output_proj = nn.Linear(num_slots * memory_dim, input_size)
        
        self.memory_slots = nn.Parameter(torch.zeros(num_slots, memory_dim))
        nn.init.xavier_uniform_(self.memory_slots)
        
    def forward(self, x, prev_memory=None):
        batch_size = x.size(0)
        
        if prev_memory is None:
            memory = self.memory_slots.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            memory = prev_memory
            
        query_slots = self.slot_generator(x).view(batch_size, self.num_slots, self.memory_dim)
        
        attended_memory, attention_weights = self.slot_attention(
            query_slots, memory, memory
        )
        
        updated_memory = memory + self.memory_proj(attended_memory)
        
        memory_output = updated_memory.view(batch_size, -1)
        output = self.output_proj(memory_output)
        
        return output, updated_memory

class AssociativeMemory(nn.Module):
    def __init__(self, input_size, memory_size, key_dim, value_dim):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        
        self.key_memory = nn.Parameter(torch.zeros(memory_size, key_dim))
        self.value_memory = nn.Parameter(torch.zeros(memory_size, value_dim))
        
        self.query_proj = nn.Linear(input_size, key_dim)
        self.key_proj = nn.Linear(input_size, key_dim)
        self.value_proj = nn.Linear(input_size, value_dim)
        self.output_proj = nn.Linear(value_dim, input_size)
        
        nn.init.xavier_uniform_(self.key_memory)
        nn.init.xavier_uniform_(self.value_memory)
        
    def forward(self, x, prev_keys=None, prev_values=None):
        batch_size = x.size(0)
        
        query = self.query_proj(x)
        
        if prev_keys is not None and prev_values is not None:
            keys = torch.cat([self.key_memory.unsqueeze(0).expand(batch_size, -1, -1), prev_keys], dim=1)
            values = torch.cat([self.value_memory.unsqueeze(0).expand(batch_size, -1, -1), prev_values], dim=1)
        else:
            keys = self.key_memory.unsqueeze(0).expand(batch_size, -1, -1)
            values = self.value_memory.unsqueeze(0).expand(batch_size, -1, -1)
            
        attention_scores = torch.bmm(query.unsqueeze(1), keys.transpose(1, 2)).squeeze(1)
        attention_weights = F.softmax(attention_scores / math.sqrt(self.key_dim), dim=-1)
        
        retrieved_values = torch.bmm(attention_weights.unsqueeze(1), values).squeeze(1)
        output = self.output_proj(retrieved_values)
        
        new_key = self.key_proj(x).unsqueeze(1)
        new_value = self.value_proj(x).unsqueeze(1)
        
        if prev_keys is not None:
            updated_keys = torch.cat([prev_keys, new_key], dim=1)
            updated_values = torch.cat([prev_values, new_value], dim=1)
        else:
            updated_keys = new_key
            updated_values = new_value
            
        return output, updated_keys, updated_values