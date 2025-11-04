# core/attention_memory.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class AttentionMemory(nn.Module):
    def __init__(self, input_size, memory_size, key_dim, value_dim, num_heads=8):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.num_heads = num_heads
        
        self.key_proj = nn.Linear(input_size, num_heads * key_dim)
        self.value_proj = nn.Linear(input_size, num_heads * value_dim)
        self.query_proj = nn.Linear(input_size, num_heads * key_dim)
        self.output_proj = nn.Linear(num_heads * value_dim, input_size)
        
        self.memory_keys = nn.Parameter(torch.zeros(memory_size, num_heads * key_dim))
        self.memory_values = nn.Parameter(torch.zeros(memory_size, num_heads * value_dim))
        
        nn.init.xavier_uniform_(self.memory_keys)
        nn.init.xavier_uniform_(self.memory_values)
        
    def forward(self, x, prev_keys=None, prev_values=None):
        batch_size = x.size(0)
        
        query = self.query_proj(x).view(batch_size, self.num_heads, self.key_dim)
        
        if prev_keys is not None and prev_values is not None:
            keys = torch.cat([
                self.memory_keys.unsqueeze(0).expand(batch_size, -1, -1).view(batch_size, self.memory_size, self.num_heads, self.key_dim),
                prev_keys
            ], dim=1)
            values = torch.cat([
                self.memory_values.unsqueeze(0).expand(batch_size, -1, -1).view(batch_size, self.memory_size, self.num_heads, self.value_dim),
                prev_values
            ], dim=1)
        else:
            keys = self.memory_keys.unsqueeze(0).expand(batch_size, -1, -1).view(batch_size, self.memory_size, self.num_heads, self.key_dim)
            values = self.memory_values.unsqueeze(0).expand(batch_size, -1, -1).view(batch_size, self.memory_size, self.num_heads, self.value_dim)
            
        attention_scores = torch.einsum('bhd,bnhd->bhn', query, keys)
        attention_weights = F.softmax(attention_scores / math.sqrt(self.key_dim), dim=-1)
        
        retrieved_values = torch.einsum('bhn,bnhd->bhd', attention_weights, values)
        retrieved_values = retrieved_values.contiguous().view(batch_size, -1)
        
        output = self.output_proj(retrieved_values)
        
        new_key = self.key_proj(x).view(batch_size, 1, self.num_heads, self.key_dim)
        new_value = self.value_proj(x).view(batch_size, 1, self.num_heads, self.value_dim)
        
        if prev_keys is not None:
            updated_keys = torch.cat([prev_keys, new_key], dim=1)
            updated_values = torch.cat([prev_values, new_value], dim=1)
        else:
            updated_keys = new_key
            updated_values = new_value
            
        return output, updated_keys, updated_values

class SparseMemory(nn.Module):
    def __init__(self, input_size, memory_size, memory_dim, sparsity_factor=0.1, top_k=32):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.sparsity_factor = sparsity_factor
        self.top_k = top_k
        
        self.memory = nn.Parameter(torch.zeros(memory_size, memory_dim))
        self.write_proj = nn.Linear(input_size, memory_dim)
        self.read_proj = nn.Linear(input_size, memory_dim)
        self.output_proj = nn.Linear(memory_dim, input_size)
        
        nn.init.xavier_uniform_(self.memory)
        
    def forward(self, x, prev_memory=None):
        batch_size = x.size(0)
        
        if prev_memory is None:
            memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            memory = prev_memory
            
        write_vector = self.write_proj(x)
        read_query = self.read_proj(x)
        
        similarity = F.cosine_similarity(read_query.unsqueeze(1), memory, dim=-1)
        
        top_k = min(self.top_k, self.memory_size)
        _, top_indices = torch.topk(similarity, top_k, dim=-1)
        
        sparse_attention = torch.zeros_like(similarity)
        sparse_attention.scatter_(-1, top_indices, 1.0)
        
        retrieved = torch.bmm(sparse_attention.unsqueeze(1), memory).squeeze(1)
        output = self.output_proj(retrieved)
        
        write_strength = torch.sigmoid(torch.sum(write_vector * retrieved, dim=-1, keepdim=True))
        memory_update = torch.bmm(write_strength.unsqueeze(-1), write_vector.unsqueeze(1))
        
        memory = memory + memory_update
        
        return output, memory

class HierarchicalMemory(nn.Module):
    def __init__(self, input_size, memory_sizes, memory_dims, num_levels=3):
        super().__init__()
        self.input_size = input_size
        self.memory_sizes = memory_sizes
        self.memory_dims = memory_dims
        self.num_levels = num_levels
        
        self.level_projections = nn.ModuleList([
            nn.Linear(input_size, dim) for dim in memory_dims
        ])
        
        self.memories = nn.ParameterList([
            nn.Parameter(torch.zeros(size, dim)) for size, dim in zip(memory_sizes, memory_dims)
        ])
        
        self.attention_layers = nn.ModuleList([
            nn.MultiheadAttention(dim, num_heads=8, batch_first=True) for dim in memory_dims
        ])
        
        self.output_merger = nn.Linear(sum(memory_dims), input_size)
        
        for memory in self.memories:
            nn.init.xavier_uniform_(memory)
            
    def forward(self, x, prev_memories=None):
        batch_size = x.size(0)
        
        if prev_memories is None:
            memories = [
                memory.unsqueeze(0).expand(batch_size, -1, -1) 
                for memory in self.memories
            ]
        else:
            memories = prev_memories
            
        level_outputs = []
        updated_memories = []
        
        for i in range(self.num_levels):
            level_query = self.level_projections[i](x).unsqueeze(1)
            
            attended_output, attention_weights = self.attention_layers[i](
                level_query, memories[i], memories[i]
            )
            
            level_outputs.append(attended_output.squeeze(1))
            
            memory_update = self.level_projections[i](x).unsqueeze(1)
            updated_memory = memories[i] + memory_update
            updated_memories.append(updated_memory)
            
        combined_output = torch.cat(level_outputs, dim=-1)
        output = self.output_merger(combined_output)
        
        return output, updated_memories