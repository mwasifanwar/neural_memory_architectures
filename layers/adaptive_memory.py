# layers/adaptive_memory.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMemory(nn.Module):
    def __init__(self, input_size, memory_size, memory_dim, adaptation_layers=3):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.adaptation_layers = adaptation_layers
        
        self.memory = nn.Parameter(torch.zeros(memory_size, memory_dim))
        
        self.adaptation_network = nn.Sequential(
            nn.Linear(input_size, memory_dim),
            nn.ReLU(),
            *[nn.Sequential(
                nn.Linear(memory_dim, memory_dim),
                nn.ReLU()
            ) for _ in range(adaptation_layers - 2)],
            nn.Linear(memory_dim, memory_dim * 2)
        )
        
        self.output_proj = nn.Linear(memory_dim, input_size)
        
        nn.init.xavier_uniform_(self.memory)
        
    def forward(self, x, prev_memory=None):
        batch_size = x.size(0)
        
        if prev_memory is None:
            memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            memory = prev_memory
            
        adaptation_params = self.adaptation_network(x)
        scale = adaptation_params[:, :self.memory_dim].unsqueeze(1)
        shift = adaptation_params[:, self.memory_dim:].unsqueeze(1)
        
        adapted_memory = memory * torch.sigmoid(scale) + shift
        
        query = x.unsqueeze(1)
        similarity = F.cosine_similarity(query, adapted_memory, dim=-1)
        attention_weights = F.softmax(similarity, dim=-1)
        
        retrieved = torch.bmm(attention_weights.unsqueeze(1), adapted_memory).squeeze(1)
        output = self.output_proj(retrieved)
        
        memory_update = x.unsqueeze(1)
        updated_memory = adapted_memory + memory_update
        
        return output, updated_memory

class GatedMemory(nn.Module):
    def __init__(self, input_size, memory_size, memory_dim):
        super().__init__()
        self.input_size = input_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        self.memory = nn.Parameter(torch.zeros(memory_size, memory_dim))
        
        self.input_gate = nn.Linear(input_size, memory_dim)
        self.forget_gate = nn.Linear(input_size, memory_dim)
        self.output_gate = nn.Linear(input_size, memory_dim)
        self.candidate_gate = nn.Linear(input_size, memory_dim)
        
        self.read_gate = nn.Linear(input_size, memory_dim)
        self.output_proj = nn.Linear(memory_dim, input_size)
        
        nn.init.xavier_uniform_(self.memory)
        
    def forward(self, x, prev_memory=None):
        batch_size = x.size(0)
        
        if prev_memory is None:
            memory = self.memory.unsqueeze(0).expand(batch_size, -1, -1)
        else:
            memory = prev_memory
            
        input_gate = torch.sigmoid(self.input_gate(x).unsqueeze(1))
        forget_gate = torch.sigmoid(self.forget_gate(x).unsqueeze(1))
        output_gate = torch.sigmoid(self.output_gate(x).unsqueeze(1))
        candidate = torch.tanh(self.candidate_gate(x).unsqueeze(1))
        
        updated_memory = forget_gate * memory + input_gate * candidate
        
        read_gate = torch.sigmoid(self.read_gate(x).unsqueeze(1))
        read_weights = F.softmax(read_gate * updated_memory, dim=1)
        
        retrieved = torch.sum(read_weights * updated_memory, dim=1)
        output = output_gate.squeeze(1) * torch.tanh(self.output_proj(retrieved))
        
        return output, updated_memory

class DynamicMemoryLayer(nn.Module):
    def __init__(self, input_size, min_memory_size, max_memory_size, memory_dim, growth_threshold=0.1):
        super().__init__()
        self.input_size = input_size
        self.min_memory_size = min_memory_size
        self.max_memory_size = max_memory_size
        self.memory_dim = memory_dim
        self.growth_threshold = growth_threshold
        
        self.memory = nn.Parameter(torch.zeros(max_memory_size, memory_dim))
        self.memory_usage = nn.Parameter(torch.zeros(max_memory_size))
        
        self.input_proj = nn.Linear(input_size, memory_dim)
        self.output_proj = nn.Linear(memory_dim, input_size)
        self.usage_decay = 0.95
        
        nn.init.xavier_uniform_(self.memory)
        
    def forward(self, x, prev_memory=None, prev_usage=None):
        batch_size = x.size(0)
        current_memory_size = self.min_memory_size
        
        if prev_memory is not None:
            memory = prev_memory
            usage = prev_usage
            current_memory_size = memory.size(1)
        else:
            memory = self.memory[:current_memory_size].unsqueeze(0).expand(batch_size, -1, -1)
            usage = self.memory_usage[:current_memory_size].unsqueeze(0).expand(batch_size, -1)
            
        projected_input = self.input_proj(x).unsqueeze(1)
        
        similarity = F.cosine_similarity(projected_input, memory, dim=-1)
        attention_weights = F.softmax(similarity, dim=-1)
        
        retrieved = torch.bmm(attention_weights.unsqueeze(1), memory).squeeze(1)
        output = self.output_proj(retrieved)
        
        usage = usage * self.usage_decay + attention_weights
        
        if current_memory_size < self.max_memory_size:
            max_usage, _ = torch.max(usage, dim=-1)
            grow_mask = (max_usage > self.growth_threshold).float()
            
            if torch.any(grow_mask):
                new_memory_slots = projected_input * grow_mask.unsqueeze(-1).unsqueeze(-1)
                memory = torch.cat([memory, new_memory_slots], dim=1)
                usage = torch.cat([usage, torch.zeros(batch_size, 1, device=x.device)], dim=1)
                
        memory_update = projected_input * attention_weights.unsqueeze(-1)
        updated_memory = memory + memory_update
        
        return output, updated_memory, usage