# models/memory_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..core.memory_networks import NeuralTuringMachine, DifferentiableNeuralComputer
from ..core.attention_memory import AttentionMemory, HierarchicalMemory

class MemoryEnhancedRNN(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, memory_dim, num_layers=2, memory_type='ntm'):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_layers = num_layers
        self.memory_type = memory_type
        
        self.rnn_layers = nn.ModuleList([
            nn.GRU(input_size if i == 0 else hidden_size, hidden_size, batch_first=True)
            for i in range(num_layers)
        ])
        
        if memory_type == 'ntm':
            self.memory_module = NeuralTuringMachine(hidden_size, hidden_size, memory_size, memory_dim)
        elif memory_type == 'dnc':
            self.memory_module = DifferentiableNeuralComputer(hidden_size, hidden_size, memory_size, memory_dim)
        elif memory_type == 'attention':
            self.memory_module = AttentionMemory(hidden_size, memory_size, memory_dim, memory_dim)
        else:
            self.memory_module = HierarchicalMemory(hidden_size, [memory_size], [memory_dim])
            
        self.output_proj = nn.Linear(hidden_size + memory_dim, input_size)
        
    def forward(self, x, hidden_states=None, memory_states=None):
        batch_size, seq_len, _ = x.size()
        
        if hidden_states is None:
            hidden_states = [None] * self.num_layers
        if memory_states is None:
            memory_states = None
            
        rnn_output = x
        new_hidden_states = []
        
        for i, rnn_layer in enumerate(self.rnn_layers):
            rnn_output, hidden_out = rnn_layer(rnn_output, hidden_states[i])
            new_hidden_states.append(hidden_out)
            
        memory_output, new_memory_states = self.memory_module(
            rnn_output[:, -1], memory_states
        )
        
        combined = torch.cat([rnn_output[:, -1], memory_output], dim=-1)
        output = self.output_proj(combined)
        
        return output, new_hidden_states, new_memory_states

class MemoryTransformer(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, memory_size, memory_dim, num_memory_layers=2):
        super().__init__()
        self.input_size = input_size
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        self.input_proj = nn.Linear(input_size, d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.memory_layers = nn.ModuleList([
            HierarchicalMemory(d_model, [memory_size], [memory_dim])
            for _ in range(num_memory_layers)
        ])
        
        self.output_proj = nn.Linear(d_model + num_memory_layers * memory_dim, input_size)
        
    def forward(self, x, memory_states=None):
        batch_size, seq_len, _ = x.size()
        
        if memory_states is None:
            memory_states = [None] * len(self.memory_layers)
            
        projected_input = self.input_proj(x)
        
        transformer_output = self.transformer(projected_input)
        
        memory_outputs = []
        new_memory_states = []
        
        for i, memory_layer in enumerate(self.memory_layers):
            memory_out, new_memory = memory_layer(
                transformer_output[:, -1], memory_states[i]
            )
            memory_outputs.append(memory_out)
            new_memory_states.append(new_memory)
            
        combined = torch.cat([transformer_output[:, -1]] + memory_outputs, dim=-1)
        output = self.output_proj(combined)
        
        return output, new_memory_states

class ContinualLearningModel(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, memory_dim, num_tasks=10):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_tasks = num_tasks
        
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
        self.memory_banks = nn.ModuleList([
            AttentionMemory(hidden_size, memory_size, memory_dim, memory_dim)
            for _ in range(num_tasks)
        ])
        
        self.task_classifier = nn.Linear(hidden_size, num_tasks)
        self.output_proj = nn.Linear(hidden_size + memory_dim, input_size)
        
        self.current_task = 0
        
    def forward(self, x, task_id=None, memory_states=None):
        if task_id is None:
            task_id = self.current_task
            
        if memory_states is None:
            memory_states = [None] * self.num_tasks
            
        encoded = self.encoder(x)
        
        task_logits = self.task_classifier(encoded)
        
        memory_output, new_memory_states = self.memory_banks[task_id](
            encoded, memory_states[task_id]
        )
        
        combined = torch.cat([encoded, memory_output], dim=-1)
        output = self.output_proj(combined)
        
        full_memory_states = memory_states.copy()
        full_memory_states[task_id] = new_memory_states
        
        return output, task_logits, full_memory_states
        
    def set_task(self, task_id):
        self.current_task = task_id