# models/reasoning_models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import networkx as nx
import numpy as np
from ..core.memory_cells import AssociativeMemory, DynamicMemory
from ..core.attention_memory import SparseMemory, HierarchicalMemory

class ReasoningNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, memory_dim, num_steps=5):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_steps = num_steps
        
        self.step_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_size + memory_dim, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, memory_dim)
            ) for _ in range(num_steps)
        ])
        
        self.memory_module = AssociativeMemory(hidden_size, memory_size, memory_dim, memory_dim)
        self.reasoning_proj = nn.Linear(memory_dim * num_steps, input_size)
        
    def forward(self, x, memory_states=None):
        batch_size = x.size(0)
        
        if memory_states is None:
            memory_keys = None
            memory_values = None
        else:
            memory_keys, memory_values = memory_states
            
        reasoning_steps = []
        current_state = x
        
        for step_net in self.step_networks:
            if memory_keys is not None:
                memory_query = torch.cat([current_state, 
                                        torch.mean(memory_keys, dim=1)], dim=-1)
            else:
                memory_query = torch.cat([current_state, 
                                        torch.zeros(batch_size, self.memory_dim, device=x.device)], dim=-1)
                
            reasoning_step = step_net(memory_query)
            reasoning_steps.append(reasoning_step)
            
            memory_output, memory_keys, memory_values = self.memory_module(
                reasoning_step, memory_keys, memory_values
            )
            
            current_state = current_state + memory_output
            
        reasoning_trajectory = torch.cat(reasoning_steps, dim=-1)
        output = self.reasoning_proj(reasoning_trajectory)
        
        return output, (memory_keys, memory_values)

class KnowledgeGraphModel(nn.Module):
    def __init__(self, input_size, entity_size, relation_size, memory_size, memory_dim):
        super().__init__()
        self.input_size = input_size
        self.entity_size = entity_size
        self.relation_size = relation_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        
        self.entity_embeddings = nn.Embedding(entity_size, memory_dim)
        self.relation_embeddings = nn.Embedding(relation_size, memory_dim)
        
        self.memory_module = DynamicMemory(input_size, memory_size, memory_dim)
        self.kg_attention = nn.MultiheadAttention(memory_dim, num_heads=8, batch_first=True)
        
        self.output_proj = nn.Linear(memory_dim * 2, input_size)
        
        nn.init.xavier_uniform_(self.entity_embeddings.weight)
        nn.init.xavier_uniform_(self.relation_embeddings.weight)
        
    def forward(self, x, entity_ids, relation_ids, memory_state=None):
        batch_size = x.size(0)
        
        entity_emb = self.entity_embeddings(entity_ids)
        relation_emb = self.relation_embeddings(relation_ids)
        
        kg_representation = entity_emb + relation_emb
        
        memory_output, new_memory_state = self.memory_module(x, memory_state)
        
        attended_kg, attention_weights = self.kg_attention(
            memory_output.unsqueeze(1), kg_representation, kg_representation
        )
        
        combined = torch.cat([memory_output, attended_kg.squeeze(1)], dim=-1)
        output = self.output_proj(combined)
        
        return output, new_memory_state

class TemporalMemoryModel(nn.Module):
    def __init__(self, input_size, hidden_size, memory_size, memory_dim, num_temporal_scales=3):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.memory_dim = memory_dim
        self.num_temporal_scales = num_temporal_scales
        
        self.temporal_encoders = nn.ModuleList([
            nn.GRU(input_size, hidden_size, batch_first=True)
            for _ in range(num_temporal_scales)
        ])
        
        self.temporal_memories = nn.ModuleList([
            HierarchicalMemory(hidden_size, [memory_size], [memory_dim])
            for _ in range(num_temporal_scales)
        ])
        
        self.scale_weights = nn.Parameter(torch.ones(num_temporal_scales))
        self.output_proj = nn.Linear(hidden_size * num_temporal_scales, input_size)
        
    def forward(self, x, sequences, memory_states=None):
        batch_size = x.size(0)
        
        if memory_states is None:
            memory_states = [None] * self.num_temporal_scales
            
        temporal_outputs = []
        new_memory_states = []
        
        for i, (encoder, memory) in enumerate(zip(self.temporal_encoders, self.temporal_memories)):
            seq_length = sequences[i].size(1) if i < len(sequences) else 1
            
            if seq_length > 1:
                encoded_seq, _ = encoder(sequences[i])
                temporal_feat = encoded_seq[:, -1]
            else:
                temporal_feat = x
                
            memory_output, new_memory = memory(temporal_feat, memory_states[i])
            
            weighted_output = self.scale_weights[i] * memory_output
            temporal_outputs.append(weighted_output)
            new_memory_states.append(new_memory)
            
        combined = torch.cat(temporal_outputs, dim=-1)
        output = self.output_proj(combined)
        
        return output, new_memory_states