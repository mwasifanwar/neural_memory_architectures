# examples/reasoning_examples.py
import torch
import torch.nn as nn
import numpy as np
from ..models.reasoning_models import ReasoningNetwork, KnowledgeGraphModel, TemporalMemoryModel
from ..utils.training_utils import MemoryTrainer

def logical_reasoning(num_rules=10, rule_length=5):
    print("Running Logical Reasoning Experiment")
    
    def generate_logical_data(batch_size, num_rules, rule_len):
        premises = torch.randint(0, 2, (batch_size, rule_len, num_rules)).float()
        rules = torch.randn(batch_size, num_rules, num_rules)
        
        conclusions = torch.bmm(premises.view(batch_size, rule_len, num_rules), rules)
        conclusions = torch.sigmoid(conclusions)
        
        return premises, rules, conclusions
    
    input_size = num_rules
    hidden_size = 128
    memory_size = 256
    memory_dim = 64
    
    model = ReasoningNetwork(input_size, hidden_size, memory_size, memory_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    trainer = MemoryTrainer(model, optimizer, criterion)
    
    train_premises, train_rules, train_conclusions = generate_logical_data(500, num_rules, rule_length)
    val_premises, val_rules, val_conclusions = generate_logical_data(100, num_rules, rule_length)
    
    class LogicalDataset(torch.utils.data.Dataset):
        def __init__(self, premises, rules, conclusions):
            self.premises = premises
            self.rules = rules
            self.conclusions = conclusions
            
        def __len__(self):
            return len(self.premises)
            
        def __getitem__(self, idx):
            return self.premises[idx], self.rules[idx], self.conclusions[idx]
    
    train_dataset = LogicalDataset(train_premises, train_rules, train_conclusions)
    val_dataset = LogicalDataset(val_premises, val_rules, val_conclusions)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
    
    def custom_forward(model, x, memory_states):
        premises, rules, _ = x
        output, memory_states = model(premises[:, -1], memory_states)
        return output, memory_states
    
    def custom_criterion(output, targets):
        premises, rules, conclusions = targets
        return nn.MSELoss()(output, conclusions[:, -1])
    
    trainer.model.forward = lambda x, ms: custom_forward(trainer.model, x, ms)
    trainer.criterion = custom_criterion
    
    losses = trainer.train(train_loader, val_loader, epochs=100)
    
    print("Logical reasoning experiment completed")
    return model, losses

def temporal_reasoning(sequence_length=20, feature_size=16):
    print("Running Temporal Reasoning Experiment")
    
    def generate_temporal_data(batch_size, seq_len, feat_size):
        sequences = []
        for i in range(3):
            scale = 0.1 * (i + 1)
            seq = torch.randn(batch_size, seq_len // 3, feat_size) * scale
            sequences.append(seq)
            
        full_sequence = torch.cat(sequences, dim=1)
        
        current_input = full_sequence[:, -1]
        targets = full_sequence[:, -1] + 0.1 * full_sequence[:, 0]
        
        return current_input, sequences, targets
    
    input_size = feature_size
    hidden_size = 64
    memory_size = 128
    memory_dim = 32
    
    model = TemporalMemoryModel(input_size, hidden_size, memory_size, memory_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    trainer = MemoryTrainer(model, optimizer, criterion)
    
    train_inputs, train_sequences, train_targets = generate_temporal_data(500, sequence_length, feature_size)
    val_inputs, val_sequences, val_targets = generate_temporal_data(100, sequence_length, feature_size)
    
    class TemporalDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, sequences, targets):
            self.inputs = inputs
            self.sequences = sequences
            self.targets = targets
            
        def __len__(self):
            return len(self.inputs)
            
        def __getitem__(self, idx):
            return self.inputs[idx], self.sequences[idx], self.targets[idx]
    
    train_dataset = TemporalDataset(train_inputs, train_sequences, train_targets)
    val_dataset = TemporalDataset(val_inputs, val_sequences, val_targets)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
    
    def custom_forward(model, x, memory_states):
        current_input, sequences, _ = x
        output, memory_states = model(current_input, sequences, memory_states)
        return output, memory_states
    
    def custom_criterion(output, targets):
        current_input, sequences, true_targets = targets
        return nn.MSELoss()(output, true_targets)
    
    trainer.model.forward = lambda x, ms: custom_forward(trainer.model, x, ms)
    trainer.criterion = custom_criterion
    
    losses = trainer.train(train_loader, val_loader, epochs=100)
    
    print("Temporal reasoning experiment completed")
    return model, losses

def knowledge_reasoning(num_entities=50, num_relations=10, feature_size=32):
    print("Running Knowledge Reasoning Experiment")
    
    def generate_knowledge_data(batch_size, num_ent, num_rel, feat_size):
        entity_ids = torch.randint(0, num_ent, (batch_size,))
        relation_ids = torch.randint(0, num_rel, (batch_size,))
        
        inputs = torch.randn(batch_size, feat_size)
        targets = torch.randn(batch_size, feat_size)
        
        return inputs, entity_ids, relation_ids, targets
    
    input_size = feature_size
    entity_size = num_entities
    relation_size = num_relations
    memory_size = 256
    memory_dim = 64
    
    model = KnowledgeGraphModel(input_size, entity_size, relation_size, memory_size, memory_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    trainer = MemoryTrainer(model, optimizer, criterion)
    
    train_inputs, train_entities, train_relations, train_targets = generate_knowledge_data(500, num_entities, num_relations, feature_size)
    val_inputs, val_entities, val_relations, val_targets = generate_knowledge_data(100, num_entities, num_relations, feature_size)
    
    class KnowledgeDataset(torch.utils.data.Dataset):
        def __init__(self, inputs, entities, relations, targets):
            self.inputs = inputs
            self.entities = entities
            self.relations = relations
            self.targets = targets
            
        def __len__(self):
            return len(self.inputs)
            
        def __getitem__(self, idx):
            return self.inputs[idx], self.entities[idx], self.relations[idx], self.targets[idx]
    
    train_dataset = KnowledgeDataset(train_inputs, train_entities, train_relations, train_targets)
    val_dataset = KnowledgeDataset(val_inputs, val_entities, val_relations, val_targets)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
    
    def custom_forward(model, x, memory_states):
        inputs, entities, relations, _ = x
        output, memory_states = model(inputs, entities, relations, memory_states)
        return output, memory_states
    
    def custom_criterion(output, targets):
        inputs, entities, relations, true_targets = targets
        return nn.MSELoss()(output, true_targets)
    
    trainer.model.forward = lambda x, ms: custom_forward(trainer.model, x, ms)
    trainer.criterion = custom_criterion
    
    losses = trainer.train(train_loader, val_loader, epochs=100)
    
    print("Knowledge reasoning experiment completed")
    return model, losses