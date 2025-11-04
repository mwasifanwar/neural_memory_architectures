# examples/memory_experiments.py
import torch
import torch.nn as nn
import numpy as np
from ..core.memory_networks import NeuralTuringMachine, DifferentiableNeuralComputer
from ..models.memory_models import MemoryEnhancedRNN, ContinualLearningModel
from ..utils.training_utils import MemoryTrainer, ContinualLearningTrainer

def copy_task(sequence_length=20, vector_size=8, num_sequences=1000):
    print("Running Copy Task Experiment")
    
    def generate_copy_data(batch_size, seq_len, vec_size):
        sequences = torch.randint(0, 2, (batch_size, seq_len, vec_size)).float()
        inputs = torch.cat([sequences, torch.zeros(batch_size, seq_len, vec_size)], dim=1)
        targets = torch.cat([torch.zeros(batch_size, seq_len, vec_size), sequences], dim=1)
        return inputs, targets
    
    input_size = vector_size
    hidden_size = 64
    memory_size = 128
    memory_dim = 32
    
    model = NeuralTuringMachine(input_size, hidden_size, memory_size, memory_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    trainer = MemoryTrainer(model, optimizer, criterion)
    
    train_inputs, train_targets = generate_copy_data(100, sequence_length, vector_size)
    val_inputs, val_targets = generate_copy_data(20, sequence_length, vector_size)
    
    train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
    val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16)
    
    losses = trainer.train(train_loader, val_loader, epochs=50)
    
    print("Copy task training completed")
    return model, losses

def associative_recall(num_patterns=100, pattern_size=16, query_size=8):
    print("Running Associative Recall Experiment")
    
    def generate_associative_data(batch_size, num_pat, pat_size, query_size):
        patterns = torch.randn(batch_size, num_pat, pat_size)
        queries = torch.randn(batch_size, query_size)
        
        indices = torch.randint(0, num_pat, (batch_size,))
        targets = patterns[torch.arange(batch_size), indices]
        
        return queries, patterns, targets
    
    input_size = query_size
    hidden_size = 64
    memory_size = num_patterns
    memory_dim = pattern_size
    
    model = DifferentiableNeuralComputer(input_size, hidden_size, memory_size, memory_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    trainer = MemoryTrainer(model, optimizer, criterion)
    
    train_queries, train_patterns, train_targets = generate_associative_data(500, num_patterns, pattern_size, query_size)
    val_queries, val_patterns, val_targets = generate_associative_data(100, num_patterns, pattern_size, query_size)
    
    class AssociativeDataset(torch.utils.data.Dataset):
        def __init__(self, queries, patterns, targets):
            self.queries = queries
            self.patterns = patterns
            self.targets = targets
            
        def __len__(self):
            return len(self.queries)
            
        def __getitem__(self, idx):
            return self.queries[idx], self.patterns[idx], self.targets[idx]
    
    train_dataset = AssociativeDataset(train_queries, train_patterns, train_targets)
    val_dataset = AssociativeDataset(val_queries, val_patterns, val_targets)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
    
    def custom_train_epoch(trainer, dataloader):
        trainer.model.train()
        total_loss = 0
        memory_states = None
        
        for queries, patterns, targets in tqdm(dataloader, desc="Training"):
            queries = queries.to(trainer.device)
            patterns = patterns.to(trainer.device)
            targets = targets.to(trainer.device)
            
            trainer.optimizer.zero_grad()
            
            output, memory_states = trainer.model(queries, memory_states)
            loss = trainer.criterion(output, targets)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), 1.0)
            trainer.optimizer.step()
            
            total_loss += loss.item()
            
        return total_loss / len(dataloader), memory_states
    
    trainer.train_epoch = lambda dl: custom_train_epoch(trainer, dl)
    losses = trainer.train(train_loader, val_loader, epochs=100)
    
    print("Associative recall training completed")
    return model, losses

def continual_learning(num_tasks=5, task_complexity=10):
    print("Running Continual Learning Experiment")
    
    def generate_task_data(num_samples, task_id, num_tasks):
        task_specific = torch.randn(num_samples, task_complexity) * (task_id + 1)
        shared_features = torch.randn(num_samples, task_complexity)
        
        inputs = torch.cat([task_specific, shared_features], dim=1)
        targets = task_specific + shared_features
        
        return inputs, targets
    
    input_size = task_complexity * 2
    hidden_size = 128
    memory_size = 256
    memory_dim = 64
    
    model = ContinualLearningModel(input_size, hidden_size, memory_size, memory_dim, num_tasks)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    trainer = ContinualLearningTrainer(model, optimizer, criterion)
    
    task_loaders = {}
    task_memories = [None] * num_tasks
    
    for task_id in range(num_tasks):
        train_inputs, train_targets = generate_task_data(1000, task_id, num_tasks)
        val_inputs, val_targets = generate_task_data(200, task_id, num_tasks)
        
        train_dataset = torch.utils.data.TensorDataset(train_inputs, train_targets)
        val_dataset = torch.utils.data.TensorDataset(val_inputs, val_targets)
        
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=32)
        
        task_loaders[task_id] = (train_loader, val_loader)
        
        task_memory = trainer.train_task(task_id, train_loader, val_loader, epochs=50)
        task_memories[task_id] = task_memory
        
    final_performance = trainer.evaluate_all_tasks(task_loaders)
    
    print("Continual learning experiment completed")
    print("Task performances:", final_performance)
    
    return model, final_performance, task_memories