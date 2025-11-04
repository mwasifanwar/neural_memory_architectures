# utils/training_utils.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

class MemoryTrainer:
    def __init__(self, model, optimizer, criterion, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)
        
        self.train_losses = []
        self.val_losses = []
        self.memory_metrics = []
        
    def train_epoch(self, dataloader, task_id=None):
        self.model.train()
        total_loss = 0
        memory_states = None
        
        for batch in tqdm(dataloader, desc="Training"):
            if isinstance(batch, (list, tuple)):
                x, y = batch
            else:
                x = batch
                y = batch
                
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            
            if task_id is not None and hasattr(self.model, 'set_task'):
                self.model.set_task(task_id)
                
            if hasattr(self.model, 'forward_with_memory'):
                output, memory_states = self.model.forward_with_memory(x, memory_states)
            else:
                output, memory_states = self.model(x, memory_states)
                
            loss = self.criterion(output, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        self.train_losses.append(avg_loss)
        return avg_loss, memory_states
    
    def validate(self, dataloader, task_id=None):
        self.model.eval()
        total_loss = 0
        memory_states = None
        
        with torch.no_grad():
            for batch in dataloader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                else:
                    x = batch
                    y = batch
                    
                x = x.to(self.device)
                y = y.to(self.device)
                
                if task_id is not None and hasattr(self.model, 'set_task'):
                    self.model.set_task(task_id)
                    
                if hasattr(self.model, 'forward_with_memory'):
                    output, memory_states = self.model.forward_with_memory(x, memory_states)
                else:
                    output, memory_states = self.model(x, memory_states)
                    
                loss = self.criterion(output, y)
                total_loss += loss.item()
                
        avg_loss = total_loss / len(dataloader)
        self.val_losses.append(avg_loss)
        return avg_loss
    
    def train(self, train_loader, val_loader, epochs, task_id=None):
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            train_loss, _ = self.train_epoch(train_loader, task_id)
            val_loss = self.validate(val_loader, task_id)
            
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'best_model.pth')
                
        return self.train_losses, self.val_losses

class ContinualLearningTrainer:
    def __init__(self, model, optimizer, criterion, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)
        
        self.task_performance = {}
        self.memory_consolidation = {}
        
    def train_task(self, task_id, train_loader, val_loader, epochs):
        print(f"Training on task {task_id}")
        
        self.model.current_task = task_id
        task_memory = None
        
        for epoch in range(epochs):
            total_loss = 0
            self.model.train()
            
            for batch in tqdm(train_loader, desc=f"Task {task_id} Epoch {epoch+1}"):
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                else:
                    x = batch
                    y = batch
                    
                x = x.to(self.device)
                y = y.to(self.device)
                
                self.optimizer.zero_grad()
                
                output, task_logits, task_memory = self.model(x, task_id, task_memory)
                
                loss = self.criterion(output, y)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            val_loss = self.validate_task(task_id, val_loader, task_memory)
            
            print(f'Task {task_id}, Epoch {epoch+1}/{epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}')
            
        self.task_performance[task_id] = val_loss
        return task_memory
    
    def validate_task(self, task_id, val_loader, task_memory=None):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x, y = batch
                else:
                    x = batch
                    y = batch
                    
                x = x.to(self.device)
                y = y.to(self.device)
                
                output, _, _ = self.model(x, task_id, task_memory)
                loss = self.criterion(output, y)
                total_loss += loss.item()
                
        return total_loss / len(val_loader)
    
    def evaluate_all_tasks(self, task_loaders):
        performance = {}
        
        for task_id, (_, val_loader) in task_loaders.items():
            loss = self.validate_task(task_id, val_loader)
            performance[task_id] = loss
            
        return performance