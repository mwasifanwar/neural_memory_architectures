# utils/memory_utils.py
import torch
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation
import seaborn as sns

class MemoryVisualizer:
    def __init__(self):
        self.fig = None
        self.ax = None
        
    def plot_memory_usage(self, memory_usage, title="Memory Usage Over Time"):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            
        self.ax.clear()
        
        if isinstance(memory_usage, list):
            memory_usage = torch.stack(memory_usage).cpu().numpy()
        else:
            memory_usage = memory_usage.cpu().numpy()
            
        sns.heatmap(memory_usage.T, ax=self.ax, cmap='viridis', cbar=True)
        self.ax.set_title(title)
        self.ax.set_xlabel('Time Step')
        self.ax.set_ylabel('Memory Slot')
        
        plt.tight_layout()
        return self.fig
    
    def plot_attention_patterns(self, attention_weights, title="Attention Patterns"):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            
        self.ax.clear()
        
        if isinstance(attention_weights, list):
            attention_weights = torch.stack(attention_weights).cpu().numpy()
        else:
            attention_weights = attention_weights.cpu().numpy()
            
        im = self.ax.imshow(attention_weights, cmap='hot', interpolation='nearest', aspect='auto')
        self.ax.set_title(title)
        self.ax.set_xlabel('Memory Slots')
        self.ax.set_ylabel('Time Steps')
        plt.colorbar(im, ax=self.ax)
        
        plt.tight_layout()
        return self.fig
    
    def plot_memory_content(self, memory_content, title="Memory Content"):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            
        self.ax.clear()
        
        if isinstance(memory_content, list):
            memory_content = torch.cat(memory_content, dim=0).cpu().numpy()
        else:
            memory_content = memory_content.cpu().numpy()
            
        if len(memory_content.shape) == 3:
            memory_content = memory_content.reshape(-1, memory_content.shape[-1])
            
        sns.heatmap(memory_content, ax=self.ax, cmap='coolwarm', center=0, cbar=True)
        self.ax.set_title(title)
        self.ax.set_xlabel('Memory Dimensions')
        self.ax.set_ylabel('Memory Slots × Time Steps')
        
        plt.tight_layout()
        return self.fig

class MemoryAnalyzer:
    def __init__(self):
        self.metrics = {}
        
    def compute_memory_efficiency(self, memory_usage, threshold=0.1):
        active_slots = (memory_usage > threshold).float().mean(dim=-1)
        efficiency = active_slots.mean().item()
        return efficiency
    
    def compute_memory_stability(self, memory_content):
        if len(memory_content) < 2:
            return 0.0
            
        changes = []
        for i in range(1, len(memory_content)):
            change = torch.norm(memory_content[i] - memory_content[i-1]).item()
            changes.append(change)
            
        stability = 1.0 / (1.0 + np.mean(changes))
        return stability
    
    def compute_retrieval_accuracy(self, queries, memories, targets):
        similarities = F.cosine_similarity(queries.unsqueeze(1), memories, dim=-1)
        predicted_indices = similarities.argmax(dim=-1)
        accuracy = (predicted_indices == targets).float().mean().item()
        return accuracy
    
    def analyze_memory_dynamics(self, memory_states, operations):
        analysis = {}
        
        if isinstance(memory_states, list):
            memory_states = torch.stack(memory_states)
            
        analysis['memory_variance'] = memory_states.var(dim=[0, 1]).mean().item()
        analysis['memory_entropy'] = self._compute_entropy(memory_states)
        analysis['operation_frequency'] = self._count_operations(operations)
        
        return analysis
    
    def _compute_entropy(self, memory_states):
        probabilities = F.softmax(memory_states.flatten(), dim=0)
        entropy = -torch.sum(probabilities * torch.log(probabilities + 1e-8)).item()
        return entropy
    
    def _count_operations(self, operations):
        if not operations:
            return {}
        
        op_counts = {}
        for op in operations:
            op_type = op.get('type', 'unknown')
            op_counts[op_type] = op_counts.get(op_type, 0) + 1
            
        return op_counts