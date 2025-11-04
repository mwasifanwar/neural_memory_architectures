# main.py
import argparse
from examples.memory_experiments import copy_task, associative_recall, continual_learning
from examples.reasoning_examples import logical_reasoning, temporal_reasoning, knowledge_reasoning

def main():
    parser = argparse.ArgumentParser(description="Neural Memory Architectures")
    parser.add_argument('--experiment', 
                       choices=['copy', 'associative', 'continual', 
                               'logical', 'temporal', 'knowledge', 'all'],
                       default='all',
                       help='Experiment to run')
    
    args = parser.parse_args()
    
    experiments = {
        'copy': copy_task,
        'associative': associative_recall,
        'continual': continual_learning,
        'logical': logical_reasoning,
        'temporal': temporal_reasoning,
        'knowledge': knowledge_reasoning
    }
    
    if args.experiment == 'all':
        for name, experiment in experiments.items():
            print(f"\n{'='*60}")
            print(f"Running {name} experiment...")
            print('='*60)
            experiment()
    else:
        experiments[args.experiment]()

if __name__ == "__main__":
    main()