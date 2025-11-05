

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from datetime import datetime






class ExperimentRunner:
    """
    Automated experiment runner for systematic evaluation
    """
    def __init__(self, base_config, save_dir='experiments'):
        self.base_config = base_config
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
        self.results = {}
        
    def run_experiment_1_num_points(self):
        """
        Experiment 1: Effect of number of point annotations
        """
        print("=" * 60)
        print("EXPERIMENT 1: Effect of Number of Point Annotations")
        print("=" * 60)
        
        num_points_list = [50, 100, 200, 500, 1000, 2000]
        results = {
            'num_points': [],
            'miou': [],
            'per_class_iou': [],
            'train_time': [],
            'annotation_percentage': []
        }
        
        for num_points in num_points_list:
            print(f"\n--- Training with {num_points} points ---")
            
            
            img_size = self.base_config['image_size']
            total_pixels = img_size[0] * img_size[1]
            annotation_pct = (num_points / total_pixels) * 100
            
            
            sampler = PointLabelSampler(
                num_points=num_points,
                strategy='uniform'  
            )
            
            
            start_time = datetime.now()
            model, metrics = self.train_model(sampler=sampler)
            train_time = (datetime.now() - start_time).total_seconds() / 3600  
            
            
            miou, per_class_iou = self.evaluate_model(model)
            
            
            results['num_points'].append(num_points)
            results['miou'].append(miou)
            results['per_class_iou'].append(per_class_iou)
            results['train_time'].append(train_time)
            results['annotation_percentage'].append(annotation_pct)
            
            print(f"Results: mIoU = {miou:.4f}, Time = {train_time:.2f}h")
        
        
        exp1_dir = self.save_dir / 'exp1_num_points'
        exp1_dir.mkdir(exist_ok=True)
        
        with open(exp1_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        
        self.plot_experiment_1(results, exp1_dir)
        
        self.results['experiment_1'] = results
        return results
    
    def run_experiment_2_sampling_strategy(self):
        """
        Experiment 2: Effect of sampling strategy
        """
        print("=" * 60)
        print("EXPERIMENT 2: Effect of Sampling Strategy")
        print("=" * 60)
        
        strategies = ['random', 'uniform', 'balanced', 'boundary', 'cluster']
        num_points = 200  
        
        results = {
            'strategy': [],
            'miou': [],
            'per_class_iou': [],
            'train_time': []
        }
        
        for strategy in strategies:
            print(f"\n--- Training with {strategy} sampling ---")
            
            
            sampler = PointLabelSampler(
                num_points=num_points,
                strategy=strategy
            )
            
            
            start_time = datetime.now()
            model, metrics = self.train_model(sampler=sampler)
            train_time = (datetime.now() - start_time).total_seconds() / 3600
            
            
            miou, per_class_iou = self.evaluate_model(model)
            
            
            results['strategy'].append(strategy)
            results['miou'].append(miou)
            results['per_class_iou'].append(per_class_iou)
            results['train_time'].append(train_time)
            
            print(f"Results: mIoU = {miou:.4f}, Time = {train_time:.2f}h")
        
        
        exp2_dir = self.save_dir / 'exp2_sampling_strategy'
        exp2_dir.mkdir(exist_ok=True)
        
        with open(exp2_dir / 'results.json', 'w') as f:
            json.dump(results, f, indent=4)
        
        
        self.plot_experiment_2(results, exp2_dir)
        
        self.results['experiment_2'] = results
        return results
    
    def train_model(self, sampler, num_epochs=None):
        """
        Train model with given configuration
        Implement this based on your training pipeline
        """
        if num_epochs is None:
            num_epochs = self.base_config['num_epochs']
        
        
        
        
        
        
        
        
        
        
        print(f"Training for {num_epochs} epochs...")
        
        
        model = None  
        metrics = {'loss': [], 'miou': []}
        
        return model, metrics
    
    def evaluate_model(self, model):
        """
        Evaluate model on validation set
        Returns: (mean_iou, per_class_iou)
        """
        
        
        
        
        miou = 0.0  
        per_class_iou = []  
        
        return miou, per_class_iou
    
    def plot_experiment_1(self, results, save_dir):
        """
        Visualize Experiment 1 results
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        
        axes[0].plot(results['num_points'], results['miou'], 
                    marker='o', linewidth=2, markersize=8)
        axes[0].set_xlabel('Number of Point Annotations', fontsize=12)
        axes[0].set_ylabel('Mean IoU (%)', fontsize=12)
        axes[0].set_title('Performance vs Annotation Budget', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xscale('log')
        
        
        axes[1].plot(results['annotation_percentage'], results['miou'],
                    marker='s', linewidth=2, markersize=8, color='orange')
        axes[1].set_xlabel('Annotation Percentage (%)', fontsize=12)
        axes[1].set_ylabel('Mean IoU (%)', fontsize=12)
        axes[1].set_title('Annotation Efficiency', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        
        axes[2].bar(range(len(results['num_points'])), results['train_time'],
                   color='green', alpha=0.7)
        axes[2].set_xlabel('Configuration', fontsize=12)
        axes[2].set_ylabel('Training Time (hours)', fontsize=12)
        axes[2].set_title('Training Efficiency', fontsize=14, fontweight='bold')
        axes[2].set_xticks(range(len(results['num_points'])))
        axes[2].set_xticklabels([str(n) for n in results['num_points']], rotation=45)
        axes[2].grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(save_dir / 'experiment_1_results.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_dir / 'experiment_1_results.png'}")
        
        
        df = pd.DataFrame({
            'Num Points': results['num_points'],
            'mIoU (%)': [f"{x:.2f}" for x in results['miou']],
            'Annotation (%)': [f"{x:.4f}" for x in results['annotation_percentage']],
            'Train Time (h)': [f"{x:.2f}" for x in results['train_time']]
        })
        
        df.to_csv(save_dir / 'results_table.csv', index=False)
        print(f"✓ Saved results table to {save_dir / 'results_table.csv'}")
        
        plt.close()
    
    def plot_experiment_2(self, results, save_dir):
        """
        Visualize Experiment 2 results
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        
        strategies = results['strategy']
        mious = results['miou']
        
        colors = plt.cm.Set3(range(len(strategies)))
        bars = axes[0].bar(strategies, mious, color=colors, alpha=0.8, edgecolor='black')
        axes[0].set_xlabel('Sampling Strategy', fontsize=12)
        axes[0].set_ylabel('Mean IoU (%)', fontsize=12)
        axes[0].set_title('Strategy Performance Comparison', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].tick_params(axis='x', rotation=45)
        
        
        for bar in bars:
            height = bar.get_height()
            axes[0].text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}',
                        ha='center', va='bottom', fontsize=10)
        
        
        if results['per_class_iou'] and len(results['per_class_iou'][0]) > 0:
            per_class_data = np.array(results['per_class_iou'])
            
            im = axes[1].imshow(per_class_data.T, cmap='RdYlGn', aspect='auto', vmin=0, vmax=100)
            axes[1].set_xlabel('Sampling Strategy', fontsize=12)
            axes[1].set_ylabel('Class ID', fontsize=12)
            axes[1].set_title('Per-Class IoU (%)', fontsize=14, fontweight='bold')
            axes[1].set_xticks(range(len(strategies)))
            axes[1].set_xticklabels(strategies, rotation=45)
            axes[1].set_yticks(range(per_class_data.shape[1]))
            
            
            plt.colorbar(im, ax=axes[1], label='IoU (%)')
            
            
            for i in range(len(strategies)):
                for j in range(per_class_data.shape[1]):
                    text = axes[1].text(i, j, f'{per_class_data[i, j]:.1f}',
                                       ha="center", va="center", color="black", fontsize=8)
        
        plt.tight_layout()
        plt.savefig(save_dir / 'experiment_2_results.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved visualization to {save_dir / 'experiment_2_results.png'}")
        
        
        df = pd.DataFrame({
            'Strategy': results['strategy'],
            'mIoU (%)': [f"{x:.2f}" for x in results['miou']],
            'Train Time (h)': [f"{x:.2f}" for x in results['train_time']]
        })
        
        df.to_csv(save_dir / 'results_table.csv', index=False)
        print(f"✓ Saved results table to {save_dir / 'results_table.csv'}")
        
        plt.close()
    
    def generate_report(self):
        """
        Generate comprehensive experiment report
        """
        report_path = self.save_dir / 'experiment_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("POINT-SUPERVISED SEGMENTATION - EXPERIMENT REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            
            if 'experiment_1' in self.results:
                f.write("\n" + "-" * 80 + "\n")
                f.write("EXPERIMENT 1: Effect of Number of Point Annotations\n")
                f.write("-" * 80 + "\n\n")
                
                exp1 = self.results['experiment_1']
                for i, num_points in enumerate(exp1['num_points']):
                    f.write(f"Configuration {i+1}:\n")
                    f.write(f"  - Number of Points: {num_points}\n")
                    f.write(f"  - Annotation %: {exp1['annotation_percentage'][i]:.4f}%\n")
                    f.write(f"  - Mean IoU: {exp1['miou'][i]:.4f}%\n")
                    f.write(f"  - Training Time: {exp1['train_time'][i]:.2f}h\n\n")
                
                
                best_idx = np.argmax(exp1['miou'])
                f.write(f"\nBest Configuration: {exp1['num_points'][best_idx]} points ")
                f.write(f"(mIoU: {exp1['miou'][best_idx]:.4f}%)\n")
            
            
            if 'experiment_2' in self.results:
                f.write("\n" + "-" * 80 + "\n")
                f.write("EXPERIMENT 2: Effect of Sampling Strategy\n")
                f.write("-" * 80 + "\n\n")
                
                exp2 = self.results['experiment_2']
                for i, strategy in enumerate(exp2['strategy']):
                    f.write(f"Strategy: {strategy}\n")
                    f.write(f"  - Mean IoU: {exp2['miou'][i]:.4f}%\n")
                    f.write(f"  - Training Time: {exp2['train_time'][i]:.2f}h\n\n")
                
                
                best_idx = np.argmax(exp2['miou'])
                f.write(f"\nBest Strategy: {exp2['strategy'][best_idx]} ")
                f.write(f"(mIoU: {exp2['miou'][best_idx]:.4f}%)\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n✓ Generated comprehensive report: {report_path}")



if __name__ == "__main__":
    
    base_config = {
        'image_size': (256, 256),
        'num_classes': 5,
        'num_epochs': 50,
        'batch_size': 8,
        'learning_rate': 0.001,
    }
    
    
    runner = ExperimentRunner(base_config, save_dir='experiments')
    
    
    print("Starting automated experiments...\n")
    
    
    runner.run_experiment_1_num_points()
    
    
    runner.run_experiment_2_sampling_strategy()
    
    
    runner.generate_report()
    
    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 80)
    print(f"\nResults saved in: {runner.save_dir}")
    print("Check the following files:")
    print("  - experiments/exp1_num_points/experiment_1_results.png")
    print("  - experiments/exp2_sampling_strategy/experiment_2_results.png")
    print("  - experiments/experiment_report.txt")
