

from loveda_training import LoveDATrainer
import json
from pathlib import Path
import numpy as np

def quick_demo():
    """
    Quick demo version with reduced configurations
    Completes in 2-3 hours instead of 12-20 hours
    """
    
    print("\n" + "="*80)
    print("QUICK DEMO MODE - REDUCED SCOPE FOR FASTER COMPLETION")
    print("="*80 + "\n")
    
    base_config = {
        'data_root': '/Users/pepo_abdo/Desktop/Remote Sensing Segmentation/data',
        'scene': 'urban',  
        'image_size': 256,  
        'num_classes': 7,
        'encoder': 'resnet34',
        'batch_size': 8,
        'num_epochs': 20,  
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'num_workers': 0,
        'save_freq': 10,
        'ignore_background': True,
        'use_class_weights': False,
    }
    
    all_results = {
        'experiment_1': {'num_points': [], 'miou': [], 'per_class_iou': [], 'annotation_percentage': []},
        'experiment_2': {'strategy': [], 'miou': [], 'per_class_iou': []}
    }
    
    
    
    
    print("\n" + "="*80)
    print("EXPERIMENT 1: EFFECT OF NUMBER OF POINTS (3 configurations)")
    print("="*80 + "\n")
    
    num_points_list = [200, 500]  
    
    for num_points in num_points_list:
        print(f"\n{'='*60}")
        print(f"Training with {num_points} points")
        print(f"{'='*60}\n")
        
        config = base_config.copy()
        config['num_points'] = num_points
        config['sampling_strategy'] = 'uniform'
        config['checkpoint_dir'] = f'checkpoints/demo_exp1_points_{num_points}'
        
        trainer = LoveDATrainer(config)
        history = trainer.train()
        
        
        img_pixels = config['image_size'] ** 2
        annotation_pct = (num_points / img_pixels) * 100
        
        
        all_results['experiment_1']['num_points'].append(num_points)
        all_results['experiment_1']['miou'].append(trainer.best_miou)
        all_results['experiment_1']['per_class_iou'].append(history['val_per_class_iou'][-1])
        all_results['experiment_1']['annotation_percentage'].append(annotation_pct)
        
        print(f"\n✓ Completed: {num_points} points → mIoU: {trainer.best_miou:.2f}%")
    
    
    exp1_dir = Path('experiments/demo_exp1_num_points')
    exp1_dir.mkdir(exist_ok=True, parents=True)
    with open(exp1_dir / 'results.json', 'w') as f:
        json.dump(all_results['experiment_1'], f, indent=4)
    
    
    
    
    print("\n" + "="*80)
    print("EXPERIMENT 2: EFFECT OF SAMPLING STRATEGY (3 strategies)")
    print("="*80 + "\n")
    
    strategies = ['random', 'uniform', 'balanced']  
    num_points = 200
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Training with {strategy} sampling")
        print(f"{'='*60}\n")
        
        config = base_config.copy()
        config['num_points'] = num_points
        config['sampling_strategy'] = strategy
        config['checkpoint_dir'] = f'checkpoints/demo_exp2_strategy_{strategy}'
        
        trainer = LoveDATrainer(config)
        history = trainer.train()
        
        
        all_results['experiment_2']['strategy'].append(strategy)
        all_results['experiment_2']['miou'].append(trainer.best_miou)
        all_results['experiment_2']['per_class_iou'].append(history['val_per_class_iou'][-1])
        
        print(f"\n✓ Completed: {strategy} → mIoU: {trainer.best_miou:.2f}%")
    
    
    exp2_dir = Path('experiments/demo_exp2_sampling_strategy')
    exp2_dir.mkdir(exist_ok=True, parents=True)
    with open(exp2_dir / 'results.json', 'w') as f:
        json.dump(all_results['experiment_2'], f, indent=4)
    
    
    
    
    print("\n" + "="*80)
    print("GENERATING RESULTS SUMMARY")
    print("="*80 + "\n")
    
    
    import matplotlib.pyplot as plt
    
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    exp1 = all_results['experiment_1']
    ax1.plot(exp1['num_points'], exp1['miou'], marker='o', linewidth=2.5, markersize=10)
    ax1.set_xlabel('Number of Points', fontsize=12, fontweight='bold')
    ax1.set_ylabel('mIoU (%)', fontsize=12, fontweight='bold')
    ax1.set_title('Exp 1: Points vs Performance', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    for x, y in zip(exp1['num_points'], exp1['miou']):
        ax1.annotate(f'{y:.1f}%', (x, y), textcoords="offset points", 
                    xytext=(0,10), ha='center', fontsize=10)
    
    
    exp2 = all_results['experiment_2']
    bars = ax2.bar(exp2['strategy'], exp2['miou'], color=['#ff9999','#66b3ff','#99ff99'], 
                   alpha=0.8, edgecolor='black')
    ax2.set_xlabel('Sampling Strategy', fontsize=12, fontweight='bold')
    ax2.set_ylabel('mIoU (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Exp 2: Strategy Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('experiments/demo_results_summary.png', dpi=300, bbox_inches='tight')
    print(f"✓ Saved visualization: experiments/demo_results_summary.png")
    
    
    summary_path = Path('experiments/demo_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("QUICK DEMO RESULTS SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXPERIMENT 1: Number of Point Annotations\n")
        f.write("-"*80 + "\n")
        for i in range(len(exp1['num_points'])):
            f.write(f"  {exp1['num_points'][i]:4d} points → mIoU: {exp1['miou'][i]:5.2f}% ")
            f.write(f"(annotation: {exp1['annotation_percentage'][i]:.3f}%)\n")
        
        best_idx = np.argmax(exp1['miou'])
        f.write(f"\n  Best: {exp1['num_points'][best_idx]} points with {exp1['miou'][best_idx]:.2f}% mIoU\n")
        
        f.write("\n" + "="*80 + "\n\n")
        f.write("EXPERIMENT 2: Sampling Strategies\n")
        f.write("-"*80 + "\n")
        for i in range(len(exp2['strategy'])):
            f.write(f"  {exp2['strategy'][i]:12s} → mIoU: {exp2['miou'][i]:5.2f}%\n")
        
        best_idx = np.argmax(exp2['miou'])
        f.write(f"\n  Best: {exp2['strategy'][best_idx]} with {exp2['miou'][best_idx]:.2f}% mIoU\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("KEY FINDINGS\n")
        f.write("="*80 + "\n\n")
        
        f.write("1. Annotation Efficiency:\n")
        best_perf_idx = np.argmax(exp1['miou'])
        annotation_reduction = 100 - exp1['annotation_percentage'][best_perf_idx]
        f.write(f"   - Achieved {exp1['miou'][best_perf_idx]:.2f}% mIoU with only ")
        f.write(f"{exp1['annotation_percentage'][best_perf_idx]:.3f}% annotation cost\n")
        f.write(f"   - Annotation reduction: ~{annotation_reduction:.1f}%\n\n")
        
        f.write("2. Sampling Strategy Impact:\n")
        miou_range = max(exp2['miou']) - min(exp2['miou'])
        f.write(f"   - Performance range: {miou_range:.2f}% across strategies\n")
        f.write(f"   - Best strategy: {exp2['strategy'][np.argmax(exp2['miou'])]}\n\n")
        
        f.write("3. Practical Implications:\n")
        f.write("   - Point supervision is viable for remote sensing tasks\n")
        f.write("   - Significant cost savings with minimal performance loss\n")
        f.write("   - Strategy selection matters for optimal results\n\n")
    
    print(f"✓ Saved text summary: {summary_path}")
    
    
    print("\n" + "="*80)
    print("QUICK DEMO COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"\nExperiment 1 - Best: {exp1['num_points'][np.argmax(exp1['miou'])]} points → {max(exp1['miou']):.2f}% mIoU")
    print(f"Experiment 2 - Best: {exp2['strategy'][np.argmax(exp2['miou'])]} → {max(exp2['miou']):.2f}% mIoU")
    print(f"\nResults saved in: experiments/")
    print(f"  - demo_results_summary.png")
    print(f"  - demo_summary.txt")
    print(f"  - demo_exp1_num_points/results.json")
    print(f"  - demo_exp2_sampling_strategy/results.json")
    
    return all_results


if __name__ == "__main__":
    results = quick_demo()
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. Check 'experiments/demo_results_summary.png' for visualizations")
    print("2. Read 'experiments/demo_summary.txt' for detailed results")
