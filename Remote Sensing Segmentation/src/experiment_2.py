from train_loveda import LoveDATrainer
strategies = ['random', 'uniform', 'balanced', 'boundary', 'cluster']

for strategy in strategies:
    config = {
        'data_root': './LoveDA',
        'scene': 'both',
        'image_size': 512,
        'num_classes': 7,
        'num_points': 200,  
        'sampling_strategy': strategy,  
        'encoder': 'resnet34',
        'batch_size': 8,
        'num_epochs': 30,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'num_workers': 4,
        'save_freq': 10,
        'checkpoint_dir': f'checkpoints/exp2_strategy_{strategy}',
        'ignore_background': False,
        'use_class_weights': False,
    }
    
    trainer = LoveDATrainer(config)
    trainer.train()