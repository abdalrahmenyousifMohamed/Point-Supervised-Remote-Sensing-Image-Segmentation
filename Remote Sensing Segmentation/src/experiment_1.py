num_points_list = [50, 100, 200, 500, 1000]
from train_loveda import LoveDATrainer
for num_points in num_points_list:
    config = {
        'data_root': '.Remote Sensing Segmentation/Train',
        'scene': 'both',
        'image_size': 512,
        'num_classes': 7,
        'num_points': num_points,  
        'sampling_strategy': 'uniform',  
        'encoder': 'resnet34',
        'batch_size': 8,
        'num_epochs': 30,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'num_workers': 4,
        'save_freq': 10,
        'checkpoint_dir': f'checkpoints/exp1_points_{num_points}',
        'ignore_background': False,
        'use_class_weights': False,
    }
    
    trainer = LoveDATrainer(config)
    trainer.train()
