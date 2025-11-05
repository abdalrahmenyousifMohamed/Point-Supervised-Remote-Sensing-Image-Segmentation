from loveda_training import LoveDATrainer

def run_training():
    config = {
        'data_root': 'Remote Sensing Segmentation/data',
        'scene': 'urban',
        'image_size': 256,
        'num_classes': 7,
        'num_points': 100,
        'sampling_strategy': 'uniform',
        'encoder': 'resnet34',
        'batch_size': 4,
        'num_epochs': 5,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'optimizer': 'adamw',
        'num_workers': 0,  
        'save_freq': 5,
        'checkpoint_dir': 'checkpoints/test_run',
        'ignore_background': False,
        'use_class_weights': False,
    }
    
    trainer = LoveDATrainer(config)
    history = trainer.train()
    return history

if __name__ == '__main__':
    run_training()
