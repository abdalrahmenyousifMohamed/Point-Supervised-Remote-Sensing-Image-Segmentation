
from loveda_dataset import LoveDADataset


dataset = LoveDADataset(
    root_dir='./LoveDA',
    split='train',
    scene='both',
    image_size=512
)

print(f"Dataset size: {len(dataset)}")
print(f"Classes: {dataset.class_names}")


dataset.visualize_sample(0, save_path='test_sample.png')


dataset.get_class_distribution()