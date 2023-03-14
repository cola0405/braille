from src.utils.SimpleTrainModel import TrainModel
from src.utils.SimpleDataProcessor import DataProcessor


tm = TrainModel()
img_dir = '/Users/cola/Downloads/braille'
tm.load_dataset(img_dir)
tm.train_model('braille.h5', 10)

