from src.utils.SimpleDataProcessor import DataProcessor

if __name__ == '__main__':

    # labels文件中不要含有空格
    dataset_dir = '/Users/cola/Downloads/braille'
    img_dir = dataset_dir + '/img'
    test_dir = dataset_dir + '/test'
    valid_dir = dataset_dir + '/valid'

    dp = DataProcessor()

    dp.clean_output_img(img_dir)
    dp.add_data(img_dir, 5000)
    dp.update_img_labels(dataset_dir)

    # dp.clean_output_img(test_dir)
    # dp.add_data(test_dir, 500)
    # dp.update_test_labels(dataset_dir)
    #
    dp.clean_output_img(valid_dir)
    dp.add_data(valid_dir, 2000)
    dp.update_valid_labels(dataset_dir)




