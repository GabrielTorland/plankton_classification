import splitfolders as sf

def split(from_path, to_path)
    """Splits the dataset into train, val, and test where train is 80%, val is 10% and test is 10%
    
    from_path (str): Path to folder containing unsplit images
    to_path (str): Path to folder where the split images will be stored
    
    return: None
    """
    sf.ratio("./../baseline_training_set/", "./../baseline_trainingset_split/", seed=42, ratio=(0.8,0.1,0.1))