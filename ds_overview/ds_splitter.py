import splitfolders as sf

def split(from_path, to_path, train, val, test)
    """Splits the dataset into train, val, and test where train is 80%, val is 10% and test is 10%
    
    from_path (str): Path to folder containing unsplit images
    to_path (str): Path to folder where the split images will be stored
    train (float): Percentage of data to trainable data
    val (float): Percentage of data to be validation data
    test (float): Percentage of data to be test data
    
    
    """
    sf.ratio(from_path, to_path, seed=42, ratio=(train,val,test))