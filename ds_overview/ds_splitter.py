import splitfolders as sf
import os
import argparse

# Change the working directory to the location of the script
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def split(from_path, to_path, train, val, test, seed):
    """Splits the dataset into train, val, and test where train is 80%, val is 10% and test is 10%
    
    from_path (str): Path to folder containing unsplit images
    to_path (str): Path to folder where the split images will be stored
    train (float): Percentage of data to trainable data
    val (float): Percentage of data to be validation data
    test (float): Percentage of data to be test data
    seed (float): Seed for the image selection
    
    
    """
    sf.ratio(from_path, to_path, seed=seed, ratio=(train,val,test))

def define_arguments(parser):
    """Defines the arguments for the script

    Args:
        parser (argparse.ArgumentParser): The parser to add the arguments to
    """    
    parser.add_argument('-s', '--source', type=str, help='Path to folder containing unsplit images')
    parser.add_argument('-d', '--destionation', type=str, help='Path to folder where the split images will be stored')
    parser.add_argument('--train', type=float, default=0.8, help='Training set precentage (default: 0.8)')
    parser.add_argument('--val', type=float, default=0.1, help='Validation set precentage (default: 0.1)')
    parser.add_argument('--test', type=float, default=0.1, help='Test set precentage (default: 0.1)')
    parser.add_argument('--seed', type=float, default=1337, help='Seed for the image selection (default: 1337)')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog = 'DATA SET SPLITTER', description='Split dataset into train, val, and test set')
    define_arguments(parser)
    args = parser.parse_args()
    split(args.source, args.destionation, args.train, args.val, args.test, args.seed)

