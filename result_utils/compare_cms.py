

def get_trainings_results():
    """
    Returns the results of the first, second, third, and forth training sessions.
    
    Returns:
        Tuple[List[List[float]], List[List[float]], List[List[float]], List[List[float]]]: A tuple containing four lists of lists of floats.
    """
    first_training = [
        [0.59, 0.44, 0.53, 0.47, 0.75, 0.91, 0.94, 0.91, 0.94, 1.00],
        [0.18, 0.29, 0.38, 0.34, 0.46, 0.56, 0.52, 0.56, 0.54, 0.51],
        [0.98, 0.94, 0.96, 0.97, 0.97, 0.99, 0.99, 0.99, 0.99, 0.99],
        [0.85, 0.55, 0.70, 0.74, 0.75, 0.85, 0.85, 0.86, 0.85, 0.85],
        [0.97, 0.95, 0.97, 0.95, 0.97, 0.98, 0.99, 0.99, 0.98, 0.98],
        [0.96, 0.88, 0.94, 0.95, 0.95, 0.97, 0.97, 0.97, 0.97, 0.97],
        [0.14, 0.21, 0.29, 0.27, 0.32, 0.41, 0.40, 0.41, 0.41, 0.39],
        [0.98, 0.94, 0.98, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99],
        [0.17, 0.06, 0.45, 0.43, 0.45, 0.68, 0.64, 0.66, 0.70, 0.79],
        [0.83, 0.58, 0.85, 0.83, 0.87, 0.89, 0.89, 0.90, 0.89, 0.87],
        [0.94, 0.90, 0.94, 0.93, 0.95, 0.96, 0.96, 0.96, 0.95, 0.96],
        [0.99, 0.93, 0.99, 0.98, 0.99, 0.99, 0.99, 1.00, 0.99, 1.00]
    ]

    second_training = [    
        [0.91, 0.91, 0.81, 0.81, 0.88, 0.97, 0.97, 1.00, 0.94, 1.00],
        [0.60, 0.54, 0.50, 0.51, 0.55, 0.72, 0.72, 0.72, 0.72, 0.71],
        [0.97, 0.92, 0.97, 0.97, 0.97, 0.98, 0.99, 0.99, 0.99, 0.99],
        [0.84, 0.69, 0.81, 0.80, 0.83, 0.89, 0.90, 0.90, 0.89, 0.88],
        [0.98, 0.93, 0.97, 0.97, 0.97, 0.99, 0.99, 0.99, 0.99, 0.99],
        [0.92, 0.69, 0.82, 0.86, 0.86, 0.93, 0.93, 0.92, 0.92, 0.93],
        [0.51, 0.42, 0.36, 0.39, 0.45, 0.54, 0.54, 0.53, 0.57, 0.51],
        [0.97, 0.91, 0.98, 0.97, 0.98, 0.98, 0.98, 0.98, 0.98, 0.98],
        [0.89, 0.66, 0.70, 0.77, 0.66, 0.89, 0.91, 0.87, 0.83, 0.87],
        [0.92, 0.78, 0.88, 0.89, 0.90, 0.92, 0.92, 0.91, 0.92, 0.91],
        [0.91, 0.82, 0.94, 0.94, 0.94, 0.95, 0.94, 0.95, 0.94, 0.94],
        [1.00, 0.97, 0.99, 0.99, 0.99, 1.00, 1.00, 1.00, 1.00, 1.00]
    ]

    third_training = [
        [0.91, 0.91, 0.81, 0.81, 0.84, 0.97, 0.97, 1.00, 0.97, 1.00],
        [0.65, 0.50, 0.55, 0.53, 0.59, 0.69, 0.68, 0.75, 0.69, 0.66],
        [0.98, 0.93, 0.97, 0.98, 0.98, 0.99, 0.99, 0.99, 0.99, 0.99],
        [0.89, 0.64, 0.72, 0.73, 0.75, 0.85, 0.84, 0.87, 0.85, 0.85],
        [0.97, 0.92, 0.96, 0.97, 0.96, 0.99, 0.98, 0.99, 0.98, 0.98],
        [0.94, 0.85, 0.91, 0.93, 0.94, 0.96, 0.96, 0.96, 0.96, 0.96],
        [0.31, 0.39, 0.41, 0.37, 0.41, 0.53, 0.52, 0.51, 0.55, 0.51],
        [0.98, 0.93, 0.98, 0.98, 0.98, 0.98, 0.98, 0.99, 0.98, 0.98],
        [0.85, 0.64, 0.70, 0.68, 0.60, 0.89, 0.85, 0.89, 0.79, 0.87],
        [0.84, 0.77, 0.86, 0.87, 0.85, 0.91, 0.91, 0.90, 0.90, 0.90],
        [0.94, 0.85, 0.94, 0.94, 0.95, 0.95, 0.95, 0.94, 0.94, 0.95],
        [0.99, 0.97, 0.99, 0.99, 0.99, 0.99, 0.99, 1.00, 0.99, 0.99]
    ]

    forth_training = [
        [0.91, 1.00, 0.91, 0.97, 0.97, 0.84, 0.88, 0.84, 0.97, 0.97],
        [0.59, 0.71, 0.70, 0.70, 0.75, 0.58, 0.44, 0.63, 0.59, 0.71],
        [0.97, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.97, 0.99, 0.99],
        [0.71, 0.84, 0.88, 0.88, 0.88, 0.81, 0.83, 0.83, 0.85, 0.89],
        [0.90, 0.96, 0.99, 0.99, 0.99, 0.98, 0.95, 0.96, 0.97, 0.99],
        [0.91, 0.98, 0.97, 0.97, 0.97, 0.96, 0.94, 0.96, 0.98, 0.98],
        [0.47, 0.52, 0.52, 0.56, 0.55, 0.49, 0.44, 0.46, 0.52, 0.56],
        [0.96, 0.99, 0.99, 0.99, 0.99, 0.98, 0.98, 0.98, 0.99, 0.99],
        [0.89, 0.89, 0.94, 0.87, 0.85, 0.79, 0.81, 0.91, 0.70, 0.91],
        [0.84, 0.92, 0.94, 0.94, 0.94, 0.89, 0.90, 0.88, 0.90, 0.92],
        [0.89, 0.94, 0.96, 0.96, 0.96, 0.94, 0.95, 0.95, 0.95, 0.96],
        [0.99, 1.00, 1.00, 1.00, 1.00, 0.99, 0.98, 0.99, 1.00, 1.00]
    ]

    return first_training, second_training, third_training, forth_training

def calc_diff(new_training, baseline):
    """
    Calculates the difference between the new training and the baseline training results.
    
    Args:
        new_training (List[List[float]]): The new training results.
        baseline (List[List[float]]): The baseline training results.
        
    Returns:
        List[List[float]]: A list of lists of floats representing the differences between the new
        training and the baseline training results.
    """
    return [[new - base for new, base in zip(new_row, base_row)]
            for new_row, base_row in zip(new_training, baseline)]

def print_diffs(diffs):
    """
    Prints the differences between the new training and the baseline training results.
    
    Args:
        diffs (List[List[float]]): A list of lists of floats representing the differences between
        the new training and the baseline training results.
    """
    for row in diffs:
        print(', '.join(f"{diff:.2f}" for diff in row))

def main():
    """
    The main function of the script. Retrieves training results, calculates the differences
    between them, and prints the differences.
    """
    t1, t2, t3, t4 = get_trainings_results()
    diffs = calc_diff(t2, t1)
    print_diffs(diffs)
    print("---------------------------------------------------------------------------")
    diffs = calc_diff(t3, t1)
    print_diffs(diffs)
    print("---------------------------------------------------------------------------")
    diffs = calc_diff(t4, t3)
    print_diffs(diffs)

if __name__ == "__main__":
    main()
