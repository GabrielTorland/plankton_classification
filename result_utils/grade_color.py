import re
import colorsys

def find_number_matches(text, pattern):
    return [match.group(1) for match in re.finditer(pattern, text)]

def get_table(file_path):
    with open(file_path) as file:
        content = file.read()
    return content

def get_colors_rgb(relative_differences, negative_color_hsv, positive_color_hsv):
    rgb_colors = []
    for ratio in relative_differences:
        if ratio < 0:
            color_hsv = negative_color_hsv[0], min(negative_color_hsv[1] * (abs(ratio) + 0.25), 1), negative_color_hsv[2] 
        else:
            color_hsv = positive_color_hsv[0], min(positive_color_hsv[1] * (ratio + 0.25), 1), positive_color_hsv[2]
        # Convert HSV to RGB
        color_rgb = tuple(round(i * 255) for i in colorsys.hsv_to_rgb(*color_hsv))
        rgb_colors.append(color_rgb)
    return rgb_colors

def replace_color_in_table(table, rgb_colors):

    color_iterator = iter(rgb_colors)

    def replace_color(match):
        try:
            rgb_color = next(color_iterator)
            return f'\\cellcolor[rgb]{{{rgb_color[0]/255},{rgb_color[1]/255},{rgb_color[2]/255}}}'
        except StopIteration:
            return match.group(0)

    pattern = r'\\cellcolor\{(color5|Salmon Pink)\}'
    new_table = re.sub(pattern, replace_color, table)
    return new_table

def write_table(table, file_path):
    with open(file_path, 'w') as file:
        file.write(table)

def main():
    file_path = "table.txt"
    number_pattern = r'\((-?0?\.\d*[1-9]\d*)\)'

    f1 = False

    table = get_table(file_path)

    matched_numbers = find_number_matches(table, number_pattern)

    # Convert the matched numbers to floats
    float_numbers = [float(num) for num in matched_numbers]

    if f1:
        # Second benchmark
        #for i in [3, 4, 8, 9, 13, 14, 18, 19, 23, 24, 27, 28, 32, 33, 37, 38, 42, 46, 47]:
        #    float_numbers[i] = -float_numbers[i]
        # Third benchmark
        for i in [3, 4, 8, 9, 12, 16, 17, 21, 22, 26, 27, 30, 31, 34, 35, 38, 42, 43]:
            float_numbers[i] = -float_numbers[i]
        # Forth benchmark
        #for i in [3, 4, 8, 9, 13, 14, 18, 19, 23, 24, 28, 29, 33, 34, 38, 39, 42, 43, 47, 48]:
        #    float_numbers[i] = -float_numbers[i]

    print("Extracted numbers as floats:", float_numbers)


    relative_differences = [num/0.2 for num in float_numbers]

    negative_color_hsv = (0/360, 28.6/100, 100/100) # Salmon red
    positive_color_hsv = (138/360, 28.6/100, 100/100)# Mint green

    rgb_colors = get_colors_rgb(relative_differences, negative_color_hsv, positive_color_hsv)

    table = replace_color_in_table(table, rgb_colors)

    write_table(table, file_path)



if __name__ == "__main__":
    main()
