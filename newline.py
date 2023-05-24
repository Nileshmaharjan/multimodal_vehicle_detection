with open('dataset/VEDAI/fold10test.txt', 'r') as file:
    line = file.readline().strip()  # Read the line and remove leading/trailing whitespace

    # Split the line by spaces and join the resulting list with newline characters
    new_line = '\n'.join(line.split(' '))

    with open('dataset/VEDAI/fold10test.txt', 'w') as modified_file:
        modified_file.write(new_line)
