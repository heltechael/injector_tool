with open('csv_data_IGIS/all_thumbs_from_unapproved.csv', 'r') as input_file, open('csv_data_IGIS/all_thumbs_from_unapproved_subset.csv', 'w') as output_file:
    for i, line in enumerate(input_file):
        if i < 10000:
            output_file.write(line)
        else:
            break