'''
File used for preprocessing the greek word dictionary. It includes over 
1 million words - most of which are not actually used often - so we
create a new dictionary that includes every row of the original one
that has a frequency greater than 1.

Original disctionary layout:
    ⇒ {word} {frequency}
    
Original dictionary: 1047200

Processed dictionary: 580641

Removed rows: 466559
'''
input_file = 'Thesis/spell_dict_with_freq.dic'  
output_file = 'Thesis/filtered_dictionary.dic' 

original_total = 0
processed_total = 0

with open(input_file, 'r', encoding='utf-8') as infile, \
     open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        original_total += 1
        parts = line.strip().split()
        if len(parts) == 2:  
            try:
                frequency = int(parts[-1]) # [-1] is the frequency.
                if frequency > 1:
                    outfile.write(line)
                    processed_total += 1
            except ValueError:
                print(f"Skipping line: {line.strip()}")

removed_rows = original_total - processed_total

print('Original dictionary:')
print(f'Rows: {original_total}\n')

print('Processed dictionary:')
print(f'Rows: {processed_total}\n')

print(f'Removed rows: {removed_rows}')