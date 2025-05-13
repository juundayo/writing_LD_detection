import os 

file_path = '/home/ml3/Desktop/Thesis/.venv/Data/GreekLetters' 

character_types = ['SMALL', 'CAPS']
sub_folders = ['SingleCharacters', 'DoubleCharacters']

images = []

for char_type in character_types:
    char_type_dir = os.path.join(file_path, char_type)
    if not os.path.exists(char_type_dir):
        print('TEST - Error in Caps-Small character type!')
        continue
        
    for sub_folder in sub_folders:
        sub_folder_dir = os.path.join(char_type_dir, sub_folder)
        if not os.path.exists(sub_folder_dir):
            print('TEST - Error in Double-Single character type!')
            continue
        
        for letter_folder in sorted(os.listdir(sub_folder_dir)):
            letter_path = os.path.join(sub_folder_dir, letter_folder)
            if not os.path.isdir(letter_path):
                print('TEST - Error in letters!')
                continue
                
            # Saving the letter as a class name.
            class_name = letter_folder
            
            for img_name in os.listdir(letter_path):
                
                # Skipping to avoid duplicates!
                if '_aug' in img_name:
                    img_to_remove = os.path.join(letter_path, img_name)
                    print('Removing ', img_to_remove)
                    os.remove(img_to_remove)
