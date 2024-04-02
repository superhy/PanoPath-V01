'''
Created on 23 May 2023

@author: yang
'''
import os
import re
import shutil

from pdf2image.pdf2image import convert_from_path
import pytesseract


def clear_directory(path):
    for filename in os.listdir(path):
        file_path = os.path.join(path, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'Failed to delete {file_path}. Reason: {e}')

def convert_pdf_to_txt_ocr(src_folder_path, dest_folder_path):
    '''
    '''
    
    # Clear the destination directory
    clear_directory(dest_folder_path)
    
    for subdir, dirs, files in os.walk(src_folder_path):
        for file in files:
            src_file_path = subdir + os.sep + file
            if src_file_path.lower().endswith(".pdf") or src_file_path.endswith('.PDF'):
                print(f"Processing file: {src_file_path}")
                # Convert the PDF to images
                images = convert_from_path(src_file_path)
                text = ""
                for i in range(len(images)):
                    # Perform OCR on the image
                    text += pytesseract.image_to_string(images[i])
                
                # Construct the destination file path
                file_base_name = os.path.basename(src_file_path)
                match = re.search(r"(TCGA-\w{2}-\w{4})", file_base_name)
                if match:
                    new_base_name = match.group(1)
                    dest_file_path = dest_folder_path + os.sep + new_base_name + ".txt"
                else:
                    dest_file_path = dest_folder_path + os.path.splitext(src_file_path[len(src_folder_path):])[0] + ".txt"
                
                # Create the destination directory if it does not exist
                os.makedirs(os.path.dirname(dest_file_path), exist_ok=True)
                # Write the text to the destination file
                with open(dest_file_path, 'w', encoding='utf-8', errors='ignore') as f:
                    f.write(text)

if __name__ == '__main__':
    '''
    sometimes, will have some unit test at here
    '''
    convert_pdf_to_txt_ocr('D:/PanoPath-filesys/TCGA/pdf', 'D:/PanoPath-filesys/TCGA/text')
    
    
    