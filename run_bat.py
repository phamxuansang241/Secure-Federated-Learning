import os


folder_path = ".\\batch_files\\fed_compress\\csic2010\\5_clients"

for root, dirs, files in os.walk (folder_path):
    for filename in files:
        if filename.endswith(".bat"):
            print(f"Executing .bat file: {os.path.join(root, filename)}")
            os.system(os.path.join(root, filename))
            print('\n\n')
        # os.system(os.path.join(folder_path, filename))
