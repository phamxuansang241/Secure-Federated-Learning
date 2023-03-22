import os


folder_path = "generate_utils\\generated_files\\batch_files\\fedavg\\mnist"

for root, dirs, files in os.walk (folder_path):
    for filename in files:
        if filename.endswith(".bat"):
            print(f"Executing .bat file: {os.path.join(root, filename)}")
            os.system(os.path.join(root, filename))
            print('\n\n')
        # os.system(os.path.join(folder_path, filename))

