import os


<<<<<<< HEAD
folder_path = ".\\batch_files\\fedavg\\csic2010\\5_clients"
=======
folder_path = ".\\batch_files\\fedavg\\smsspam"
>>>>>>> 007caa7b3a47db4acdafba79622c9ac62a652185

for root, dirs, files in os.walk (folder_path):
    for filename in files:
        if filename.endswith(".bat"):
            print(f"Executing .bat file: {os.path.join(root, filename)}")
            os.system(os.path.join(root, filename))
            print('\n\n')
        # os.system(os.path.join(folder_path, filename))
