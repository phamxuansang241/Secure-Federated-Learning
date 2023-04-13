import os
import os
import shutil


# training mode
dir_list_x = ['dssgd']
# datasets
dir_list_y = ['csic2010', 'smsspam']
# dir_list_y = ['mnist']
# number of clients
dir_list_z = [10]
# dir_list_z = [5, 10, 20, 40, 50]
# number of global epochs

dir_list_t = ['iid', 'noniid_label_dir']
# dir_list_t = ['iid', 'noniid_label_quantity']


batch_file_base = 'generate_utils/generated_files/batch_files'

if not os.path.exists(batch_file_base):
    os.makedirs(batch_file_base)

for x in dir_list_x:
    folder = f'{batch_file_base}/{x}'
    if os.path.isdir(folder):
        shutil.rmtree(folder)

    for y in dir_list_y:
        for z in dir_list_z:
            z = str(z) + '_clients'
            for t in dir_list_t:
                # t = str(t) + '_global_epochs'
                batch_file = f'{batch_file_base}/{x}/{y}/{z}/{t}/50_global_epochs/batch.bat'
                
                if not os.path.exists(os.path.dirname(batch_file)):
                    os.makedirs(os.path.dirname(batch_file))
                
                with open(batch_file, 'w') as f:
                    f.write('@echo off\n\n')
                    f.write(f'    echo +++++++++\n')
                    f.write(f'for /f "delims=" %%a in (\'dir /s /b .\\generate_utils\\generated_files\\json_files\\{x}\\{y}\\{z}\\{t}\\50_global_epochs\\*.json\') do (\n')
                    f.write(f'    python main.py -cf "%%a"\n')
                    f.write(f'    echo python main.py -cf "%%a"\n')
                    f.write(f'    echo +++++++++\n')
                    f.write(f')\n\n')
                    f.write('echo Batch process complete\n')

