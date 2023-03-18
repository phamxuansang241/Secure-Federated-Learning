import os
import os
import shutil


# training mode
dir_list_x = ['fedavg', 'fed_compress', 'fed_ecc', 'fed_elgamal']
# datasets
dir_list_y = ['csic2010', 'mnist', 'smsspam']
# number of clients
dir_list_z = [50]
# number of global epochs
dir_list_t = ['iid', 'noniid_labeldir']

batch_file_base = 'batch_files'

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
                    f.write(f'for /f "delims=" %%a in (\'dir /s /b .\\json_files\\{x}\\{y}\\{z}\\{t}\\50_global_epochs\\*.json\') do (\n')
                    f.write(f'    python main.py -cf "%%a"\n')
                    f.write(f'    echo python main.py -cf "%%a"\n')
                    f.write(f'    echo +++++++++\n')
                    f.write(f')\n\n')
                    f.write('echo Batch process complete\n')

