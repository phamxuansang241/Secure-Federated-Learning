import encryption_lib
import numpy as np
import time

nb_client = 5
mtx_size = 5

start_time = time.time()
encrypt = encryption_lib.EccEncryption(nb_client, mtx_size)
encrypt.client_setup_private_params()
encrypt.client_setup_keys()
end_time = time.time()
print("Time to initialize keys for clients: ", end_time-start_time)

start_time = time.time()
encrypt.calculate_server_public_key()
end_time = time.time()
print("Time to calculate public keys for server: ", end_time-start_time)

encrypt.perform_phase_one(short_ver=False)

sum_mtx_M = np.zeros((mtx_size, mtx_size))
sum_mtx_N = np.zeros((mtx_size, mtx_size))
for i in range(nb_client):
    sum_mtx_M = sum_mtx_M + encrypt.client_private_parameter['m_i'][i]
    sum_mtx_N = sum_mtx_N + encrypt.client_private_parameter['n_i'][i]

print('sum of matrix m_i: \n', sum_mtx_M)
print('sum of matrix n_i: \n', sum_mtx_N)
print('Matrix M: \n', encrypt.server_decoded_message['M'])
print('Matrix N: \n', encrypt.server_decoded_message['N'])

np.testing.assert_array_equal(sum_mtx_M, encrypt.server_decoded_message['M'], err_msg="Two matrice not equal")