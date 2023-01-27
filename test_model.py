import encryption_lib
import time


start_time = time.time()
encrypt = encryption_lib.EccEncryption(5, 350)
end_time = time.time()

print(end_time-start_time)
encrypt.calculate_server_public_key()

for i in range(5):
    encrypt.calculate_encoded_message_phase_one()