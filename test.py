import mouse
import time

time_start = time.time()
for i in range(1000):
    time.sleep(0.01666666666665151)
print((time.time() - time_start)/1000)
