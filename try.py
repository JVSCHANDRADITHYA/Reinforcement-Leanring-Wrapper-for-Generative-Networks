import time
import matplotlib.pyplot as plt
import sys
import numpy as np
# measure time to convert a string of "1"*d into an integer for varying digit lengths
sys.set_int_max_str_digits(9999999) 
# starting from 1 digit to 1 million digits
digit_sizes = np.logspace(0, 7, num=20, dtype=int)
times = []

for d in digit_sizes:
    s = "1" * d
    start = time.time()
    x = int(s)  # big integer creation
    end = time.time()
    times.append(end - start)

plt.figure(figsize=(18,5))
plt.plot(digit_sizes, times)
plt.xlabel("Number of digits")
plt.ylabel("Time to convert to int (seconds)")
plt.title("Cost of Creating Large Integers in Python")
plt.grid(True)

plt.savefig("plot.png")
print("Done")

