import numpy as np

list_of_nparrays = []
for i in range(21):
    temp = []
    for j in range(100):
        temp.append(np.random.randint(0, 35) / 10.0)
    temp1 = np.array(temp)

    temp.clear()
    for j in range(100):
        temp.append(np.random.randint(0, 500))
    temp2 = np.array(temp)

    list_of_nparrays.append(np.array([temp1, temp2]))


np.savez("test_data", *list_of_nparrays)
p = np.load("test_data.npz")

