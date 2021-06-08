import numpy as np
import matplotlib.pyplot as plt


n_threads = np.array([1, 2, 4, 8, 16, 32, 64])
Times = np.array([438, 313, 146, 79, 49, 55, 35]) #s
Names = ['t2.micro', 't3.micro', 'c5.xlarge', 'c5.2xlarge', 'c5.4xlarge', 'c5a.8xlarge', 'c5a.16xlarge']

Times_threadRipper = np.array([399.4399435520172, 

])

plt.figure()
plt.title('Max vrille 8DoFs (version article bioptim)')
for i in range(7):
	plt.plot(n_threads[i], Times[i], 'o', label=Names[i])
plt.xlabel('Nb threads')
plt.ylabel('Temps resolution [s]')
plt.legend()
plt.show()

