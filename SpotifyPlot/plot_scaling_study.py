import pandas as pd
import matplotlib.pyplot as plt


def conditionOMPData():
    serial_times = [6.301945, 6.497803, 6.248677]

    total_serial = 0
    for time in serial_times:
        total_serial += time
    total_serial /= len(serial_times)

    omp_times = [
        [6.514394, 6.601567, 6.590459], 
        [3.308685, 3.353846, 3.371353], 
        [2.298171, 2.299603, 2.276771], 
        [1.768315, 1.746593, 1.765074], 
        [1.442557, 1.446394, 1.465262], 
        [1.283129, 1.257047, 1.225724], 
        [1.286735, 1.293697, 1.263102], 
        [1.213842, 1.165172, 1.190828], 
        [0.922001, 0.890765, 0.907381], 
        [0.870513, 0.900094, 0.826877]]

    omp_threads = [1, 2, 3, 4, 5, 6, 7, 8, 12, 16]

    condensed_omp_times = []

    for i in range(0, len(omp_times)):
        new_time = 0
        for time in omp_times[i]:
            new_time += time
        new_time /= len(omp_times[i])
        condensed_omp_times.append(new_time)

    return omp_threads, condensed_omp_times, total_serial

def conditionGPUData():
    gpu_threads = [16, 32, 64, 128, 256, 512]

    gpu_times = [
        [0.719224, 0.721331, 0.711376],
        [0.536088, 0.530974, 0.531105],
        [1.607722, 1.459259, 1.469630],
        [1.603406, 1.630459, 1.605275],
        [1.613788, 1.631238, 1.620053],
        [1.587205, 1.824759, 1.607862]
    ]

    condensed_gpu_times = []
    for time_list in gpu_times:
        new_time = 0
        for time in time_list:
            new_time += time
        new_time /= len(time_list)
        condensed_gpu_times.append(new_time)

    return gpu_threads, condensed_gpu_times


def plotSharedTime():

    omp_threads, condensed_omp_times, total_serial = conditionOMPData()

    plt.plot(omp_threads, condensed_omp_times, label='OpenMP', marker='o', linestyle='-')
    plt.plot(1, total_serial, label='Serial', marker='x', markersize=10, linestyle='-')

    plt.xlabel('Threads')
    plt.ylabel('Time')
    plt.title('Scaling Study Time (Shared Memory)')

    plt.legend()

    plt.show()


def plotSharedEfficiency():
    omp_threads, condensed_omp_times, total_serial = conditionOMPData()

    efficiency = []
    for i in range(len(omp_threads)):
        theoritical_efficiency = total_serial / omp_threads[i]
        efficiency.append(theoritical_efficiency / condensed_omp_times[i])

    plt.plot(omp_threads, efficiency, label='OpenMP', marker='o', linestyle='-')

    plt.xlabel('Threads')
    plt.ylabel('Efficiency')
    plt.title('Scaling Study Efficiency (Shared Memory)')

    plt.legend()

    plt.show()

def plotGPUBlocks():
    gpu_threads, condensed_gpu_times = conditionGPUData()

    plt.plot(gpu_threads, condensed_gpu_times, label='GPU', marker='o', linestyle='-')

    plt.xlabel('Threads')
    plt.ylabel('Time')
    plt.title('Scaling Study Block Size (Shared Memory)')

    plt.legend()

    plt.show()

def main():
    plotSharedTime()
    plotSharedEfficiency()
    plotGPUBlocks()


if __name__ == "__main__":
    main()