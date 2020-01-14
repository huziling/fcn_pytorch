import multiprocessing
import subprocess

def work(cmd):
    return subprocess.call(cmd, shell=True, executable="/bin/bash")

if __name__ == '__main__':
    count = multiprocessing.cpu_count()
    # count = 1
    pool = multiprocessing.Pool(processes=count)

    cmds = []
    num = 5000
    for i in range(count):
        cmds.append('./run.sh {} generate{}_'.format(num, i))
    print(cmds)
    print(pool.map(work, cmds))

    cmds = []
    cmds.append('./dealfinal.sh')
    pool.map(work, cmds)