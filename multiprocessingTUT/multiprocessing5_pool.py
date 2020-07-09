# View more python learning tutorial on my Youtube and Youku channel!!!

# Youtube video tutorial: https://www.youtube.com/channel/UCdyjiB5H8Pu7aDTNVXTTpcg
# Youku video tutorial: http://i.youku.com/pythontutorial

import multiprocessing as mp


def job(x):
    return x * x


def multicore():
    # 进程池就是我们将所要运行的东西，放到池子里，Python会自行解决多进程的问题
    # 丢向Pool的函数有返回值，而Process的没有返回值
    # Pool默认大小是CPU的核数，我们也可以通过在Pool中传入processes参数即可自定义需要的核数量
    pool = mp.Pool(processes=2)
    # 在map()中需要放入函数和需要迭代运算的值，然后它会自动分配给CPU核，返回结果
    res = pool.map(job, range(10))
    print(res)
    # apply_async()中只能传递一个值，它只会放入一个核进行运算，
    # 但是传入值时要注意是可迭代的，所以在传入值后需要加逗号，形成参数列表，同时需要用get()方法获取返回值
    res = pool.apply_async(job, (2,))
    print(res.get())
    # 在此我们将apply_async() 放入迭代器中，定义一个新的multi_res
    multi_res = [pool.apply_async(job, (i,)) for i in range(10)]
    print([res.get() for res in multi_res])


if __name__ == '__main__':
    multicore()
