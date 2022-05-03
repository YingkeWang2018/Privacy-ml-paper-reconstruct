import pickle
import matplotlib.pyplot as plt
objects = []
with (open("p=0.01,n=0.03,c=2.3_seed=1234_batch=500paper_acc", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break


with (open("p=0.01,n=0.03,c=2.3_seed=1234_batch=500standard_acc", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break

x_lst = list(range(0, len(objects[0])))
y_lst1 = objects[0]
y_lst2 = objects[1]

plt.xlabel('iterations')
plt.ylabel('test acc')
plt.plot(x_lst, y_lst1, label="p=0.01,n=0.03,c=2.3_seed=1234_batch=500standard_acc")
plt.plot(x_lst, y_lst2, label="p=0.01,n=0.03,c=2.3_seed=1234_batch=500paper_acc")

plt.legend(loc='upper left')

plt.show()
a = 5