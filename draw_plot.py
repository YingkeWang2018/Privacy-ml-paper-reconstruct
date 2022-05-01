import pickle
import matplotlib.pyplot as plt
objects = []
with (open("P=0.01,n=0.1,c=1.3,ind=0.8_YK_acc", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break


with (open("P=0.01,n=0.1,c=1.3,ind=0.2_YK_acc", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
with (open("P=0.01,n=0.1,c=1.3,ind=0.5_YK_acc", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
with (open("P=0.01,n=0.1,c=1.3,ind=jimmy_acc", "rb")) as openfile:
    while True:
        try:
            objects.append(pickle.load(openfile))
        except EOFError:
            break
x_lst = list(range(0, len(objects[0])))
y_lst1 = objects[0]
y_lst2 = objects[1]
y_lst3 = objects[2]
y_lst4 = objects[3]
plt.xlabel('epcohs')
plt.ylabel('test acc')
plt.plot(x_lst, y_lst1, label="P=0.01,n=0.1,c=1.3,ind=0.8_YK_acc")
plt.plot(x_lst, y_lst2, label="P=0.01,n=0.1,c=1.3,ind=0.2_YK_acc")
plt.plot(x_lst, y_lst3, label="P=0.01,n=0.1,c=1.3,ind=0.5_YK_acc")
plt.plot(x_lst, y_lst4, label="P=0.01,n=0.1,c=1.3,ind=jimmy_acc")
plt.legend(loc='upper left')
plt.show()
a = 5