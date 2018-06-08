people = ['Dr. Christopher Brooks', 'Dr. Kevyn Collins-Thompson', 'Dr. VG Vinod Vydiswaran', 'Dr. Daniel Romero']

def split_title_n_name(person):
    title = person.split()[0]
    lastname = person.split()[-1]
    return '{} {}'.format(title, lastname)

print(list(map(split_title_n_name, people)))


a = [1.2, 2.5, 3.7, 4.6]
b = list(map(int, a))
print('b =',b)
c = list(map (int, range(10)))
print('c =',c)
d = list(map (str, range(10)))
print('d =',d)

for i in a:
    print('i =',i)
#    print('a[i]',a[i])

x = dict.fromkeys(d)

print('x =',x)