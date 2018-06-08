import csv

with open('mpg.csv') as csvfile:
#    frd = csv.DictReader(csvfile)
    mpg = list(csv.DictReader(csvfile))

print('mpg.items()',mpg.items())

#print('mgg [:3]',mpg[:3])
print('mpg',type(mpg))
print('mpg',mpg.__len__())
print('mpg[0].keys = ',mpg[0].keys())
print('mpg[1].keys = ',mpg[1].keys())
print('mpg[0].values =',mpg[0].values())
print('mpg[1].values =',mpg[1].values())
print('mpg[0] =',mpg[0])
print('avg hwy =', sum(float(d['hwy']) for d in mpg) / len(mpg))
print('avg cty =', sum(float(d['cty']) for d in mpg) / len(mpg))
print('avg cty =%3.3f'  % (sum(float(d['cty']) for d in mpg) / len(mpg)))

cylinders = set(d['cyl'] for d in mpg)
print('cylinders =', cylinders)

CtyMpgByCyl = []

for c in cylinders:
    summpg = 0
    cyltypecount = 0
    for d in mpg:
        if d['cyl'] == c:
            summpg += float(d['cty'])
            cyltypecount += 1
    CtyMpgByCyl.append((c, summpg / cyltypecount))

CtyMpgByCyl.sort(key=lambda x: x[0])
print('CtyMpgByCly = ',CtyMpgByCyl)

vehicleclass = set(d['class'] for d in mpg)
print('vehicleclass = ', vehicleclass)


HwyMpgByClass = []
for t in vehicleclass:
    summpg = 0
    vclasscount = 0
    for d in mpg:
        if d['class'] == t:
            summpg += float(d['hwy'])
            vclasscount += 1
    HwyMpgByClass.append((t, summpg / vclasscount))

HwyMpgByClass.sort(key=lambda  x: x[1])
print('HwyMpgByClass =', HwyMpgByClass)


'''
print('frd =',frd)
print('\n\n\n')
print('mpg =',mpg[:-1])

'''