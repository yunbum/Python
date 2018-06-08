import datetime as dt
import time as tm

print('tm.time()', tm.time())

dtnow = dt.datetime.fromtimestamp(tm.time())
print('dtnow.year =', dtnow.year)
print('dtnow.month =',dtnow.month)
print('dtnow.day =', dtnow.day)
print('dtnow.hour =', dtnow.hour)
print('dtnow. minute =', dtnow.minute)
print('dtnow.second = ', dtnow.second)

delta = dt.timedelta(days = 100)
today = dt.date.today()
print('past =',today - delta)

date1 = dt.date(2018,3,3)
diff_day = date1 - dt.date(2018,3,4)
print('date1 =', date1)
#date2 = date1 + dt.days(1)
print('diff_day =', diff_day)


today = dt.date.today()

print('today = ',today)






