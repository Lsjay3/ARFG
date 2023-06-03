# Jay的开发时间：2022/10/11  9:50
import datetime
import time
t = "2020-10-16T17:36:00+08:00"
t2 = "2022-10-08T17:14:52.033762+08:00"
t3 = "2022-10-08T17:14:52.170939+08:00"

# new_t = datetime.datetime.strptime(t2, "%Y-%m-%dT%H:%M:%S.%f+08:00").strftime("%Y-%m-%d %H:%M:%S.%f")
new_t = datetime.datetime.strptime(t2, "%Y-%m-%dT%H:%M:%S.%f+08:00")
obj_stamp = int(time.mktime(new_t.timetuple()) * 1000000.0 + new_t.microsecond)
# '2020-10-16 17:36:00'

print(obj_stamp)
new_t = datetime.datetime.strptime(t3, "%Y-%m-%dT%H:%M:%S.%f+08:00")
obj_stamp = int(time.mktime(new_t.timetuple()) * 1000000.0 + new_t.microsecond)
# '2020-10-16 17:36:00'

print(obj_stamp)
t2 = "2020-10-16T17:36:00Z"
new_t2 = datetime.datetime.strptime(t2,"%Y-%m-%dT%H:%M:%SZ").strftime("%Y-%m-%d %H:%M:%S")
# '2020-10-16 17:36:00'

# 同理 什么格式都可以这样转， 如下
t3 = "2020年11月12日13点50分"
new_t3 = datetime.datetime.strptime(t3,"%Y年%m月%d日%H点%M分").strftime("%Y-%m-%d %H:%M:%S")
# '2020-11-12 13:50:00'