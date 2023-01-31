# Copyright © 2023 BAAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License")
import re
import time
import datetime


def year_now():
    """
        return the year
    """
    return str(datetime.datetime.now().year)


def month_now():
    """
        return the month
    """
    month_str = str(datetime.datetime.now().month)
    if len(month_str) == 1:
        month_str = '0' + month_str
    return month_str


def day_now():
    """
        return the day
    """
    day_str = str(datetime.datetime.now().day)
    day_str = day_str.zfill(2)
    return day_str


def ymd_hms(time_array):
    """
        Input: time array
        Return: time string xxxx-xx-xx xx:xx:xx
    """
    return time.strftime('%Y-%m-%d %H:%M:%S', time_array)


def return_stamp(time_str):
    """
        Input: time string
        Return: timestamp
    """
    souce_time_str = str(time_str.replace('/', '-').strip())
    if ':' in souce_time_str:
        if ' +' in souce_time_str:
            souce_time_str = souce_time_str.split(' +')[0].strip()
        elif ' -' in souce_time_str:
            souce_time_str = souce_time_str.split(' -')[0].strip()
        else:
            souce_time_str = souce_time_str
    else:
        souce_time_str = souce_time_str + ' ' + '00:00:00'
    try:
        time_array = time.strptime(souce_time_str, "%Y-%m-%d %H:%M:%S")
    except:
        time_array = time.strptime(souce_time_str, "%Y-%m-%d %H:%M")
    time_stamp = str(int(time.mktime(time_array)))
    return time_stamp


def return_stand(stamp):
    """
        Input: timestamp
        Return: standardized time xxxx-xx-xx xx:xx:xx
    """
    time_array2 = time.localtime(int(stamp))
    stand_time = str(ymd_hms(time_array2))
    return stand_time


def return_now_stamp():
    """
        Return: system's timestamp
    """
    return int(time.time())


def now_datetime():
    return return_stand(return_now_stamp())


def return_format_datetime(date_strs):
    """
        format the following date string style:
        ------------------------------
        '刚刚'
        '昨天'
        '前天'
        '09-3'
        '9-30'
        '09-03'
        '09-30'
        '1天前 -'
        '6小时前'
        '20分钟前'
        '2016.1.1'
        '2019-9-3'
        '昨天 2:00'
        '2016.1.01'
        '2019-09-3'
        '2019-9-03'
        '前天 14:00'
        '19年3月19日'
        '2019-09-03'
        '2016.10.01'
        '2018/09/22'
        '1537641480'
        '前天 14:00:08'
        '1670393724000'
        '09-06-18 10:44'
        '23.09.18 03:30'
        '2019年3月19日 -'
        '2019-1-01 10:18'
        '2016.10.01 23:42'
        '2018-09-23 12:58'
        '2019/03/22  11:41'
        '2019年03月22日11:41'
        '2019年05月08日 14:21'
        '2016.10.01 23:42:00'
        '2018-09-21T23:58:05Z'
        '2019/03/22  11:41:00'
        '2019-03-22  11:41:00'
        '6/3/2021, 9:01:47 AM'
        '9/23/2018 1:51:21 AM'
        '6/18/2021, 8:53:59 PM'
        '2018-09-22 20:41:15--'
        '2019年05月08日 14:21:03'
        '2010-06-26T00:00:00+00:00'
        '2018-09-22T20:41:15-04:00'
        '2018-09-23 04:30:58 +0200'
        '2018-09-22 21:19:30 -0500'
        --------------------------
        final format return style:
        xxxx-xx-xx xx:xx:xx
    """
    if date_strs.isdigit():
        if len(date_strs) >= 10:
            public_time = return_stand(str(date_strs)[:10])
        else:
            public_time = ''
    else:
        if ('M' in date_strs) and ('/' in date_strs):
            if ',' in date_strs:
                try:
                    sp_strs_list = date_strs.split(',')[0].split('/')
                    if len(sp_strs_list[-1]) > 2:
                        public_time = return_format_datetime(
                            sp_strs_list[-1] + '-' + sp_strs_list[0] + '-' + sp_strs_list[1] + ' ' + date_strs.split(',')[1].replace('PM', '').replace('AM', ''))
                    else:
                        public_time = date_strs
                except:
                    public_time = date_strs
            else:
                try:
                    sp_strs_list = date_strs.split(' ')[0].split('/')
                    if len(sp_strs_list[-1]) > 2:
                        public_time = return_format_datetime(
                            sp_strs_list[-1] + '-' + sp_strs_list[0] + '-' + sp_strs_list[1] + ' ' + date_strs.split(' ')[1].replace('PM', '').replace('AM', ''))
                    else:
                        public_time = date_strs
                except:
                    public_time = date_strs
        else:
            if 'T' in date_strs:
                try:
                    public_time = return_format_datetime(
                        date_strs.replace('T', ' ').split('+')[0].strip())
                except:
                    public_time = ''
            else:
                try:
                    mid_str = date_strs.replace('-', '').strip()
                    if '日' in mid_str:
                        deal_str = mid_str.replace(
                            '年', '-').replace('月', '-').replace('日', ' ').replace('  ', ' ').strip()
                        if len(deal_str) <= 8:
                            now_year = str(datetime.datetime.now().year)
                            public_time = return_stand(
                                return_stamp(str(now_year[:2]) + deal_str))
                        else:
                            public_time = return_stand(return_stamp(deal_str))
                    else:
                        if '刚刚' in mid_str:
                            public_time = now_datetime()
                        else:
                            try:
                                temp_strs = re.findall(r'\d+', mid_str)[0]
                            except:
                                temp_strs = ''
                            if '天前' in mid_str:
                                all_stamp = int(temp_strs) * 86399
                                public_time = return_stand(
                                    (return_now_stamp() - all_stamp))
                            elif '昨天' in mid_str:
                                all_stamp = 86399
                                public_time = return_format_datetime(return_stand(
                                    (return_now_stamp() - all_stamp)).split(' ')[0] + ' ' + date_strs.split('昨天')[1].strip())
                                if public_time == '':
                                    public_time = return_format_datetime(
                                        int(return_stand(return_now_stamp()) - all_stamp))
                            elif '前天' in mid_str:
                                all_stamp = 2 * 86399
                                public_time = return_format_datetime(return_stand(
                                    (return_now_stamp() - all_stamp)).split(' ')[0] + ' ' + date_strs.split('前天')[1].strip())
                                if public_time == '':
                                    public_time = return_format_datetime(
                                        int(return_stand(return_now_stamp()) - all_stamp))
                            elif '分钟前' in mid_str:
                                public_time = return_stand(
                                    (return_now_stamp() - int(temp_strs) * 60))
                            elif '小时前' in mid_str:
                                public_time = return_stand(
                                    (return_now_stamp() - int(temp_strs) * 3600))
                            else:
                                if '-' in date_strs:
                                    if len(date_strs) >= 4 and len(date_strs) <= 5:
                                        sp_temp = date_strs.split('-')
                                        str_one = str(sp_temp[0])
                                        str_two = str(sp_temp[1])
                                        if len(str_one) == 1:
                                            str_one = '0' + str_one
                                        if len(str_two) == 1:
                                            str_two = '0' + str_two
                                        public_time = year_now() + '-' + str_one + '-' + str_two + ' ' + '00:00:00'
                                    else:
                                        try:
                                            public_time = return_stand(
                                                return_stamp(date_strs))
                                        except:
                                            public_time = date_strs
                                else:
                                    if len(date_strs) == 4:
                                        public_time = date_strs + '-' + month_now() + '-' + day_now() + ' ' + '00:00:00'
                                    else:
                                        if '.' in date_strs:
                                            public_time = return_format_datetime(
                                                date_strs.replace('.', '-'))
                                        else:
                                            if len(date_strs) <= 10:
                                                public_time = return_stand(
                                                    return_stamp(date_strs))
                                            else:
                                                try:
                                                    public_time = return_stand(
                                                        return_stamp(date_strs))
                                                except:
                                                    public_time = date_strs
                except:
                    public_time = ''
    if 10 <= len(public_time) <= 14:
        try:
            public_time_ok = return_format_datetime(year_now()[:2] + public_time)
        except:
            public_time_ok = public_time
    else:
        public_time_ok = public_time
    if ('>' in public_time_ok) or ('<' in public_time_ok):
        return ''
    else:
        if len(date_strs) == 14:
            try:
                public_time_ok_split = public_time_ok.split('-')
                if int(public_time_ok_split[0]) > int(year_now()):
                    need_check_times = (year_now()[:2] + public_time_ok_split[2].split(' ')[
                                        0] + '-' + public_time_ok_split[1] + '-' + public_time_ok_split[0][2:] + ' ' + public_time_ok_split[2].split(' ')[1])
                else:
                    need_check_times = public_time_ok.replace('Z', '').strip()
            except:
                need_check_times = public_time_ok.replace('Z', '').strip()
        else:
            need_check_times = public_time_ok.replace('Z', '').strip()
        if need_check_times == '':
            try:
                need_check_times_ok = return_stand(date_strs)
            except:
                need_check_times_ok = ''
        else:
            if len(re.findall('-', need_check_times)) > 2:
                if need_check_times.endswith('--'):
                    need_check_times_ok = need_check_times.replace(
                        '--', '').strip()
                else:
                    try:
                        need_check_times_ok = need_check_times.replace(
                            ('-' + need_check_times.split('-')[-1]), '').strip()
                    except:
                        need_check_times_ok = ''
            else:
                need_check_times_ok = need_check_times
        if len(re.findall('-', need_check_times_ok)) > 2 or len(re.findall('/', need_check_times_ok)) > 1:
            return ''
        else:
            if 19 <= len(need_check_times_ok) <= 31:
                return need_check_times_ok
            else:
                return ''
