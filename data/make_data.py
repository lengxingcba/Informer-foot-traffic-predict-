import math
import random
import warnings

import pandas as pd
import numpy as np
import re
import time
import datetime


def alpha_weekday(weekday):
    if weekday in range(5):
        return 3
    else:
        return 0.5


def alpha_month(month):
    if month in [3, 4, 5, 6, 9, 10, 11, 12, 1]:
        return 4
    else:
        return 0.5


def alpha_tempeature(tempeature):
    if tempeature in range(35, 40):
        return 0.8
    elif tempeature + 10 in range(20):
        return 0.7
    else:
        return 4

def alpha_time(time):
    if time in range(6):
        return 0.02
    elif time in range(6,9):
        return 1
    elif time in range(9,11):
        return 0.3
    elif time in range(11,13):
        return 1
    elif time in range(13,16):
        return 0.3
    elif time in range(16,18):
        return 1
    elif time in range(18,22):
        return 0.5
    else:
        return 0.02


def make_traffic(data):
    traffic = []
    for i in range(len(data)):
        date = data.iloc[i, 0]
        month = date.split("/")[1]
        t = date.split(" ")[-1].split(":")[0]
        tem = data.iloc[i, 1]
        weekday = data.iloc[i, 8]

        a_t = alpha_tempeature(int(tem))
        a_w = alpha_weekday(int(weekday))
        a_m = alpha_month(int(month))
        a_time=alpha_time(int(t))
        tra=math.sin((int(t) * 9 * math.pi) / 23)+random.randint(10,20)
        tra=tra*a_time
        tra=tra*a_w
        tra=tra*a_m
        traffic.append((tra*a_t)//1)

    data["traffic"] = traffic
    return data


# 气象数据处理
def read_txt_file(file_path, save: bool):
    with open(file_path, 'r', encoding="utf-8") as file:
        columns = file.readline().strip().split(';')
        data = [line.strip().split(';') for line in file]

    # 去除每个元素的引号
    data = [[element.strip('\"') for element in row] for row in data]

    df = pd.DataFrame(data, columns=columns)
    df = df[::-1].reset_index(drop=True)
    if save:
        df.to_csv("weather.csv", index=False)
    return df


def xls2csv(file_path):
    xls_file = pd.read_excel(file_path)
    name = file_path.split("/")[-1].split(".")[-2]
    # 将数据转换为csv文件
    csv_file = xls_file.to_csv(name + ".csv", index=False)
    print('转换完成！')


# from pandas.core.common import SettingWithCopyWarning
#
# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
# 时间格式标准化
def date_format(date, format="%d.%m.%Y %H:%M"):
    # 转换日期格式
    date_obj = datetime.datetime.strptime(date, format)
    date = date_obj.strftime("%Y/%m/%d %H:%M:%S")

    return date


def date2weekday(date):
    year, month, day = date.split(" ")[-2].split("/")
    return datetime.datetime(int(year), int(month), int(day)).strftime("%w")


# 时间标准化，加入星期 ，去除不需要的列,填充NaN为0
def preprocess(file_path, cols):
    """
    cols:不需要的列
    """
    data = pd.read_csv(file_path)
    data = data[::-1].reset_index(drop=True)
    data = data.drop(columns=cols)
    new_date = data["date"].apply(func=date_format)
    weekday = []
    for d in new_date:
        weekday.append(date2weekday(d))
    data['date'] = new_date
    data["weekday"] = weekday
    data.fillna(0, inplace=True)
    data=make_traffic(data)
    return data


data = preprocess("weather.csv", cols=["DD", "c", "VV","WW","W'W'"])
data.to_csv("foot_traffic.csv",index=False)

