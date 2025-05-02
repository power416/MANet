import numpy as np
from scipy import signal
from obspy import Stream
import datetime
from datetime import timedelta


def normalize(data, mode='std'):
    # Normalize waveforms in each batch
    data -= np.mean(data, axis=0, keepdims=True)
    if mode == 'max':
        max_data = np.max(data, axis=0, keepdims=True)
        max_data[max_data == 0] = 1
        data /= max_data

    elif mode == 'std':
        std_data = np.std(data, axis=0, keepdims=True)
        std_data[std_data == 0] = 1
        data /= std_data
    return data


def inttostr(a, mode=0):
    str_a = str(int(a))
    if mode == 0:
        if len(str_a) == 1:
            return '0'+str_a
        else:
            return str_a
        
    if mode == 1:
        if len(str_a) == 1:
            return '00'+str_a
        elif len(str_a) == 2:
            return '0'+str_a
        else:
            return str_a


def timee(newtime_start):
    new_h_start = newtime_start // (3600*1000)
    new_m_start = (newtime_start - (new_h_start* (3600*1000))) // (60*1000)
    new_s_start = (newtime_start - (new_h_start* (3600*1000)) - (new_m_start* (60*1000))) // 1000
    new_ms_start = newtime_start - (new_h_start* (3600*1000)) - (new_m_start* (60*1000)) - new_s_start*1000
    newtime = inttostr(new_h_start, 0) + ':' + inttostr(new_m_start, 0) + ':' + inttostr(new_s_start, 0) + '.' + inttostr(new_ms_start, 1)

    return newtime


def trans_time(num1):
    day_time = 24 * 3600 * 1000

    day_diff = num1 // day_time

    num2 = num1 - day_diff * day_time

    return day_diff, num2


def transfor(st):

    component0 = st[0].stats.component
    component1 = st[1].stats.component
    component2 = st[2].stats.component

    st2 = Stream()

    if component0 == 'Z' and component1 == 'N' and component2 == 'E':
        st2 += st[0]
        st2 += st[1]
        st2 += st[2]
    elif component0 == 'Z' and component1 == 'E' and component2 == 'N':
        st2 += st[0]
        st2 += st[2]
        st2 += st[1]
    elif component0 == 'N' and component1 == 'Z' and component2 == 'E':
        st2 += st[1]
        st2 += st[0]
        st2 += st[2]
    elif component0 == 'N' and component1 == 'E' and component2 == 'Z':
        st2 += st[2]
        st2 += st[0]
        st2 += st[1]
    elif component0 == 'E' and component1 == 'Z' and component2 == 'N':
        st2 += st[1]
        st2 += st[2]
        st2 += st[0]
    elif component0 == 'E' and component1 == 'N' and component2 == 'Z':
        st2 += st[2]
        st2 += st[1]
        st2 += st[0]

    return st2


def time_calculate(start_time, diff):
    start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d%H:%M:%S')
    end_time = start_time + timedelta(hours=24*diff)
    str_end_time = str(end_time)[:10]

    return str_end_time


# def time_calculate2(date, timee, value):
#     value = value // 10
#     s = value % 60
#     m = (value // 60) % 60
#     h = value // 3600

#     start_time = date + timee
#     start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d%H:%M:%S')
#     end_time = start_time + timedelta(hours=h) + timedelta(minutes=m) + timedelta(seconds=s)
#     str_end_time = str(end_time)
#     str_end_time1 = str_end_time[11:19]

#     return str_end_time1


def preprocessing(st):
    data = None
    length = None
    newtime_z_start = None
    newtime_n_start = None
    newtime_e_start = None

    st = transfor(st)

    # print(st)

    z_h_start = int(str(st[0].stats.starttime)[11:13])
    z_m_start = int(str(st[0].stats.starttime)[14:16])
    z_s_start = int(str(st[0].stats.starttime)[17:19])
    z_ms_start = int(str(st[0].stats.starttime)[20:23])

    n_h_start = int(str(st[1].stats.starttime)[11:13])
    n_m_start = int(str(st[1].stats.starttime)[14:16])
    n_s_start = int(str(st[1].stats.starttime)[17:19])
    n_ms_start = int(str(st[1].stats.starttime)[20:23])

    e_h_start = int(str(st[2].stats.starttime)[11:13])
    e_m_start = int(str(st[2].stats.starttime)[14:16])
    e_s_start = int(str(st[2].stats.starttime)[17:19])
    e_ms_start = int(str(st[2].stats.starttime)[20:23])

    time_z_start = (z_h_start * 3600 + z_m_start * 60 + z_s_start) * 1000 + z_ms_start
    time_n_start = (n_h_start * 3600 + n_m_start * 60 + n_s_start) * 1000 + n_ms_start
    time_e_start = (e_h_start * 3600 + e_m_start * 60 + e_s_start) * 1000 + e_ms_start

    time_max = np.max([time_z_start, time_n_start, time_e_start])

    # z_h_end = int(str(st[0].stats.endtime)[11:13])
    # z_m_end = int(str(st[0].stats.endtime)[14:16])
    # z_s_end = int(str(st[0].stats.endtime)[17:19])
    # z_ms_end = int(str(st[0].stats.endtime)[20:23])

    # n_h_end = int(str(st[1].stats.endtime)[11:13])
    # n_m_end = int(str(st[1].stats.endtime)[14:16])
    # n_s_end = int(str(st[1].stats.endtime)[17:19])
    # n_ms_end = int(str(st[1].stats.endtime)[20:23])

    # e_h_end = int(str(st[2].stats.endtime)[11:13])
    # e_m_end = int(str(st[2].stats.endtime)[14:16])
    # e_s_end = int(str(st[2].stats.endtime)[17:19])
    # e_ms_end = int(str(st[2].stats.endtime)[20:23])

    # time_z_end = (z_h_end * 3600 + z_m_end * 60 + z_s_end) * 1000 + z_ms_end
    # time_n_end = (n_h_end * 3600 + n_m_end * 60 + n_s_end) * 1000 + n_ms_end
    # time_e_end = (e_h_end * 3600 + e_m_end * 60 + e_s_end) * 1000 + e_ms_end

    time_z_diff = time_max - time_z_start
    time_n_diff = time_max - time_n_start
    time_e_diff = time_max - time_e_start

    if time_z_diff<50:
        if time_n_diff<50:
            if time_e_diff<50:  # z<50, n<50, e<50
                z = st[0].data
                n = st[1].data
                e = st[2].data

                m1, n1 = signal.butter(2, [0.01, 0.08], 'bandpass')
                z = signal.filtfilt(m1, n1, z)
                m2, n2 = signal.butter(2, [0.01, 0.08], 'bandpass')
                n = signal.filtfilt(m2, n2, n)
                m3, n3 = signal.butter(2, [0.01, 0.08], 'bandpass')
                e = signal.filtfilt(m3, n3, e)

                newtime_z_start = time_z_start
                newtime_n_start = time_n_start
                newtime_e_start = time_e_start
                length = np.min([len(z), len(n), len(e)])
                data = np.zeros([length, 3])
                data[:, 0] = z[:length]
                data[:, 1] = n[:length]
                data[:, 2] = e[:length]

            else:  # z<50, n<50, e>=50
                num = time_e_diff // 50
                z = st[0].data
                n = st[1].data
                e = st[2].data[num:]

                m1, n1 = signal.butter(2, [0.01, 0.08], 'bandpass')
                z = signal.filtfilt(m1, n1, z)
                m2, n2 = signal.butter(2, [0.01, 0.08], 'bandpass')
                n = signal.filtfilt(m2, n2, n)
                m3, n3 = signal.butter(2, [0.01, 0.08], 'bandpass')
                e = signal.filtfilt(m3, n3, e)

                newtime_z_start = time_z_start
                newtime_n_start = time_n_start
                newtime_e_start = time_e_start + 50 * num
                length = np.min([len(z), len(n), len(e)])
                data = np.zeros([length, 3])
                data[:, 0] = z[:length]
                data[:, 1] = n[:length]
                data[:, 2] = e[:length]

        else:
            if time_e_diff<50:  # z<50, n>=50, e<50
                num = time_n_diff // 50
                z = st[0].data
                n = st[1].data[num:]
                e = st[2].data

                m1, n1 = signal.butter(2, [0.01, 0.08], 'bandpass')
                z = signal.filtfilt(m1, n1, z)
                m2, n2 = signal.butter(2, [0.01, 0.08], 'bandpass')
                n = signal.filtfilt(m2, n2, n)
                m3, n3 = signal.butter(2, [0.01, 0.08], 'bandpass')
                e = signal.filtfilt(m3, n3, e)

                newtime_z_start = time_z_start
                newtime_n_start = time_n_start + 50 * num
                newtime_e_start = time_e_start
                length = np.min([len(z), len(n), len(e)])
                data = np.zeros([length, 3])
                data[:, 0] = z[:length]
                data[:, 1] = n[:length]
                data[:, 2] = e[:length]

            else:  # z<50, n>=50, e>=50
                num1 = time_n_diff // 50
                num2 = time_e_diff // 50
                z = st[0].data
                n = st[1].data[num1:]
                e = st[2].data[num2:]

                m1, n1 = signal.butter(2, [0.01, 0.08], 'bandpass')
                z = signal.filtfilt(m1, n1, z)
                m2, n2 = signal.butter(2, [0.01, 0.08], 'bandpass')
                n = signal.filtfilt(m2, n2, n)
                m3, n3 = signal.butter(2, [0.01, 0.08], 'bandpass')
                e = signal.filtfilt(m3, n3, e)

                newtime_z_start = time_z_start
                newtime_n_start = time_n_start + 50 * num1
                newtime_e_start = time_e_start + 50 * num2
                length = np.min([len(z), len(n), len(e)])
                data = np.zeros([length, 3])
                data[:, 0] = z[:length]
                data[:, 1] = n[:length]
                data[:, 2] = e[:length]
        
    else:
        if time_n_diff<50:
            if time_e_diff<50:  # z>=50, n<50, e<50
                num = time_z_diff // 50
                z = st[0].data[num:]
                n = st[1].data
                e = st[2].data

                m1, n1 = signal.butter(2, [0.01, 0.08], 'bandpass')
                z = signal.filtfilt(m1, n1, z)
                m2, n2 = signal.butter(2, [0.01, 0.08], 'bandpass')
                n = signal.filtfilt(m2, n2, n)
                m3, n3 = signal.butter(2, [0.01, 0.08], 'bandpass')
                e = signal.filtfilt(m3, n3, e)

                newtime_z_start = time_z_start + 50 * num
                newtime_n_start = time_n_start
                newtime_e_start = time_e_start
                length = np.min([len(z), len(n), len(e)])
                data = np.zeros([length, 3])
                data[:, 0] = z[:length]
                data[:, 1] = n[:length]
                data[:, 2] = e[:length]

            else:  # z>=50, n<50, e>=50
                num1 = time_z_diff // 50
                num2 = time_e_diff // 50
                z = st[0].data[num1:]
                n = st[1].data
                e = st[2].data[num2:]

                m1, n1 = signal.butter(2, [0.01, 0.08], 'bandpass')
                z = signal.filtfilt(m1, n1, z)
                m2, n2 = signal.butter(2, [0.01, 0.08], 'bandpass')
                n = signal.filtfilt(m2, n2, n)
                m3, n3 = signal.butter(2, [0.01, 0.08], 'bandpass')
                e = signal.filtfilt(m3, n3, e)

                newtime_z_start = time_z_start + 50 * num1
                newtime_n_start = time_n_start
                newtime_e_start = time_e_start + 50 * num2
                length = np.min([len(z), len(n), len(e)])
                data = np.zeros([length, 3])
                data[:, 0] = z[:length]
                data[:, 1] = n[:length]
                data[:, 2] = e[:length]

        else:
            if time_e_diff<50:  # z>=50, n>=50, e<50
                num1 = time_z_diff // 50
                num2 = time_n_diff // 50
                z = st[0].data[num1:]
                n = st[1].data[num2:]
                e = st[2].data

                m1, n1 = signal.butter(2, [0.01, 0.08], 'bandpass')
                z = signal.filtfilt(m1, n1, z)
                m2, n2 = signal.butter(2, [0.01, 0.08], 'bandpass')
                n = signal.filtfilt(m2, n2, n)
                m3, n3 = signal.butter(2, [0.01, 0.08], 'bandpass')
                e = signal.filtfilt(m3, n3, e)

                newtime_z_start = time_z_start + 50 * num1
                newtime_n_start = time_n_start + 50 * num2
                newtime_e_start = time_e_start
                length = np.min([len(z), len(n), len(e)])
                data = np.zeros([length, 3])
                data[:, 0] = z[:length]
                data[:, 1] = n[:length]
                data[:, 2] = e[:length]

            else:  # z>=50, n>=50, e>=50
                num1 = time_z_diff // 50
                num2 = time_n_diff // 50
                num3 = time_e_diff // 50
                z = st[0].data[num1:]
                n = st[1].data[num2:]
                e = st[2].data[num3:]

                m1, n1 = signal.butter(2, [0.01, 0.08], 'bandpass')
                z = signal.filtfilt(m1, n1, z)
                m2, n2 = signal.butter(2, [0.01, 0.08], 'bandpass')
                n = signal.filtfilt(m2, n2, n)
                m3, n3 = signal.butter(2, [0.01, 0.08], 'bandpass')
                e = signal.filtfilt(m3, n3, e)

                newtime_z_start = time_z_start + 50 * num1
                newtime_n_start = time_n_start + 50 * num2
                newtime_e_start = time_e_start + 50 * num3
                length = np.min([len(z), len(n), len(e)])
                data = np.zeros([length, 3])
                data[:, 0] = z[:length]
                data[:, 1] = n[:length]
                data[:, 2] = e[:length]
    
    # newtime_z_end = newtime_z_start + (length - 1) * 50
    # newtime_n_end = newtime_n_start + (length - 1) * 50
    # newtime_e_end = newtime_e_start + (length - 1) * 50

    # new_z_start = timee(newtime_z_start)
    # new_n_start = timee(newtime_n_start)
    # new_e_start = timee(newtime_e_start)

    # new_z_end = timee(newtime_z_end)
    # new_n_end = timee(newtime_n_end)
    # new_e_end = timee(newtime_e_end)

    return data, newtime_z_start, newtime_n_start, newtime_e_start