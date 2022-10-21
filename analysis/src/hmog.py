from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy.signal import medfilt

base_dir = Path("C:\\devtools\\analysis\\hmog_dataset")

################## Prepare data ##########################


ACTIVITY_COLS = [
    "activity_id",
    "subject_id",
    "session_number",
    "start_time",
    "end_time",
    "relative_start_time",
    "relative_end_time",
    "gesture_scenario",
    "task_id",
    "content_id",
]

READING_SITTING = [1, 7, 13, 19]
READING_WALKING = [2, 8, 14, 20]
WRITING_SITTING = [3, 9, 15, 21]
WRITING_WALKING = [4, 10, 16, 22]
MAP_SITTING = [5, 11, 17, 23]
MAP_WALKING = [6, 12, 18, 24]

SENSOR_COLS = [
    "systime",
    "event_time",
    "activity_id",
    "x",
    "y",
    "z",
    "orientation", # 0 portrait; 1 90 deg counter-clock; 3 90 deg clock
]

TOUCHEVENT_COLS = [
    "systime",
    "event_time",
    "activity_id",
    "pointer_count",
    "pointer_id",
    "action_type",
    "x", "y",
    "pressure", "size",
    "orientation"
]

def join_data_with_activity(csv_filename, cols, out_filename, activity_name, activity_ids):
    user_dirs = [ x for x in base_dir.iterdir() if x.is_dir() ]

    for u_idx, user_dir in enumerate(user_dirs):
        print(f"{u_idx}/{len(user_dirs)}; {user_dir.name}", end='\r')
        
        ws_dir = user_dir / str(activity_name)
        ws_dir.mkdir(parents=True, exist_ok=True)

        user_data = pd.DataFrame()
        session_cnt = 0    

        for session_dir in sorted(user_dir.iterdir()):   
            if (not session_dir.is_dir() or not session_dir.name[0].isdigit()):
                continue

            df_act = pd.read_csv(
                        session_dir / "activity.csv",
                        names=ACTIVITY_COLS,
                        header=None, engine='pyarrow')

            if (df_act[df_act['task_id'].isin(activity_ids)].empty):
                continue

            try:
                session_data = pd.read_csv(
                                session_dir / csv_filename,
                                names = cols,
                                header=None, engine='pyarrow')
            except:
                if (csv_filename != "TouchEvent_im.csv"):
                    continue
                
                else:
                    session_data = pd.read_csv(
                                session_dir / "TouchEvent.csv",
                                names = cols,
                                header=None, engine='pyarrow')
                    
            
            session_data['session_seq'] = session_cnt
            session_cnt += 1
            session_data = session_data.merge(df_act, on='activity_id')
            user_data    = pd.concat([user_data, session_data])

        if (csv_filename == "TouchEvent_im.csv"): # handle naming missmatch
            user_data['action_type'] = user_data['action_type'].replace({5: 0, 6: 1})
        
        
        user_data.reset_index().to_feather(ws_dir / out_filename)


def preprocess_data(activity_name):
    activity_ids = []

    if (activity_name == "writing_sitting"):
        activity_ids = WRITING_SITTING
    elif (activity_name == "writing_walking"):
        activity_ids = WRITING_WALKING
    elif (activity_name == "reading_sitting"):
        activity_ids = READING_SITTING
    elif (activity_name == "reading_walking"):
        activity_ids = READING_WALKING
    elif (activity_name == "map_sitting"):
        activity_ids = MAP_SITTING
    elif (activity_name == "map_walking"):
        activity_ids = MAP_WALKING

    inlist = [
        ("Accelerometer.csv", SENSOR_COLS, "accelerometer.feather"),
        ("Gyroscope.csv", SENSOR_COLS, "gyroscope.feather"),
        ("Magnetometer.csv", SENSOR_COLS, "magnetometer.feather"),
        ("TouchEvent_im.csv",TOUCHEVENT_COLS, "touch.feather")
    ]

    for in_file, col_names, out_file in inlist:
        join_data_with_activity(in_file, col_names, out_file, activity_name, activity_ids)
        print(f"{activity_name} / {out_file}", end='\r')
    


##########################################################

def dump(to_dmp, out):
    with open( "hmog_" + str(out) + '.pkl', 'wb') as f:
        pickle.dump(to_dmp, f)
        
def retrieve(name):
    with open( "hmog_" + str(name) + '.pkl', 'rb') as f:
        return pickle.load(f)


def get_sensor(user_dir, sensor, activity_type="writing_sitting"):
    COLUMNS = ["activity_id", 'subject_id', 'session_seq', 'session_number', 'content_id',
           "systime", "event_time", "relative_start_time",
           'x', 'y', 'z', 'orientation' ]
    
    filename = sensor + ".feather"
    df = pd.read_feather( user_dir / activity_type / filename, columns=COLUMNS )
        
    df['relative_event_time'] = df['event_time'] - df['relative_start_time']*int(1e6)
    df = df[df['relative_event_time'] > 0]
       
    df['m'] = np.linalg.norm(df[['x', 'y', 'z']], axis=1)
    
    return df.dropna()


def get_accelerometer(user_dir, activity_type):
    df = get_sensor(user_dir, "accelerometer", activity_type)
        
    return df


def get_all_accelerometer(activity_type):
    user_dirs = [ x for x in base_dir.iterdir() if x.is_dir() ]

    acc = []

    for user_dir in user_dirs:
        acc.append(get_accelerometer(user_dir, activity_type))

    return acc


def get_gyroscope(user_dir, activity_type):
    df = get_sensor(user_dir, "gyroscope", activity_type)
        
    return df

def get_all_gyroscope(activity_type):
    user_dirs = [ x for x in base_dir.iterdir() if x.is_dir() ]

    gyr = []

    for user_dir in user_dirs:
        gyr.append(get_gyroscope(user_dir, activity_type))
        
    return gyr

def get_magnetometer(user_dir, activity_type):
    df = get_sensor(user_dir, "magnetometer", activity_type)
        
    return df

def get_all_magnetometer(activity_type):
    user_dirs = [ x for x in base_dir.iterdir() if x.is_dir() ]

    mags = []

    for user_dir in user_dirs:
        mags.append(get_magnetometer(user_dir, activity_type))
        
    return mags


def get_all_touches(activity_type):
    COLUMNS = ['systime', 'event_time', 'activity_id', 'pointer_count',
       'pointer_id', 'action_type', 'x', 'y', 'pressure', 'size',
       'orientation', 'session_seq', 'subject_id', 'session_number',
       'start_time', 'end_time', 'relative_start_time', 'relative_end_time',
       'content_id']

    user_dirs = [ x for x in base_dir.iterdir() if x.is_dir() ]

    touches = []

    for user_dir in user_dirs:
        df = pd.read_feather( user_dir / activity_type / "touch.feather", columns=COLUMNS )
        touches.append( df )

    return touches


def plot_sensor(df, session_seq, content_id, ax=None, denoise=None):
    df = df[(df['session_seq'] == session_seq) & (df['content_id'] == content_id)].copy()

    if (denoise):
        df['x'] = medfilt(df[['x']].values.reshape(-1), 51)
        df['y'] = medfilt(df[['y']].values.reshape(-1), 51)
        df['z'] = medfilt(df[['z']].values.reshape(-1), 51)
        
    df['m'] = np.linalg.norm(df[['x', 'y', 'z']], axis=1)
    df['relative_event_time [s]'] = df['relative_event_time'] / 1e9

    return df[['x','y','z', 'm', 'relative_event_time [s]']].plot(x='relative_event_time [s]', ax=ax)












import math

def energy(series):
    return np.sum(np.power(np.abs(series), 2)) / len(series)

from antropy import spectral_entropy
def entropy(series):
    hz = 100
    
    entropy = spectral_entropy(np.abs(series), hz, normalize=True)
    
    if (np.isnan(entropy)):
        return 0
    else:
        return entropy 

def custom_entropy(series):
    n = len(series)
    spec = np.fft.fft(series)
    psd = ( np.power(spec, 2) / n )
    scaled_psd = psd / np.sum(psd)
    return np.abs(-1*np.sum(scaled_psd*np.log2(scaled_psd)))

def get_all_touch_features(activity_type):
    user_dirs = [ x for x in base_dir.iterdir() if x.is_dir() ]

    touch_features = []

    for u_idx, user_dir in enumerate(user_dirs):
        print(f"{u_idx}; {user_dir.name}", end='\r')

        df = aggregate_touch_groups_lite( get_touch_groups(user_dir, activity_type) )
        touch_features.append( df )


    return touch_features

def get_touch_groups(user_dir, activity_type):

    IN_COLUMNS = ["activity_id", 'subject_id', 'session_seq', 'session_number', 'content_id',
                  "systime", "event_time", "relative_start_time",
                  "pointer_count", "pointer_id", "action_type", "x", "y", "pressure", "size", "orientation" ]

    df = pd.read_feather( user_dir / activity_type / "touch.feather", columns=IN_COLUMNS )
    df = df[df['pointer_id'] == 0] #filter out multitouch
    # df = df[df['orientation'] == 0]
    
    last_activity = df['activity_id'].min()

    touch_groups_list = []
    touch_group = []

    for idx, row in df.iterrows():

        if (row['activity_id'] != last_activity and len(touch_group) > 0):
            print(f"Missing UP in the acitivty {row['activity_id']}")
            touch_group = []
            
        if (row['action_type'] == 0): # down / BEGIN
            if (len(touch_group) > 0): # touch down, but group not empty
                touch_groups_list.append(touch_group)
                
            touch_group = []
            touch_group.append(row)

        elif (row['action_type'] == 1): # up / ENDED
            touch_group.append(row)
            touch_groups_list.append(touch_group)
            touch_group = []
            
        else: # move
            touch_group.append(row)

        last_activity = row['activity_id']
        
    return touch_groups_list

import math
from pingouin import circ_mean, circ_r

def aggregate_touch_groups_lite(touch_groups_list):

    touch_data_list = []

    for idx, touch_group in enumerate(touch_groups_list):
        agg_touch = {}

        agg_touch['duration'] = touch_group[-1]['event_time'] - touch_group[0]['event_time']
        agg_touch['start_x'] = touch_group[0]["x"]
        agg_touch['start_y'] = touch_group[0]["y"]
        agg_touch['stop_x'] = touch_group[-1]["x"]
        agg_touch['stop_y'] = touch_group[-1]["y"]

        if (idx != 0):
            flight_time = touch_group[0]['event_time'] - touch_groups_list[idx-1][-1]['event_time']
            if (flight_time > 2000):
                agg_touch['flight_time'] = 0
            else:
                agg_touch['flight_time'] = flight_time
        else:
            agg_touch['flight_time'] = 0

        pressure = []
        area = []
        for touch in touch_group:
            pressure.append(touch["pressure"])
            area.append(touch["size"])

        agg_touch['mid_pressure'] = np.median( pressure[ math.floor(len(pressure) / 2) - 1 : math.ceil(len(pressure) / 2) ] )
        agg_touch['mid_area'] = np.median( area[math.floor(len(area) / 2) - 1 : math.ceil(len(area) / 2) ] )

        agg_touch['activity_id'] = touch_group[0]['activity_id']
        agg_touch['subject_id'] = touch_group[0]['subject_id']
        agg_touch['session_seq'] = touch_group[0]['session_seq']
        agg_touch['session_number'] = touch_group[0]['session_number']
        agg_touch['content_id'] = touch_group[0]['content_id']
        agg_touch['relative_start_time'] = touch_group[0]['relative_start_time']
        agg_touch['systime_start'] = touch_group[0]['systime']
        agg_touch['systime_end'] = touch_group[-1]['systime']
        agg_touch['event_time_start'] = touch_group[0]['event_time']
        agg_touch['event_time_end'] = touch_group[-1]['event_time']
        agg_touch['relative_event_time_start'] = (touch_group[0]['event_time'] - touch_group[0]['relative_start_time'])*1e6
        agg_touch['relative_event_time_end'] = (touch_group[-1]['event_time'] - touch_group[0]['relative_start_time'])*1e6
        agg_touch['pointer_count'] = touch_group[0]['pointer_count']
        agg_touch['pointer_id'] = touch_group[0]['pointer_id']
        agg_touch['orientation'] = touch_group[0]['orientation']


        touch_data_list.append(agg_touch)
        
    return pd.DataFrame(touch_data_list)




def aggregate_touch_groups(touch_groups_list):
    touch_data_list = []

    for idx, touch_group in enumerate(touch_groups_list):
        agg_touch = {}

        agg_touch['duration'] = touch_group[-1]['event_time'] - touch_group[0]['event_time']
        agg_touch['start_x'] = touch_group[0]["x"]
        agg_touch['start_y'] = touch_group[0]["y"]
        agg_touch['stop_x'] = touch_group[-1]["x"]
        agg_touch['stop_y'] = touch_group[-1]["y"]
        agg_touch['line_distance'] = math.sqrt((agg_touch['start_x'] - agg_touch['stop_x']) ** 2 + (agg_touch['start_y'] - agg_touch['stop_y']) ** 2)
        agg_touch['line_direction'] = math.atan2(agg_touch['stop_y'] - agg_touch['start_y'], agg_touch['stop_x'] - agg_touch['start_x'])

        if (idx != 0):
            agg_touch['flight_time'] = touch_group[0]['event_time'] - touch_groups_list[idx-1][-1]['event_time']
        else:
            agg_touch['flight_time'] = 0


        xd = []
        yd = []
        td = []
        prev = touch_group[0]

        for i, curr in enumerate(touch_group):
            xd.append(curr["x"] - prev["x"])
            yd.append(curr["y"] - prev["y"])
            td.append(curr["event_time"] - prev["event_time"])
            prev = curr

        angl = []
        v = []
        pairwised = []

        for i in range(len(xd)):
            angl.append(math.atan2(yd[i], xd[i]))
            pairwised.append(math.sqrt(xd[i]**2 + yd[i]**2))
            if td[i] == 0:
                v.append(0)
            else:
                v.append(math.sqrt(xd[i] ** 2 + yd[i] ** 2) / td[i])

        agg_touch['v_p20'] = np.percentile(v, .20)
        agg_touch['v_p50'] = np.percentile(v, .50)
        agg_touch['v_p80'] = np.percentile(v, .80)
        agg_touch['mean_resultant_len'] = circ_r(np.array(angl))
        agg_touch['avg_direction'] = circ_mean(angl)
        agg_touch['median_v_last3'] = np.median(v[-3:])
        agg_touch['traj_len'] = sum(pairwised)
        if (agg_touch['duration'] == 0):
            agg_touch['avg_v'] = 0
        else:
            agg_touch['avg_v'] = agg_touch['traj_len'] / agg_touch['duration']

        if (agg_touch['traj_len'] == 0):
            agg_touch['line_traj_ratio'] = 1
        else:
            agg_touch['line_traj_ratio'] = agg_touch['line_distance'] / agg_touch['traj_len']

        a = []
        prev = v[0]
        for i, curr in enumerate(v):
            if td[i] == 0:
                a.append(0.0)
            else:
                a.append((curr - prev) / td[i])

            prev = curr

        agg_touch['a_p20'] = np.percentile(a, .20)
        agg_touch['a_p50'] = np.percentile(a, .50)
        agg_touch['a_p80'] = np.percentile(a, .80)
        agg_touch['median_a_first5'] = np.median(a[:5])


        pressure = []
        area = []
        for touch in touch_group:
            pressure.append(touch["pressure"])
            area.append(touch["size"])

        agg_touch['mid_pressure'] = np.median( pressure[ math.floor(len(pressure) / 2) - 1 : math.ceil(len(pressure) / 2) ] )
        agg_touch['mid_area'] = np.median( area[math.floor(len(area) / 2) - 1 : math.ceil(len(area) / 2) ] )




        ################################18;19;20;21 -- deviation from end to end line
        
        vecs = []
        for touch in touch_group:
            vecs.append(
                (touch["x"] - touch_group[0]["x"], touch["y"] - touch_group[0]["y"])
            )
        
        perpendicular = (-vecs[-1][1], vecs[-1][0])

        norm = math.sqrt(perpendicular[0] ** 2 + perpendicular[1] ** 2)

        if (norm == 0):
            agg_touch['max_line_dev'] = 0
            agg_touch['line_dev_p20'] = 0
            agg_touch['line_dev_p50'] = 0
            agg_touch['line_dev_p80'] = 0
        
        else:
            perpendicular = (perpendicular[0] / norm, perpendicular[1] / norm)

            projection = []
            for i in range(len(vecs)):
                projection.append(vecs[i][0] * perpendicular[0] + vecs[i][1] * perpendicular[1])

            max_dev = 0
            for i in projection:
                if abs(i) > abs(max_dev):
                    max_dev = i
            
            agg_touch['max_line_dev'] = max_dev
            agg_touch['line_dev_p20'] = np.percentile(projection, .20)
            agg_touch['line_dev_p50'] = np.percentile(projection, .50)
            agg_touch['line_dev_p80'] = np.percentile(projection, .80)
        ########################################

        midX = []
        midY = []

        for touch in touch_group:
            midX.append(touch["x"])
            midY.append(touch["y"])

        if (np.std(midX) < 5 and np.std(midY) < 5) or len(touch_group) < 3:
            agg_touch['guess'] = 'tap'
        else:
            agg_touch['guess'] = 'move'



        agg_touch['activity_id'] = touch_group[0]['activity_id']
        agg_touch['subject_id'] = touch_group[0]['subject_id']
        agg_touch['session_seq'] = touch_group[0]['session_seq']
        agg_touch['session_number'] = touch_group[0]['session_number']
        agg_touch['content_id'] = touch_group[0]['content_id']
        agg_touch['relative_start_time'] = touch_group[0]['relative_start_time']
        agg_touch['systime_start'] = touch_group[0]['systime']
        agg_touch['systime_end'] = touch_group[-1]['systime']
        agg_touch['event_time_start'] = touch_group[0]['event_time']
        agg_touch['event_time_end'] = touch_group[-1]['event_time']
        agg_touch['relative_event_time_start'] = (touch_group[0]['event_time'] - touch_group[0]['relative_start_time'])*1e6
        agg_touch['relative_event_time_end'] = (touch_group[-1]['event_time'] - touch_group[0]['relative_start_time'])*1e6
        agg_touch['pointer_count'] = touch_group[0]['pointer_count']
        agg_touch['pointer_id'] = touch_group[0]['pointer_id']
        agg_touch['orientation'] = touch_group[0]['orientation']


        touch_data_list.append(agg_touch)
        
    return pd.DataFrame(touch_data_list)


from scipy.stats import kurtosis, iqr, skew

def acc_stats(series, prefix):
    d = {}
    d[(prefix, 'mean')] = np.mean(series)
    d[(prefix, 'max')] = np.max(series)
    d[(prefix, 'min')] = np.min(series)
    d[(prefix, 'var')] = np.var(series)
    d[(prefix, 'std')] = np.std(series)
    d[(prefix, 'ran')] = d[(prefix, 'max')] - d[(prefix, 'min')]
    d[(prefix, 'kurt')] = kurtosis(series)
    d[(prefix, 'skew')] = skew(series)
    d[(prefix, 'iqr')] = iqr(series)
    d[(prefix, 'q25')] = np.quantile(series, 0.25)
    d[(prefix, 'q50')] = np.quantile(series, 0.50)
    d[(prefix, 'q75')] = np.quantile(series, 0.75)

    return d

def aggregate_accelerometer_by_touch(touch_data, accelerometer_data, eps_ms = 0):
            
    acc = accelerometer_data.copy()
    acc['relative_event_time'] = acc['relative_event_time']/1e6
    acc = acc.set_index(['activity_id', 'relative_event_time'])
        
    data = []
    
        
    for idx, row in touch_data.iterrows():        
        touch = {}

        before = acc[(row['activity_id'], row['relative_event_time_start']-100) : 
                    (row['activity_id'], row['relative_event_time_start'])]
        
        batch = acc[(row['activity_id'], row['relative_event_time_start']-eps_ms) : 
                    (row['activity_id'], row['relative_event_time_end']+eps_ms)]

        after = acc[(row['activity_id'], row['relative_event_time_end']) : 
                    (row['activity_id'], row['relative_event_time_end']+100)]

        end = acc[(row['activity_id'], row['relative_event_time_end']) : 
                    (row['activity_id'], row['relative_event_time_end']+200)]
        
        if (len(batch) == 0):
            continue

        touch[('id', 'activity_id')] = row['activity_id']
        touch[('id', 'subject_id')] = row['subject_id']
        touch[('id', 'session_seq')] = row['session_seq']
        touch[('id', 'content_id')] = row['content_id']

        touch[('event', 'orientation')] = row['orientation']
        
        touch[('touch', 'relative_event_time_start')] = row['relative_event_time_start']
        touch[('touch', 'relative_event_time_end')] = row['relative_event_time_end']
        touch[('touch', 'duration')] = row['duration']
        
        touch[('touch', 'start_x')] = row['start_x']
        touch[('touch', 'start_y')] = row['start_y']
        touch[('touch', 'stop_x')] = row['stop_x']
        touch[('touch', 'stop_y')] = row['stop_y']
        touch[('touch', 'line_distance')] = row['line_distance']
        touch[('touch', 'line_direction')] = row['line_direction']
        touch[('touch', 'flight_time')] = row['flight_time']        
        
        touch = touch | acc_stats(batch['x'], 'acc_x')
        touch = touch | acc_stats(batch['y'], 'acc_y')
        touch = touch | acc_stats(batch['z'], 'acc_z')
        touch = touch | acc_stats(batch['m'], 'acc_m')

        touch = touch | acc_stats(batch['x'].diff().fillna(0), 'acc_xdiff')
        touch = touch | acc_stats(batch['y'].diff().fillna(0), 'acc_ydiff')
        touch = touch | acc_stats(batch['z'].diff().fillna(0), 'acc_zdiff')
        touch = touch | acc_stats(batch['m'].diff().fillna(0), 'acc_mdiff')
      
        touch[('acc_m', 'energy')] = energy(np.fft.fft(batch['m']))
        touch[('acc_x', 'energy')] = energy(np.fft.fft(batch['x']))
        touch[('acc_y', 'energy')] = energy(np.fft.fft(batch['y']))
        touch[('acc_z', 'energy')] = energy(np.fft.fft(batch['z']))
    
        touch[('acc_x', 'cust_entropy')] = custom_entropy(batch['x'])
        touch[('acc_y', 'cust_entropy')] = custom_entropy(batch['y'])
        touch[('acc_z', 'cust_entropy')] = custom_entropy(batch['z'])
        touch[('acc_m', 'cust_entropy')] = custom_entropy(batch['m'])
        
        touch[('acc_x', 'entropy')] = entropy(batch['x'])
        touch[('acc_y', 'entropy')] = entropy(batch['y'])
        touch[('acc_z', 'entropy')] = entropy(batch['z'])
        touch[('acc_m', 'entropy')] = entropy(batch['m'])

        ## Grasp Resistance Features
        avg100msBefore = [np.mean(before['x']), np.mean(before['y']), np.mean(before['z']), np.mean(before['m'])]
        avg100msAfter = [np.mean(after['x']), np.mean(after['y']), np.mean(after['z']), np.mean(after['m'])]

        touch[('acc_m', 'before_after')] = np.abs(avg100msAfter[3] - avg100msBefore[3])
        touch[('acc_x', 'before_after')] = np.abs(avg100msAfter[0] - avg100msBefore[0])
        touch[('acc_y', 'before_after')] = np.abs(avg100msAfter[1] - avg100msBefore[1])
        touch[('acc_z', 'before_after')] = np.abs(avg100msAfter[2] - avg100msBefore[2])

        touch[('acc_m', 'net_change')] = np.abs(touch[('acc_m', 'mean')] - avg100msBefore[3])
        touch[('acc_x', 'net_change')] = np.abs(touch[('acc_x', 'mean')] - avg100msBefore[0])
        touch[('acc_y', 'net_change')] = np.abs(touch[('acc_y', 'mean')] - avg100msBefore[1])
        touch[('acc_z', 'net_change')] = np.abs(touch[('acc_z', 'mean')] - avg100msBefore[2])

        touch[('acc_m', 'max_change')] = np.abs(touch[('acc_m', 'max')] - avg100msBefore[3])
        touch[('acc_x', 'max_change')] = np.abs(touch[('acc_x', 'max')] - avg100msBefore[0])
        touch[('acc_y', 'max_change')] = np.abs(touch[('acc_y', 'max')] - avg100msBefore[1])
        touch[('acc_z', 'max_change')] = np.abs(touch[('acc_z', 'max')] - avg100msBefore[2])

        # ## Grasp Stability Features
        # avgDiffs = [[], [], [], []]
        # for idx, row in end.iterrows():



        
        data.append( touch )      

    if (len(data) == 0):
        return pd.DataFrame()  
                
    return pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(data[0].keys()) )






import random

def train_test_positive(df, n_train):
    
    X_train = df.reset_index()[df.reset_index()['session_seq'] < n_train].copy()
    
    test_list = []
    X_test = df.reset_index()[df.reset_index()['session_seq'] >= n_train].copy()
    unique_activity_list = X_test["activity_id"].unique().tolist()
    for act_id in unique_activity_list:
        test_list.append( X_test[(X_test['activity_id'] == act_id)].copy() )
    
#     X_test = df.reset_index()[df.reset_index()['session_seq'] >= n_train].copy()
    
    return X_train, test_list

class Neighbors:
    
    def __init__(self):
        self.mean_vectors = {}
        self.dfs = {}

    def add(self, idx, df):
        self.mean_vectors[idx] = (df.mean(axis=0))
        self.dfs[idx] = df
        
    def get_vectors(self):
        return self.mean_vectors
    
    def get_nearest(self, k, a):
        d = {}
        
        mean_vector_a = a.mean(axis=0)
        
        for b in self.mean_vectors:
            d[b] = (np.linalg.norm(mean_vector_a - self.mean_vectors[b]))
            
        d = {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}
        ret_idx = []
        for i in d:
            ret_idx.append(i)
            
        ret = []
        for i in ret_idx[:k]:
            ret.append( self.dfs[i] )
            
        return ret


def train_test_negative(user_list, user_idx, n_train):
    
    idx_permutation = random.sample(range(0, len(user_list)), len(user_list)) # get idx permutation
    g1 = idx_permutation[:len(idx_permutation)//2]
    g2 = idx_permutation[len(idx_permutation)//2:]
    
    ## Train neg.

    # neig = Neighbors()
    # for idx in g1:
    #     if (idx == user_idx):
    #         continue

    #     tmp_df = user_list[idx].dropna().reset_index()
    #     activity_id = random.sample(tmp_df["activity_id"].unique().tolist(), 1)[0]
    #     neig.add( idx, tmp_df[(tmp_df["activity_id"] == activity_id)].copy()[features] )


    # train_list = neig.get_nearest(n_train, X_train[features])

    train_list = []
    for idx in random.sample(g1, n_train):
        if (idx == user_idx):
            continue
            
        tmp_df = user_list[idx].dropna().reset_index()
        activity_id = random.sample(tmp_df["activity_id"].unique().tolist(), 1)[0]
        train_list.append( tmp_df[(tmp_df["activity_id"] == activity_id)].copy() ) 
    
#     X_train = pd.concat(train_list)
    
    ## Test pos.
    test_list = []
    for idx in g2: 
        if (idx == user_idx):
            continue
            
        tmp_df = user_list[idx].dropna().reset_index()
        activity_id = random.sample(tmp_df["activity_id"].unique().tolist(), 1)[0]
        test_list.append( tmp_df[(tmp_df["activity_id"] == activity_id)].copy() )
    
#     X_test = pd.concat(test_list)
    
    return train_list, test_list
    







from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA

def evaluate(df, user_idx, n_train, features):
    
    X_train, X_test_list = train_test_positive( df[user_idx], n_train )
    X_train_out, X_test_out = train_test_negative( df, user_idx, len(X_test_list) )
    
    X_train = X_train[features].dropna()
    
    scaler = MinMaxScaler().fit(X_train.values)
    X_train = scaler.transform(X_train.values)
    
#     pca = PCA(n_components=8).fit(X_train)
#     X_train = pca.transform(X_train)

    
    res = []
    # for p1 in [0.001, 0.01, 0.1, 0.3, 0.5, 0.75, 1.]:
    #     for p2 in [0.001, 0.01, 0.1, 0.3, 0.5, 0.75, 1.]:
    for p1 in [1, 5, 10, 50, 100, 200, 300]:
        for p2 in [0.01, 0.1, 0.2, 0.3, 0.4]:

            tmp = {
                "p1": p1,
                "p2": p2,
            }


    #         clf = OneClassSVM(nu=p1, gamma=p2)
            clf = LocalOutlierFactor(novelty=True, n_neighbors=p1, contamination=p2)
    #         clf = IsolationForest(n_estimators=p1, contamination=p2)
            clf.fit( X_train )
        
            y_pred = clf.predict( X_train )
            tmp['train_error'] = np.mean( y_pred[y_pred == -1].size / y_pred.size )

            inliers = []
            y_pred_pos = []
            for X_test in X_test_list:
                X_test = X_test[features].dropna()
                X_test = scaler.transform(X_test.values)
#                 X_test = pca.transform(X_test)

                y_pred = clf.predict( X_test )
                y_pred_pos.extend(y_pred)

                inliers.append( y_pred[y_pred == -1].size / y_pred.size )


            tmp['frr'] = np.mean(inliers)
            tmp['frr_std'] = np.std(inliers)
            tmp['frr_gini'] = gini(inliers)
            tmp['frr_list'] = inliers


            outliers = []
            y_pred_neg = []
            for X_outlier in X_test_out:
                X_outlier = X_outlier[features].dropna()
                X_outlier = scaler.transform(X_outlier.values)
#                 X_outlier = pca.transform(X_outlier)

                y_pred = clf.predict( X_outlier )
                y_pred_neg.extend(y_pred)

                outliers.append(y_pred[y_pred == 1].size / y_pred.size)

            tmp['far'] = np.mean(outliers)
            tmp['far_std'] = np.std(outliers)
            tmp['far_gini'] = gini(outliers)
            tmp['far_list'] = outliers

            tmp['har'] = (tmp['frr'] + tmp['far'])/2

            tmp['eer'] = calculate_eer(y_test=np.concatenate([ [1]*len(y_pred_pos), [-1]*len(y_pred_neg) ]), y_pred=np.concatenate([y_pred_pos, y_pred_neg]))

            res.append(tmp)


    return res

def plot_result(res):

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(16, 5), sharey=True)

    eer_sorted = pd.DataFrame(res).sort_values('eer').head(1)
    har_sorted = pd.DataFrame(res).sort_values('har').head(1)

    print(f"1st row: EER: {eer_sorted.eer.values}; HAR: {eer_sorted.har.values}; TrainError {eer_sorted.train_error.values}")
    print(f"Parameters {eer_sorted.p1.values}, {eer_sorted.p2.values}")
    print(f"2nd row: EER: {har_sorted.eer.values}; HAR: {har_sorted.har.values}; TrainError {har_sorted.train_error.values}")
    print(f"Parameters {har_sorted.p1.values}, {har_sorted.p2.values}")


    axes[0][0].set_title(f"FAR: {eer_sorted.far.values} ({eer_sorted.far_std.values}); G: {eer_sorted.far_gini.values}")
    l = eer_sorted.far_list.values[0]
    axes[0][0].bar(x=np.arange(len(l)), height=l)

    axes[0][1].set_title(f"FRR: {eer_sorted.frr.values} ({eer_sorted.frr_std.values}); G: {eer_sorted.frr_gini.values}")
    l = eer_sorted.frr_list.values[0]
    axes[0][1].bar(x=np.arange(len(l)), height=l)

    axes[1][0].set_title(f"FAR: {har_sorted.far.values} ({har_sorted.far_std.values}); G: {har_sorted.far_gini.values}")
    l = har_sorted.far_list.values[0]
    axes[1][0].bar(x=np.arange(len(l)), height=l)

    axes[1][1].set_title(f"FRR: {har_sorted.frr.values} ({har_sorted.frr_std.values}); G: {har_sorted.frr_gini.values}")
    l = har_sorted.frr_list.values[0]
    axes[1][1].bar(x=np.arange(len(l)), height=l)

    fig.tight_layout(pad=1.0)

def gini(x):
    # (Warning: This is a concise implementation, but it is O(n**2)
    # in time and memory, where n = len(x).  *Don't* pass in huge
    # samples!)

    # Mean absolute difference
    mad = np.abs(np.subtract.outer(x, x)).mean()
    # Relative mean absolute difference
    if (np.mean(x) == 0):
        return 0
    rmad = mad/np.mean(x)
    # Gini coefficient
    g = 0.5 * rmad
    return g


from sklearn.svm import SVC, OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.decomposition import PCA

class NoScaler():
    def fit(self, X):
        return self
    
    def transform(self, X):
        return X

from scipy.spatial.distance import cityblock, mahalanobis, euclidean
from sklearn.metrics import roc_curve

class NoScaler():

    def __init__(self):
        pass

    def fit(self, X):
        return self
    
    def transform(self, X):
        return X

class ManhattanDetector:
    
    def __init__(self):
        self.mean_vector = []
        self.mad_vector = []
        self.ratio = []
        
    def fit(self, X_train):
        self.mean_vector = np.array( X_train.mean(axis=0) )
        self.mad_vector = np.mean(np.absolute(X_train - np.mean(X_train, axis=0)), axis=0)
        self.ratio = (self.mean_vector / self.mad_vector)
        
    def predict(self, X):
        scores = []
        for i in range( len(X) ):
            scores.append( np.sum(np.abs( X[i] - self.ratio ) ))
            
#             dist = cityblock(X[i], self.mean_vector)
#             scores.append(dist)

        return scores

class ZScoreDetector:
    
    def __init__(self):
        self.mean_vector = []
        self.std_vector = []
        
    def fit(self, X_train):
        self.mean_vector = np.array( X_train.mean(axis=0) )
        self.std_vector = np.array( X_train.std(axis=0) )
        
    def predict(self, X):
        scores = []
        for i in range( len(X) ):
            score = np.abs(X[i] - self.mean_vector)/self.std_vector
            scores.append( score[score > 3].size )
            
        return scores
    
class MahalanobisDetector:
    
    def __init__(self):
        self.mean_vector = []
        self.cov_inv = None
        
    def fit(self, X_train):
        self.mean_vector = np.array( X_train.mean(axis=0) )
        self.covinv = np.linalg.inv( np.cov(X_train.T) )
        
    def predict(self, X):
        scores = []
        for i in range( len(X) ):
            dist = ((self.mean_vector - X[i]).T @ self.covinv) @ (self.mean_vector - X[i])
            dist = dist / (np.linalg.norm(self.mean_vector) * np.linalg.norm(X[i]))
            scores.append( dist )

        return scores
    
class EuclideanDetector:
    
    def __init__(self):
        self.mean_vector = []
        
    def fit(self, X_train):
        self.mean_vector = np.array( X_train.mean(axis=0) )
        
    def predict(self, X):
        scores = []
        for i in range( len(X) ):
            dist = np.linalg.norm(X[i] - self.mean_vector)**2
            dist = dist / (np.linalg.norm(self.mean_vector) * np.linalg.norm(X[i]))
            scores.append( dist )

        return scores

from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq
def calculate_eer(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)

    return eer, thresh

######################################


def evaluate_two_class(user_idx, train_sessions, df_list, features, scaler=NoScaler(), clf=SVC()):

    user = df_list[user_idx]

    X_train, X_test_list = train_test_positive(user, train_sessions)
    X_train_neg_list, X_test_neg_list = train_test_negative(df_list, user_idx, train_sessions*3)

    X_train = X_train[features].dropna()
    X_train_out = pd.concat(X_train_neg_list)[features].dropna()

    scaler = scaler.fit(X_train.values)
    X_train = scaler.transform(X_train.values)
    X_train_out = scaler.transform(X_train_out.values)

    X = np.concatenate([ X_train, X_train_out ])
    y = np.array([1] * len(X_train) + [-1] * len(X_train_out))

    clf.fit(X, y)

    y_pred = clf.predict( X_train )
    print("Train FRR", y_pred[y_pred == -1].size / y_pred.size)
    y_pred = clf.predict( X_train_out )
    print("Train FAR", y_pred[y_pred == 1].size / y_pred.size)

    frrs = []
    for X_test_in in X_test_list:
        X_test_in = X_test_in.dropna()[features]
        X_test_in = scaler.transform(X_test_in.values)
    #     X_test_in = pca.transform(X_test_in)
        y_pred = clf.predict( X_test_in )
        frrs.append( y_pred[y_pred == -1].size / y_pred.size )

    fars = []
    for X_test_out in X_test_neg_list:
        X_test_out = X_test_out.dropna()[features]
        X_test_out = scaler.transform(X_test_out.values)
    #     X_test_out = pca.transform(X_test_out)
        y_pred = clf.predict( X_test_out )
        fars.append( y_pred[y_pred == 1].size / y_pred.size )

    return frrs, fars

def evaluate_one_class(user_idx, train_sessions, df_list, features, scaler=NoScaler(), clf=OneClassSVM()):

    user = df_list[user_idx]

    X_train, X_test_list = train_test_positive(user, train_sessions)
    X_train_neg_list, X_test_neg_list = train_test_negative(df_list, user_idx, train_sessions*3)

    X_train = X_train[features].dropna()
    X_train_out = pd.concat(X_train_neg_list)[features].dropna()

    scaler = scaler.fit(X_train.values)
    X_train = scaler.transform(X_train.values)
    X_train_out = scaler.transform(X_train_out.values)

    clf.fit(X_train)

    y_pred = clf.predict( X_train )
    print("Train FRR", y_pred[y_pred == -1].size / y_pred.size)
    y_pred = clf.predict( X_train_out )
    print("Train FAR", y_pred[y_pred == 1].size / y_pred.size)

    frrs = []
    for X_test_in in X_test_list:
        X_test_in = X_test_in.dropna()[features]
        X_test_in = scaler.transform(X_test_in.values)
    #     X_test_in = pca.transform(X_test_in)
        y_pred = clf.predict( X_test_in )
        frrs.append( y_pred[y_pred == -1].size / y_pred.size )

    fars = []
    for X_test_out in X_test_neg_list:
        X_test_out = X_test_out.dropna()[features]
        X_test_out = scaler.transform(X_test_out.values)
    #     X_test_out = pca.transform(X_test_out)
        y_pred = clf.predict( X_test_out )
        fars.append( y_pred[y_pred == 1].size / y_pred.size )

    return frrs, fars

from numpy import inf
def evaluate_detector(user_idx, train_sessions, df_list, features, scaler=NoScaler(), clf=MahalanobisDetector(), thr=-1):

    user = df_list[user_idx]

    X_train, X_test_list = train_test_positive(user, train_sessions)
    X_train_neg_list, X_test_neg_list = train_test_negative(df_list, user_idx, train_sessions*3)

    X_train = X_train[features].dropna()
    X_train_out = pd.concat(X_train_neg_list)[features].dropna()

    scaler = scaler.fit(X_train.values)
    X_train = scaler.transform(X_train.values)
    X_train_out = scaler.transform(X_train_out.values)

    clf.fit(X_train)

    train_in_score = np.array(clf.predict(X_train))
    train_out_score = np.array(clf.predict(X_train_out))

    train_in_score[train_in_score == inf] = 0
    train_out_score[train_out_score == inf] = 0
    

    eer, thr_eer = calculate_eer( [0] * len(train_in_score) + [1] * len(train_out_score) , np.concatenate([train_in_score, train_out_score]) )
    if (thr < 0):
        thr = thr_eer

    elif (thr == 0):
        thr = 1.5*np.mean( train_in_score )

    print("train eer", eer, "thr_eer:", thr_eer, "thr:", thr)

    y_preds_pos = []
    frrs = []
    for X_test_in in X_test_list:
        X_test_in = X_test_in[features].dropna()
        X_test_in = scaler.transform(X_test_in.values)
    #     X_test_in = pca.transform(X_test_in)
        y_pred = np.array( clf.predict( X_test_in ) )
        y_preds_pos.extend( y_pred )
        frrs.append( y_pred[y_pred > thr].size / y_pred.size )

    y_preds_neg = []
    fars = []
    for X_test_out in X_test_neg_list:
        X_test_out = X_test_out[features].dropna()
        X_test_out = scaler.transform(X_test_out.values)
    #     X_test_out = pca.transform(X_test_out)
        y_pred = np.array( clf.predict( X_test_out ) )
        y_preds_neg.extend( y_pred )
        fars.append( y_pred[y_pred < thr].size / y_pred.size )


    y_preds_pos = np.array(y_preds_pos)
    y_preds_neg = np.array(y_preds_neg)
    y_preds_pos[y_preds_pos == inf] = 0
    y_preds_neg[y_preds_neg == inf] = 0
    # print(y_preds_pos[np.isnan(y_preds_pos)])
    # print(y_preds_neg[np.isnan(y_preds_neg)])

    eer, thr_eer = calculate_eer( [0] * len(y_preds_pos) + [1] * len(y_preds_neg) , np.concatenate([y_preds_pos, y_preds_neg]) )
    print("test eer", eer, "thr_eer:", thr_eer, "thr:", thr)

    return frrs, fars