import numpy as np
import pandas as pd



import sqlalchemy

class DBConnector:

    def __init__(self, url) -> None:
        self.url = url
        self.engine = sqlalchemy.create_engine(url)

    def query(self, q)  -> pd.DataFrame():
        return pd.read_sql(q, self.engine)




    # def events_by_session_id(self, event_type, session_ids, activity_type = None):
    #     q = ""
    #     if (activity_type == None):
    #         q = f"select rid from activities where session_rid in {tuple(session_ids)};"
    #     else:
    #         q = f"select rid from activities where session_rid in {tuple(session_ids)} and activity_id = '{activity_type}';"

    #     activity_ids = self.query(q).rid.to_list()
    #     return self.events_by_activity_id(event_type, activity_ids)

    # def events_by_activity_id(self, event_type, activity_ids):
    #     q = f"select * from {event_type} where activity_rid in {tuple(activity_ids)};"
    #     return self.query(q)

    def events_by_session_id(self, event_type, session_ids):
        q = f"""select s.device_id, s.channel, a.session_rid, a.activity_id, e.*
                from {event_type} e, activities a, sessions s
	            where e.activity_rid = a.rid and a.session_rid = s.rid
                and s.rid in {tuple(session_ids)};"""
        return self.query(q)

    def events_by_device_ids(self, event_type, device_ids, activity_id = None):
        q = f"""select s.device_id, s.channel, a.session_rid, a.activity_id, e.*
                    from {event_type} e, activities a, sessions s
                    where e.activity_rid = a.rid and a.session_rid = s.rid
                    and s.device_id in {tuple(device_ids)}"""

        if (activity_id != None):
            q += f"and a.activity_id = '{activity_id}'"
        
        return self.query(q)

    def get_device_ids(self, n_sessions = 2, n_activities = 8, exclude_devices = ('6b21289b9e19f146', '9bc7c8472803b00e')):
        q = f"""select a.session_rid, a.rid, a.activity_id, a.debug, s.channel, s.device_id, a.timestamp
                from activities a, sessions s
                where a.session_rid = s.rid
                and device_id not in {exclude_devices};"""

        stat = self.query(q)

        stat = stat.groupby('device_id').nunique()[['session_rid', 'rid']]\
                   .rename(columns={'session_rid': 'n_sessions', 'rid': 'n_activities'})

        stat = stat[(stat['n_sessions'] >= 2) & (stat['n_activities'] >= stat['n_sessions']*4)]
        return stat.index.values






def get_flight_times_simple(df_in):

    df = df_in[df_in['phase'] != "MOVED"].copy()
    df = df.sort_values('rid').reset_index()

    df.loc[0, 'flight_time'] = 0

    for i in range(2, len(df), 2):

        if (df.loc[i, 'phase'] != "BEGIN"):
            print("ERROR, unexpected ENDED")
            break

        if (df.loc[i, 'activity_rid'] != df.loc[i-1, 'activity_rid']):
            df.loc[i, 'flight_time'] = 0
            continue

        df.loc[i, 'flight_time'] = (df.loc[i, 'timestamp'] - df.loc[i-1, 'timestamp'])/1e6
        
    return df



def get_touch_groups(df_in):
    
    df = df_in.sort_values('rid').reset_index().copy()
    
    touch_groups = []
    touch_group = []
    
    prev_activity = df.loc[0, 'activity_rid']
    
    for i in range(len(df)):
        
        row = df.loc[i]      
        
        # TODO POINTER ID!
        if (row['pointer_count'] > 1):
            print("WARNING: Multitouch could not be recognized correctly.")
               
        if (row['phase'] == 'BEGIN'):
            if (len(touch_group) > 0): # touch down, but group not empty
                touch_groups.append(touch_group)
                
            touch_group = []
            touch_group.append(row)
            
        elif (row['phase'] == 'ENDED'): # touch up
            touch_group.append(row)
            
            touch_groups.append(touch_group)
            touch_group = []
            
        else: # move
            if (row['activity_rid'] != prev_activity):
                print("WARNING: Probably missin ENDED/BEGIN.")
                touch_groups.append(touch_group)
                touch_group = []
                
            touch_group.append(row)
            
            
        prev_activity = row['activity_rid']
    
    return touch_groups


import math
from pingouin import circ_mean, circ_r

def agg_touch_groups(touch_groups_list):
    
    touch_data_list = []
    
    for idx, touch_group in enumerate(touch_groups_list):
    
        agg_touch = {}
        
        agg_touch['device_id'] = touch_group[0]['device_id']
        agg_touch['activity_rid'] = touch_group[0]['activity_rid']
        agg_touch['debug'] = touch_group[0]['debug']
        
        agg_touch['start_ts'] = touch_group[0]['timestamp']
        agg_touch['stop_ts'] = touch_group[-1]['timestamp']
        agg_touch['down_time'] = touch_group[-1]['down_time']
        
#         agg_touch['h'] = touch_group[0]['h']
#         agg_touch['w'] = touch_group[0]['w']
        agg_touch['start_x'] = touch_group[0]["x_raw"] / touch_group[0]['w']
        agg_touch['start_y'] = touch_group[0]["y_raw"] / touch_group[0]['h']
        agg_touch['stop_x'] = touch_group[-1]["x_raw"] / touch_group[-1]['w']
        agg_touch['stop_y'] = touch_group[-1]["y_raw"] / touch_group[-1]['h']
        agg_touch['start_x_raw'] = touch_group[0]["x_raw"]
        agg_touch['start_y_raw'] = touch_group[0]["y_raw"]
        agg_touch['stop_x_raw'] = touch_group[-1]["x_raw"]
        agg_touch['stop_y_raw'] = touch_group[-1]["y_raw"]

        agg_touch['line_distance'] = math.sqrt((agg_touch['start_x'] - agg_touch['stop_x']) ** 2 + (agg_touch['start_y'] - agg_touch['stop_y']) ** 2)
        agg_touch['line_direction'] = math.atan2(agg_touch['stop_y'] - agg_touch['start_y'], agg_touch['stop_x'] - agg_touch['start_x'])
        
        area = []
        for touch in touch_group:
            area.append(touch["size"])

        agg_touch['mid_area'] = np.median( area[math.floor(len(area) / 2) - 1 : math.ceil(len(area) / 2) ] )
        agg_touch['max_area'] = np.max(area)
        agg_touch['min_area'] = np.min(area)
        agg_touch['mean_area'] = np.mean(area)


        xd = []
        yd = []
        td = []
        prev = touch_group[0]

        for i, curr in enumerate(touch_group):
            xd.append(curr["x_raw"] - prev["x_raw"])
            yd.append(curr["y_raw"] - prev["y_raw"])
            td.append(curr["down_time"])
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
        if (agg_touch['down_time'] == 0):
            agg_touch['avg_v'] = 0
        else:
            agg_touch['avg_v'] = agg_touch['traj_len'] / agg_touch['down_time']

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


        agg_touch['n'] = len(touch_group)
        
        
        touch_data_list.append(agg_touch)
        
    return touch_data_list
        
        
            
def discrim_multidim(df, cols, user_col, k):
    cpy = df.reset_index().copy()
    users = cpy[user_col].unique()
    
    res = []
    for i, u in enumerate(users):
        own_vals = cpy[cpy[user_col] == u][cols].values
        if (2*own_vals.size < k):
#             print(f"Not enough data ({own_vals.size}) for user {u}")
            continue
            
        foreign_vals = cpy[cpy[user_col] != u][cols].values
        
        tmp = {
            'user': u,
            'own_dist': own_dist_multidim(own_vals, k),
            'foreign_dist': foreign_dist_multidim(own_vals, foreign_vals, k)
        } 
        res.append(tmp)
        
    return res 


def own_dist_multidim(arr, k):
    tmp = np.array([arr[np.random.choice(arr.shape[0], 2, replace=False), :] for _ in range(k)])
    return np.linalg.norm(tmp[:,0]-tmp[:,1], axis=1).mean()

def foreign_dist_multidim(arr0, arr1, k):
    return np.linalg.norm(arr0[np.random.choice(arr0.shape[0], k, replace=True), :] - arr1[np.random.choice(arr1.shape[0], k, replace=True), :], axis=1).mean()


def discrim(df, col, user_col, k):
    cpy = df.reset_index().copy()
    users = cpy[user_col].unique()
#     print(users)
    
    res = []
#     own_dists = {}
#     foreign_dists = {}
    
    for i, u in enumerate(users):
        own_vals = cpy[cpy[user_col] == u][col].values
        foreign_vals = cpy[cpy[user_col] != u][col].values
        if (2*own_vals.size < k):
#             print(f"Not enough data ({own_vals.size}) for user {u}")
            continue
        
        tmp = {
            'user': u,
            'own_dist': own_dist(own_vals, k),
            'foreign_dist': foreign_dist(own_vals, foreign_vals, k)
        } 
        res.append(tmp)
        
#         own_dists.append( own_dist(own_vals, k) )
#         foreign_dists.append( foreign_dist(own_vals, foreign_vals, k) )
        
    return res 

def discrim_single(df, u, col, k):
    cpy = df.reset_index().copy()
    users = cpy.user.unique() 
    
    own_vals = cpy[cpy['user'] == u][col].values
    foreign_vals = cpy[cpy['user'] != u][col].values
    
    return {
        'user': u,
        'own_dist': own_dist(own_vals, k),
        'foreign_dist': foreign_dist(own_vals, foreign_vals, k)
    }
        

def own_dist(arr, k):
    tmp = np.array([np.random.choice(arr, 2, replace=False) for _ in range(k)])
    return np.abs(tmp[:,0] - tmp[:,1]).mean()

def foreign_dist(arr0, arr1, k):
    return np.abs(np.random.choice(arr0, k, replace=True) - np.random.choice(arr1, k, replace=True)).mean()

def abs_distances(own, foreign):
    return (np.array(foreign) - np.array(own)).mean()
    
def rel_distances(own, foreign):
    return ((np.array(foreign) - np.array(own))/np.array(own)).mean()


class NoScaler():

    def __init__(self):
        return None

    def fit(self, df):
        return self

    def transform(self, df):
        return df



#################### CUSTOM DATA REPRESENTATION

import warnings

class TouchAction:
    ''' Represents Touch Action from phase BEGIN to ENDED '''

    def __init__(self, df: pd.DataFrame):
        self.touch_df = None
        self.begin = 0
        self.end = 0
        self.debug = None
        self.sensors = {
            "accelerometer_events": pd.DataFrame(),
            "gyroscope_events": pd.DataFrame(),
            "linear_acceleration_events": pd.DataFrame(),
            "gravity_events": pd.DataFrame(),
            "magnetometer_events": pd.DataFrame(),
            "rotation_vector_events": pd.DataFrame()
        }

        self.sensors_real_sample_rate = {}
        self.sensors_expected_sample_rate = {}
        for sensor_name in self.sensors.keys():
            self.sensors_real_sample_rate[sensor_name] = 0
            self.sensors_expected_sample_rate[sensor_name] = 0

        self.add_touch_df(df)        

    def add_touch_df(self, df: pd.DataFrame ):
        ''' Takes DataFrame of a SINGLE touch action from BEGIN to ENDED. '''
        if (df.empty):
            warnings.warn(f"[Warning] Trying to add empty touch df for TouchAction.")
            return

        self.touch_df = df.sort_values('timestamp').copy()
        self.begin = df['timestamp'].min()
        self.end = df['timestamp'].max()

        if ( (df['debug'].unique().size > 1) and ('layer' not in df['debug'].unique()) ):
            warnings.warn(f"[Warning] Multiple debug for single TouchAction: {df.iloc[0].activity_rid}")

        self.debug = df.iloc[0].debug

    def add_sensor_df(self, df: pd.DataFrame, sensor_type: str, expected_sample_rate, eps_ms = 0):
        ''' Takes DataFrame of sensor events and copies rows according to assigned touch action time interval '''
        if (df.empty):
            warnings.warn(f"[Warning] Trying to add empty sensor df for TouchAction.")
            return

        if (sensor_type not in self.sensors.keys()):
            warnings.warn(f"[Warning] Unexpected sensor_type name {sensor_type}. Expected: {self.sensors.keys()}")

        eps = eps_ms*1e6
        # print(self.begin - eps, self.begin, self.end, self.end + eps_ms  )
        self.sensors[sensor_type] = df[(df['timestamp'] >= (self.begin - eps)) & (df['timestamp'] <= (self.end + eps))].copy()
        self.sensors[sensor_type]['m'] = np.linalg.norm(self.sensors[sensor_type][['x', 'y', 'z']], axis=1)

        # print(self.begin, self.end)
        if (self.begin == self.end):
            self.sensors_real_sample_rate[sensor_type] = 0
            return

        self.sensors_real_sample_rate[sensor_type] = len(self.sensors[sensor_type]) / (((self.end + eps) - (self.begin - eps))/1e9)
        self.sensors_expected_sample_rate[sensor_type] = expected_sample_rate

    def __lt__(self, other):
         return self.begin < other.begin

class Activity:

    def __init__(self, rid, keep_df = False):
        self.keep_df = keep_df
        
        self.touch_actions = []

        # if (keep_df == True):
        self.touch_df = pd.DataFrame(),
        self.sensors_df = {
            "accelerometer_events": pd.DataFrame(),
            "gyroscope_events": pd.DataFrame(),
            "linear_acceleration_events": pd.DataFrame(),
            "gravity_events": pd.DataFrame(),
            "magnetometer_events": pd.DataFrame(),
            "rotation_vector_events": pd.DataFrame()
        }

        ## Details
        self.rid = rid
        self.activity_name = None
        self.debug = None
        self.session_rid = None
        self.device_id = None
        self.timestamp = 0

        self.available_sensors = []
        self.expected_sample_rates = {}
        self.sensors_eps = {}
        for sensor_name in self.sensors_df.keys():
            self.expected_sample_rates[sensor_name] = 0


        self.size = 0

    def set_details(self, details: object):

        self.activity_name = details['activity_id'] if 'activity_id' in details else None
        self.rid = details['activity_rid'] if 'activity_rid' in details else None
        self.debug = details['activity_debug'] if 'activity_debug' in details else None
        self.session_rid = details['session_rid'] if 'session_rid' in details else None
        self.device_id = details['device_id'] if 'device_id' in details else None
        self.timestamp = details['activity_timestamp'] if 'activity_timestamp' in details else 0

        # for sensor in ['accelerometer', 'gyroscope', 'linear_acceleration', 'gravity', 'magnetometer', 'roatation_vector']:
        #     if (sensor in details):
        #         self.expected_sample_rates[sensor + "_events"] = details[sensor] if (details[sensor] is not None and not math.isnan(details[sensor])) else 50

        return self

    def set_expected_sample_rate(self, sensor_name, sample_rate):
        self.expected_sample_rates[sensor_name] = sample_rate

    def add_touch_df(self, df: pd.DataFrame):
        ''' Takes DataFrame of all touch events from a SINGLE activity and parses it into Touch Actions '''

        if (df.empty):
            warnings.warn(f"[Warning] Trying to add empty touch df for Activity.")
            return

        if (self.keep_df == True):
            self.touch_df = df.copy()

        # self.set_details(df.iloc[0])

        df = df.sort_values('rid').reset_index()

        pointers = [[] for i in range(10)]
        
        for row_idx in range(len(df)):
            row = df.iloc[row_idx]

            ponter_id = row['pointer_id'] 
            if (ponter_id is None or math.isnan(ponter_id)):
                ponter_id = 0
            else:
                ponter_id = int(ponter_id)

            phase = row['phase'] 
            # debug = row['debug']
            # if ( (len(pointers[ponter_id]) > 0) and (pointers[ponter_id][-1]['debug'] != debug) ):
            #     warnings.warn(f"[Warning] Touch action changed debug info during tap, probably wrong pointer_id distinction for rid {row['rid']}, pointer_id: {ponter_id}, phase: {phase}; {self.activity_name}; ignoring")

            if (phase == 'BEGIN'):
                if (len(pointers[ponter_id]) > 0): # new BEGIN without previous ENDED
                    warnings.warn(f"[Warning] Touch BEGIN without ending previous. rid: {row['rid']}; activity {self.rid}")
                    self.touch_actions.append( TouchAction(pd.DataFrame(pointers[ponter_id])) )
                    self.size += 1

                pointers[ponter_id] = []
                pointers[ponter_id].append(row)
                
            elif (phase == 'MOVED'):
                if (len(pointers[ponter_id]) == 0): # no previous touch event for this pointer
                    warnings.warn(f"[Warning] Touch MOVED without previous events. rid: {row['rid']}; activity {self.rid}")

                pointers[ponter_id].append(row)

            elif (phase == 'ENDED'):
                if (len(pointers[ponter_id]) == 0): # no previous touch event for this pointer
                    warnings.warn(f"[Warning] Touch ENDED without previous events. rid: {row['rid']}; activity {self.rid}")

                pointers[ponter_id].append(row)
                self.touch_actions.append( TouchAction(pd.DataFrame(pointers[ponter_id])) )
                self.size += 1
                pointers[ponter_id] = []

            # else:
            #     warnings.warn(f"[Warning] Unknown touch phase {phase}; activity {self.rid}")
        
        return self


    def add_sensor_df(self, df: pd.DataFrame, sensor_type: str, eps_ms = 0):
        ''' Takes DataFrame of all sensor events from a SINGLE activity '''

        if (len(self.touch_actions) == 0 and self.keep_df == False):
            warnings.warn(f"[Warning] No touch action yet, and keep_df is false. Skipping.")
            return

        if (self.keep_df == True):
            if (sensor_type not in self.sensors_df.keys()):
                warnings.warn(f"[Warning] Unexpected sensor_type name {sensor_type}. Expected: {self.sensors_df.keys()}")

            self.sensors_df[sensor_type] = df.sort_values('rid').copy()


        for touch in self.touch_actions:
            touch.add_sensor_df(df, sensor_type, self.expected_sample_rates[sensor_type], eps_ms)

        self.available_sensors.append(sensor_type)
        self.sensors_eps[sensor_type] = eps_ms

    def __str__(self):
        return f"Activity {self.rid} / {self.activity_name}: {len(self.touch_actions)} touch actions; Sensors: {self.available_sensors}"

    def __lt__(self, other):
         return self.timestamp < other.timestamp


    def __getitem__(self, item):
        return self.touch_actions[item]

    def __iter__(self):
        self._iter_idx = -1
        return self

    def __next__(self):
        self._iter_idx += 1
        if (self._iter_idx >= len(self.touch_actions)):
            raise StopIteration()

        return self.touch_actions[self._iter_idx]



class Session:
    
    def __init__(self, rid):
        self.activities_by_name = {
            'LoginActivity': [],
            'ScrollingActivity': [],
            'TypingActivity': [],
            'ClickingActivity': []
        }

        self.activities = []

        self.rid = rid
        self.model = None
        self.device_id = None
        self.session_id = None
        self.timestamp = 0

        self.size = 0

    def add_activity(self, activity: Activity):

        if (self.rid != activity.session_rid):
            warnings.warn(f"[Warning] session_rid mismatch for activity_rid {activity.rid}. Old '{self.rid}' != new '{activity.session_rid}'")

        # if (self.device_id is None):
        #     self.device_id = activity.device_id
        # elif (self.device_id != activity.device_id):
        #     warnings.warn(f"[Warning] device_id mismatch for activity_rid {activity.rid}. Old '{self.device_id}' != new '{activity.device_id}'")



        self.activities_by_name[activity.activity_name].append(activity)
        self.activities.append(activity)
        self.size += 1
    
    def set_details(self, details):

        self.model = details['channel'] if 'channel' in details else None
        self.device_id = details['device_id'] if 'device_id' in details else None
        self.session_id = details['session_id'] if 'session_id' in details else None
        self.timestamp = details['timestamp'] if 'timestamp' in details else 0

        return self
    
    def __str__(self):
        return f"Session {self.rid} activities count: {[ len(self.activities_by_name[k]) for k in self.activities_by_name.keys()]}"

    def __iter__(self):
        self._iter_idx = -1
        return self

    def __next__(self):
        self._iter_idx += 1
        if (self._iter_idx >= len(self.activities)):
            raise StopIteration()

        return self.activities[self._iter_idx]

    def __getitem__(self, item):


        if (type(item) is tuple):
            activity_name, idx = item
            return self.activities_by_name[activity_name][idx]
        
        if (type(item) is str):
            return self.activities_by_name[item]

        else:
            return self.activities[item]


    def __lt__(self, other):
         return self.timestamp < other.timestamp


class Device:

    def __init__(self, device_id: str):
        self.device_id = device_id

        self.sessions = []
        self.model = None

        self.size = 0


    def fetch_activity(self, connector: DBConnector, activity_type, eps_ms = 0, sensor_tables_names = None):

        if (sensor_tables_names is None):
            sensor_tables_names = ['accelerometer_events', 'gyroscope_events', 'linear_acceleration_events', 'magnetometer_events', 'gravity_events', 'rotation_vector_events']

        all_touches = connector.query(f"""select
                                                s.device_id, s.channel, s.session_id, s.timestamp as session_timestamp,
                                                a.session_rid, a.activity_id, a.debug as activity_debug, a.timestamp as activity_timestmap,
                                                te.*
                                            from touch_events te, activities a, sessions s
                                            where te.activity_rid = a.rid and a.session_rid = s.rid
                                            and s.device_id = '{self.device_id}'
                                            and a.activity_id = '{activity_type}'
                                            order by rid;""")

        sensors_df = {}
        for sensor_table_name in sensor_tables_names:
            sensors_df[sensor_table_name] = connector.query(f"""select
                                                                    s.device_id, s.channel, s.session_id, s.timestamp as session_timestamp,
                                                                    a.session_rid, a.activity_id, a.debug as activity_debug, a.timestamp as activity_timestmap,
                                                                    e.*, sr.{sensor_table_name.replace('_events', '')} as sample_rate
                                                                from accelerometer_events e
                                                                inner join activities a on a.rid = e.activity_rid
                                                                inner join sessions s on s.rid = a.session_rid 
                                                                left join sample_rates sr on sr.activity_rid = a.rid
                                                                where s.device_id = '{self.device_id}'
                                                                and a.activity_id = '{activity_type}'
                                                                order by rid;""")

            sensors_df[sensor_table_name]['sample_rate'] = sensors_df[sensor_table_name]['sample_rate'].fillna(50)
        
        self.model = all_touches.iloc[0]['channel']

        for session_rid, session_df in all_touches.groupby('session_rid'):

            session = Session(session_rid).set_details({
                'channel': self.model,
                'device_id': self.device_id,
                'session_id': session_df.iloc[0]['session_id'],
                'timestamp': session_df.iloc[0]['session_timestamp']
            })

            for activity_rid, activity_df in session_df.groupby('activity_rid'):

                activity = Activity(activity_rid, keep_df=True).set_details(activity_df.iloc[0])
                activity.add_touch_df(activity_df)

                for sensor_name, sensor_df in sensors_df.items():
                    tmp = sensor_df[sensor_df['activity_rid'] == activity_rid]
                    if (tmp.empty):
                        activity.set_expected_sample_rate(sensor_name, 0)
                        continue
                    else:
                        activity.set_expected_sample_rate(sensor_name, tmp.iloc[0]['sample_rate'])

                    activity.add_sensor_df(tmp, sensor_name, eps_ms)

                session.add_activity(activity)

            self.add_session(session)

        return self


    # def fetch(self, connector: DBConnector, sensor_tables_names = None, sampling_rate_constraint = None, eps_ms = 0):
    #     if (sensor_tables_names is None):
    #         sensor_tables_names = ['accelerometer_events', 'gyroscope_events', 'linear_acceleration_events', 'magnetometer_events', 'gravity_events', 'rotation_vector_events']#, "linear_acceleration_events", "gravity"]

    #     all_activities_df = connector.query(f"""select a.session_rid, a.rid, s.channel, s.device_id, s.session_id, a.activity_id, a.debug, a.timestamp as activity_timestmap, s.timestamp as session_timestamp, sr.*
	#                                         from activities a
	#                                             inner join sessions s on s.rid = a.session_rid
	#                                             left join sample_rates sr on sr.activity_rid = a.rid
	#                                         where device_id = '{self.device_id}';""")

    #     if (sampling_rate_constraint):
    #         if (sampling_rate_constraint == 'low'):
    #             all_activities_df = all_activities_df[all_activities_df['accelerometer'].isna()]
    #         elif (sampling_rate_constraint == 'high'):
    #             all_activities_df = all_activities_df[~all_activities_df['accelerometer'].isna()]

    #     self.model = all_activities_df['channel'].unique()
        
        
    #     for session_rid, session_df in all_activities_df.groupby('session_rid'):

    #         session = Session(session_rid).set_details({
    #             'channel': self.model,
    #             'device_id': self.device_id,
    #             'session_id': session_df.iloc[0]['session_id'],
    #             'timestamp': session_df.iloc[0]['session_timestamp']
    #         })

    #         activities_in_session = session_df['rid'].unique()
    #         if (activities_in_session.size == 1):
    #             all_touch_df = connector.query(f"select * from touch_events where activity_rid = '{activities_in_session[0]}' order by rid;")
    #         else:
    #             all_touch_df = connector.query(f"select * from touch_events where activity_rid in {tuple(activities_in_session)} order by rid;")

    #         sensors_df = {}
    #         for sensor_table_name in sensor_tables_names:
    #             if (activities_in_session.size == 1):
    #                 sensors_df[sensor_table_name] = connector.query(f"select * from {sensor_table_name} where activity_rid = {activities_in_session[0]} order by rid;")
    #             else:
    #                 sensors_df[sensor_table_name] = connector.query(f"select * from {sensor_table_name} where activity_rid in {tuple(activities_in_session)} order by rid;")


    #         for activity_rid, touch_df in all_touch_df.groupby('activity_rid'):

    #             row_activity = session_df[session_df['rid'] == activity_rid]

    #             activity = Activity(activity_rid, keep_df=True).set_details(row_activity.iloc[0])
    #             activity.add_touch_df(touch_df)

    #             for sensor_name, sensor_df in sensors_df.items():
    #                 tmp = sensor_df[sensor_df['activity_rid'] == activity_rid]
    #                 if (tmp.empty):
    #                     activity.set_expected_sample_rate(sensor_name, 0)
    #                     continue

    #                 activity.add_sensor_df(tmp, sensor_name, eps_ms)

    #             session.add_activity(activity)

    #         self.add_session(session)

    #     return self

    def add_session(self, session: Session):
        self.sessions.append(session)
        self.size += 1

    def __str__(self):
        return f"Device: {self.device_id} / {self.model}; {len(self.sessions)} sessions."

    def __iter__(self):
        self._iter_idx = -1
        return self

    def __next__(self):
        self._iter_idx += 1
        if (self._iter_idx >= len(self.sessions)):
            raise StopIteration()

        return self.sessions[self._iter_idx]

    def __getitem__(self, item):
         return self.sessions[item]


######## Feature extraction


def single_tap_extraction(touch: TouchAction, prev_touch : TouchAction = None):

    touch_df = touch.touch_df

    ret = {
        'pointer_id': touch_df.iloc[0]['pointer_id'],

        'tap_duration': ((touch.end - touch.begin) / 1e6),

        'debug': touch_df.iloc[0]['debug'],

        'area_start': touch_df.iloc[0]['size'],
        'area_stop': touch_df.iloc[-1]['size'],

        'force_start': touch_df.iloc[0]['force'],
        'force_stop': touch_df.iloc[-1]['force'],

        'ellipse_major_start': touch_df.iloc[0]['ellipse_major'],
        'ellipse_major_stop': touch_df.iloc[-1]['ellipse_major'],
        'ellipse_minor_start': touch_df.iloc[0]['ellipse_minor'],
        'ellipse_minor_stop': touch_df.iloc[-1]['ellipse_minor'],
        'orientation_start': touch_df.iloc[-1]['orientation'],
        'orientation_stop': touch_df.iloc[-1]['orientation'],

        'x_start': touch_df.iloc[0]['x_raw'] / touch_df.iloc[0]['w'],
        'y_start': touch_df.iloc[0]['x_raw'] / touch_df.iloc[0]['h'],
        'x_stop': touch_df.iloc[-1]['y_raw'] / touch_df.iloc[-1]['w'],
        'y_stop': touch_df.iloc[-1]['y_raw'] / touch_df.iloc[-1]['h'],

    }

    if (prev_touch is not None):
        ret['flight_time'] = ((touch.begin - prev_touch.end) / 1e6)


    return ret

def standardize(column, m = None, s = None):

    mu = column.mean() if m is None else m
    sig = column.std() if s is None else s

    if (sig == 0):
        y = (column - mu)

    else:
        y = (column - mu) / sig
    
    return y, mu, sig


def normalize(column, u = None, l = None):
    # return column - column.mean()
    # return standardize(column)
    upper = column.max() if u is None else u
    lower = column.min() if l is None else l
    y = (column - lower)/(upper-lower)
    return y, upper, lower


""" ## OLD DTW
from dtw import *
def touch_dtw(touch_action0 : TouchAction, touch_action1 : TouchAction, sensor_type, axis):

    template = touch_action0.sensors[sensor_type]
    query = touch_action1.sensors[sensor_type]

    if (template.empty):
        warnings.warn(f"{sensor_type} touch_action0 is empty.")
        return math.nan

    if (query.empty):
        warnings.warn(f"{sensor_type} touch_action1 is empty.")
        return math.nan

    template_expected_sample_rate = touch_action0.sensors_expected_sample_rate[sensor_type]
    query_expected_sample_rate = touch_action1.sensors_expected_sample_rate[sensor_type]


    # if ( np.abs(template_expected_sample_rate - touch_action0.sensors_real_sample_rate[sensor_type]) > 15 ):
    #     warnings.warn(f"Template real and expected sample rate differes significantly. Expected {template_expected_sample_rate}, real {touch_action0.sensors_real_sample_rate[sensor_type]}")

    # if ( np.abs(query_expected_sample_rate - touch_action1.sensors_real_sample_rate[sensor_type]) > 15 ):
    #     warnings.warn(f"Query real and expected sample rate differes significantly. Expected {query_expected_sample_rate}, real {touch_action1.sensors_real_sample_rate[sensor_type]}")
    
    template = template[axis].values
    query = query[axis].values

    if (template_expected_sample_rate > query_expected_sample_rate):
        ratio = int(template_expected_sample_rate // query_expected_sample_rate)
        template = template[::ratio]
        # print(f"Template oversampled by ratio of {ratio}")
        # return dtw(normalize(template[axis].values[::ratio]), normalize(query[axis].values)).distance

    elif (template_expected_sample_rate < query_expected_sample_rate):
        ratio = int(query_expected_sample_rate // template_expected_sample_rate)
        query = query[::ratio]
        # print(f"Query oversampled by ratio of {ratio}")
        # return dtw(normalize(template[axis].values), normalize(query[axis].values[::ratio])).distance

    p1 = template.mean()#max()
    p2 = template.std()#min()

    return dtw(normalize(template, p1, p2), normalize(query, p1, p2)).distance

def touch_dtw_many(touch_templates, query : TouchAction, sensor_type, axis):

    ret = []

    for i, template in enumerate(touch_templates):
        ret.append(touch_dtw(template, query, sensor_type, axis))

    return ret


def sensor_extraction(touch : TouchAction, sensor_type : str, touch_templates : list):

    sensor_short_name = sensor_type.split('_')[0]

    # dtw_out = touch_dtw_many( touch_templates, touch, sensor_type, 'z'  )

    ret = {}

    for axis in ['x', 'y', 'z', 'm']:
        ret[sensor_short_name + '_dtw_min_' + axis] = np.min( touch_dtw_many( touch_templates, touch, sensor_type, axis  ) )
        ret[sensor_short_name + '_dtw_mean_' + axis] = np.mean( touch_dtw_many( touch_templates, touch, sensor_type, axis  ) )
        ret[sensor_short_name + '_min_' + axis] = np.min( touch.sensors[sensor_type][axis] )
        ret[sensor_short_name + '_max_' + axis] = np.max( touch.sensors[sensor_type][axis] )
        ret[sensor_short_name + '_mean_' + axis] = np.mean( touch.sensors[sensor_type][axis] )
        # ret[sensor_short_name + '_at_begin_' + axis] = touch.sensors[sensor_type][axis].values[0]
        # ret[sensor_short_name + '_at_end_' + axis] = touch.sensors[sensor_type][axis].values[-1]


    return ret

def build_features(activity: Activity, touch_templates : list):

    ret = []

    prev_t = None

    for t in sorted(activity.touch_actions):
        features = {
            "session_rid": activity.session_rid,
            "activity_rid": activity.rid,
            "activity_name": activity.activity_name,
            # "lin_freq": activity.expected_sample_rates['linear_acceleration_events'],
            # "gyr_freq": activity.expected_sample_rates['gyroscope_events'],
        }
        
        features.update( single_tap_extraction(t, prev_t) )
        features.update( sensor_extraction(t, "linear_acceleration_events", touch_templates) )
        features.update( sensor_extraction(t, "gyroscope_events", touch_templates) )
        features.update( sensor_extraction(t, "accelerometer_events", touch_templates) )

        prev_t = t

        ret.append(features)

    return ret
"""

from abc import ABC, abstractmethod

class FeatureProcessor(ABC):

    @abstractmethod
    def fit_transform(self, data):
        pass

    @abstractmethod
    def transform(self, data):
        pass

    @abstractmethod
    def pairwise_distance(self, data):
        pass

    @abstractmethod
    def distance(self, x, y):
        pass


from sklearn.metrics import pairwise_distances
from scipy.stats import boxcox

from sklearn.preprocessing import StandardScaler

class DistanceFeatureProcessor(FeatureProcessor):

    def __init__(self, metric = 'manhattan') -> None:
        self.metric = metric
        self.p1 = None
        self.p2 = None

        self.mean = None
        self.std = None


    def standardize(self, data: np.array):

        if (self.std == 0):
            return (data - self.mean)
        else:
            return (data - self.mean) / self.std    

    def fit_transform(self, data: np.array):

        data = np.log1p(data)
        
        self.mean = data.mean()
        self.std = data.std()
        data = self.standardize(data)

        return data


    def transform(self, data: np.array):
        data = np.log1p(data)
        data = self.standardize(data)
        
        return data


    def pairwise_distance(self, data: np.array):
        return pairwise_distances(data.reshape(-1, 1), metric=self.metric)

    def distance(self, x: np.array, y: np.array):
        return pairwise_distances(y.reshape(-1, 1), x.reshape(-1, 1), metric=self.metric)



"""
    def preprocess(self, data):
        data = np.log1p(data)
        data, self.p1, self.p2 = standardize(data)

        return data

    def preprocess_test(self, data):

        if (self.p1 is None or self.p2 is None):
            warnings.warn("Missing parameters to preprocess test dataset.")
            self.preprocess(data)            

        data = np.log1p(data)
        data, _, _ = standardize(data, self.p1, self.p2)

        return data

    def distance_matrix(self, data):
        return pairwise_distances(data.reshape(-1, 1), metric=self.metric)
"""

from scipy import interpolate
from dtaidistance import dtw

class DTWFeatureProcessor(FeatureProcessor):

    def __init__(self, psi = 5) -> None:
        self.psi = psi

    def standardize(self, data: np.array):

        mean = data.mean()
        std = data.std()

        if (std == 0):
            return (data - mean)
        else:
            return (data - mean) / std

    def fit_transform(self, data: np.array):
        
        ret = []

        for sample in data:
            ret.append( self.transform(sample) )

        return ret

    def transform(self, data: tuple):

        # timestamps in nanoseconds
        values, timestamps = data

        tck = interpolate.splrep(timestamps, values)#, s=0 k=3)

        xnew = np.arange( timestamps.min(), timestamps.max(), 2e6 )
        ynew = interpolate.splev(xnew, tck)

        ynew = self.standardize(ynew)

        return ynew

    def pairwise_distance(self, data: list):
        return dtw.distance_matrix(data, psi=self.psi)

    def distance(self, x: np.array, y: np.array):

        ret = []

        for ts in x:
            ret.append( dtw.distance(ts, y, psi=self.psi) )

        return np.array(ret)


from sklearn.model_selection import StratifiedKFold
from PyNomaly import loop
from sklearn.metrics import roc_auc_score
from sklearn.metrics import RocCurveDisplay


class Model():

    def __init__(self, n_neighbors: int) -> None:
        self.n_neighbors = n_neighbors

        self.train_feature_matrix = None
        self.n_train = 0
        self.train_distance_matrices = {}

    def fit(self, train_feature_matrix: pd.DataFrame, feature_processors: dict, feature_weights: dict):
        self.feature_processors = feature_processors
        self.feature_weights = feature_weights
        
        self.train_feature_matrix = train_feature_matrix.copy()
        self.n_train = len(self.train_feature_matrix)

        for feature_name, processor in self.feature_processors.items():
            self.train_feature_matrix[feature_name] = processor.fit_transform( self.train_feature_matrix[feature_name].values )
            self.train_distance_matrices[feature_name] = processor.pairwise_distance( self.train_feature_matrix[feature_name].values )

        return self

    def predict(self, test_vector: pd.Series):

        n_neighbors = self.n_neighbors
        if (n_neighbors > self.n_train):
            # print(f"n_neighbors set to number larger than number of training sample ({self.n_train}). Reducing for this run.")
            n_neighbors = self.n_train

        distance_matrices = []
        fweights = []

        for feature_name, processor in self.feature_processors.items():
            sample = processor.transform( np.array(test_vector[feature_name]) )
            train_test_distance_arr = processor.distance( self.train_feature_matrix[feature_name].values, sample )

            dmatrix = np.zeros( (self.n_train+1, self.n_train+1) )
            dmatrix[:self.n_train, :self.n_train] = self.train_distance_matrices[feature_name]
            dmatrix[-1, :self.n_train] = train_test_distance_arr
            dmatrix[:self.n_train, -1] = train_test_distance_arr
                        
            distance_matrices.append( dmatrix )
            fweights.append( self.feature_weights[feature_name] )

        weighted_distance = np.average( distance_matrices, axis=0, weights=fweights )
        
        dmatrix, nmatrix = nearest_subset(weighted_distance, n_neighbors)
            
        model = loop.LocalOutlierProbability(distance_matrix=dmatrix, neighbor_matrix=nmatrix, n_neighbors=n_neighbors).fit()
        scores = model.local_outlier_probabilities

        return scores[-1]


def evaluate(feature_matrix: pd.DataFrame, device_id: str, features: dict, n_folds = 3, shuffle = False, n_neighbors = 3, ax = None, feature_weights = None):

    X = feature_matrix.copy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle)
    y = np.array(X['device_id'] == device_id, dtype=int)

    auc_scores = []
    eers = []

    for i_fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train_pos = X_train[y_train == 1]
        y_train_pos = y_train[y_train == 1]

        model = Model(n_neighbors).fit(X_train_pos, features, feature_weights)

        y_pred = []

        for idx in range(y_test.size):
            test_row = X_test.iloc[idx]

            outlier_prob = model.predict(test_row)
            y_pred.append( outlier_prob )


        auc_scores.append( roc_auc_score(1-y_test, y_pred) )
        eer, thr = calculate_eer(1-y_test, y_pred)
        eers.append( eer )

        if (ax is not None):
            RocCurveDisplay.from_predictions(1-y_test, y_pred, ax=ax, name=f"Fold {i_fold}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)
            # ax.plot([1, 0], [0, 1], "k--", alpha=0.3)              
            ax.scatter(eer, 1-eer, label=f"Fold {i_fold} (EER = {eer:.2f})")
            ax.legend()
            ax.set_title(f"Device: {device_id}\nMean AUC: {np.mean(auc_scores):.2f}\nMean EER: {np.mean(eers):.2f}")


    return np.array(auc_scores), np.array(eers)

"""
def evaluate(feature_matrix: pd.DataFrame, device_id: str, features: dict, n_folds = 3, shuffle = False, n_neighbors = 3, ax = None, feature_weights = None):

    X = feature_matrix.copy()

    skf = StratifiedKFold(n_splits=n_folds, shuffle=shuffle)
    y = np.array(X['device_id'] == device_id, dtype=int)

    # print(y[y==1].size, "positive samples")

    auc_scores = []
    eers = []

    for i_fold, (train_index, test_index) in enumerate(skf.split(X, y)):
    
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y[train_index], y[test_index]
        
        X_train_pos = X_train[y_train == 1]
        y_train_pos = y_train[y_train == 1]

        if (len(X_train_pos) < n_neighbors):
            print(f"{device_id}: Not enough train data in fold {i_fold}; #Samples {len(X_train_pos)} < {n_neighbors} neighbors.")
            n_neighbors = len(X_train_pos)
        
        # print(f"Fold {i_fold}:\n\t#train: {y_train_pos.size}: {X_train_pos.index.values}\n\t#test: {y_test.size} ({y_test[y_test == 1].size} + {y_test[y_test == 0].size})")
        
        y_pred = []
        
        for idx in range(y_test.size):
            test_row = X_test.iloc[idx]
        
            distance_matrices = {}
            for feature, processor in features.items():
                tmp = np.append( processor.preprocess(X_train_pos[feature].values), processor.preprocess_test(test_row[feature]) ) # append single test row
                distance_matrices[feature] = processor.distance_matrix(tmp)
        
            weighted_distance = np.average( list(distance_matrices.values()), axis=0, weights=feature_weights )
            
            dmatrix, nmatrix = nearest_subset(weighted_distance, n_neighbors)
            
            model = loop.LocalOutlierProbability(distance_matrix=dmatrix, neighbor_matrix=nmatrix, n_neighbors=n_neighbors).fit()
            scores = model.local_outlier_probabilities
            y_pred.append(scores[-1])
            
        
        auc_scores.append( roc_auc_score(1-y_test, y_pred) )
        eer, thr = calculate_eer(1-y_test, y_pred)
        eers.append( eer )
        # print(y_test, y_pred)
        # print(thr, eer)

        if (ax is not None):
            RocCurveDisplay.from_predictions(1-y_test, y_pred, ax=ax, name=f"Fold {i_fold}")
            ax.plot([0, 1], [0, 1], "k--", alpha=0.3)                
            ax.set_title(f"Device: {device_id}\nMean AUC: {np.mean(auc_scores)}")

    return np.array(auc_scores), np.array(eers)
"""

from sklearn.metrics import roc_curve
from scipy.interpolate import interp1d
from scipy.optimize import brentq

def calculate_eer(y_test, y_pred):
    fpr, tpr, thresholds = roc_curve(y_test, y_pred)

    eer = brentq(lambda x: 1.0 - x - interp1d(fpr, tpr)(x), 0.0, 1.0)
    thresh = interp1d(fpr, thresholds)(eer)

    return eer, thresh


"""

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# Create a StratifiedKFold object with 3 folds
kfold = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Loop through the folds
for train_index, test_index in kfold.split(X, y):
  # Get the training and test data for this fold
  X_train, X_test = X[train_index], X[test_index]
  y_train, y_test = y[train_index], y[test_index]

  # Filter out the negative class data from the training set
  X_train_pos = X_train[y_train == 1]
  y_train_pos = y_train[y_train == 1]

  # Train the model on the positive class data
  model.fit(X_train_pos, y_train_pos)
  y_pred = model.predict_proba(X_test)[:,1]  # predict probability of positive class

  # Calculate the AUC score for this fold
  auc = roc_auc_score(y_test, y_pred)

  # Add the AUC score for this fold to the list of scores
  auc_scores.append(auc)

# Calculate the mean AUC score across all folds
mean_auc = np.mean(auc_scores)



"""


def nearest_subset(pairwise_dist_matrix, n_nearest):
    nearest_idx = np.argsort(pairwise_dist_matrix)
    idx = nearest_idx[:, :n_nearest]
    
    return pairwise_dist_matrix[idx, np.arange(pairwise_dist_matrix.shape[0])[:, np.newaxis]], idx

    