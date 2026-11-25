"""주최 측 heuristic baseline (경도·inclination 의 표준편차 변화로 node 를 찾는 규칙 기반 방법).

QR_Is 는 이 baseline 의 예측을 CatBoost 의 입력 feature 로 넣는다 (논문의 "model stacking").
이 파일은 QR_Is 저장소의 utils.py (heuristic 부분) 와 node.py 를 내용 변경 없이 합친 것이고,
그 코드는 ARCLab-MIT/splid-devkit 의 baseline_heuristic 에서 왔다 ((c) 2023 Peng Mun Siew, MIT License. Node: Liz Solera).

밖에서 쓰는 함수는 add_baseline_heuristic() 하나.
  입력: 객체 1기의 DataFrame → 출력: 같은 DataFrame + 열 4개
    EW_baseline_heuristic / NS_baseline_heuristic              heuristic 이 node 로 본 시점에 'IK-CK' 같은 라벨, 나머지는 'Nothing'
    EW_baseline_heuristic_ffill / NS_baseline_heuristic_ffill  위 라벨을 다음 node 까지 채운 것 (= 그 시점의 mode)

규칙 요약 (논문 ① 5.1)
  EW: 최근 24시간 경도 표준편차가 기준 (0.03 deg) 을 넘고 전후 하루 평균 경도가 0.3 deg 넘게 다르면 ID,
      다시 기준 아래로 내려와 하루 동안 0.3 deg 안에 머물면 IK, drift 중 표준편차가 크게 바뀌면 AD
  NS: EW station-keeping 구간 안에서 inclination 변화율로 IK 를 정함
  추진 type: 구간의 경도 (최대 - 최소) 와 표준편차의 비로 CK / EK 구분 (Node.char_mode)
"""
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

class Node:
    def __init__(self,
                 satcat: str, 
                 t: datetime, 
                 t0: datetime = None,
                 t1: datetime = None,
                 dt : timedelta = None,
                 index : int = None,
                 next_index : int = None,
                 ntype: str = None,
                 signal: str = None,
                 lon: float = None,
                 confidence: float = None,
                 mtype: str = None):
        '''
        satcat -> 5 digit norad ID\n
        t -> timestamp\n
        t0 -> preceding epoch (optional)\n
        t1 -> proceding epoch (optional)\n
        dt -> timestep (optional)\n
        index -> index associated with timestep array (optional)\n
        ntype -> node type; options: "ID", "IK", "AD" (optional)\n
        lon -> satellite longitude (deg) at t0 (optional)
        '''
        self.satcat = int(satcat)
        self.t = t
        self.tstring = str(t.isoformat())
        if t0 is None:
            t0 = t
        if t1 is None:
            t1 = t0
        self.t0 = t0
        self.t1 = t1
        self.neep = t0
        self.dt = dt
        self.index = index
        self.next_index = None
        self.type = ntype
        self.signal = signal
        self.notes = []
        self.lon = lon
        self.confidence = confidence
        self.correlated = False
        self.time_err = 0.0
        self.mode_lons = None
        self.mode_incs = None
        self.EW_std = None
        self.NS_std = None
        self.mtype = mtype
    
    def char_mode(self,next_index=None,lons=None,incs=None):
        self.next_index = next_index
        self.mode_lons = lons[self.index:self.next_index]
        self.mode_incs = incs[self.index:self.next_index]
        EW_db = np.max(self.mode_lons) - np.min(self.mode_lons)
        EW_sd = np.std(self.mode_lons)+sys.float_info.epsilon
        EW = (EW_db-EW_sd)/EW_sd
        self.NS_std = np.std(incs)
        if self.type=="ID":
            self.mtype = "NK"
        elif self.type=="AD":
            self.mtype = "NK"
        elif self.type=="IK":
            self.mtype = "CK" if EW < 5.1 else "EK"
        elif self.type=="ES":
            self.mtype = "ES"
        elif self.type=="SS":
            self.mtype = "CK" if EW < 5.1 else "EK"

    def describe(self, ntype: str = None):
        '''Returns a description of the node type'''
        self.type = ntype if ntype!=None else self.type
        description = "unknown"
        if self.type == "ID":
            description = "Initiate Drift"
        elif self.type == "IK":
            description = "End Drift"
        elif self.type == "AD":
            description = "Adjust Drift"
        return description

    def ID(self):
        '''Returns node ID string'''
        id = (self.type+"."+str(self.satcat)+"@"+self.t.strftime("%Y-%m-%dT%H:%M"))
        # if self.dt != None:
        #     id = (self.type+"."+str(self.satcat)+"@"+self.t0.strftime("%Y-%m-%dT%H:%M")+"+"+str(int(self.dt.total_seconds()/60)))
        # else:
        #     id = (self.type+"."+str(self.satcat)+"@"+self.t0.strftime("%Y-%m-%dT%H:%M"))
        return id

    def note(self, notes: str):
        '''Creates a note for a node'''
        self.notes.append(notes)

    def see_notes(self):
        '''Prints all notes associated with a node'''
        print(self.ID())
        for i in self.notes:
            print("     ", i)
            
    def clear_notes(self):
        '''Deletes all notes associated with a node'''
        self.notes = []

    def correlate(self, t):
        self.correlated = True
        self.time_err = (t - self.t).total_seconds()/3600


class index_dict:
    def __init__(self):
        self.times = self.IDADIK()
        self.indices = []
        self.AD_dex = []
        self.modes = self.mode()

    class IDADIK:
        def __init__(self):
            self.ID = []
            self.AD = []
            self.IK = []

    class mode:
        def __init__(self):
            self.SK = []
            self.end = []


def detect_ew_pol_nodes(data, satcat, starttime, endtime):

    # Get std for longitude over a 24 hours window
    lon_std = []
    steps_per_day = 12
    lon_was_baseline = True
    lon_baseline = 0.03
    detected = index_dict()
    for i in range(len(data["Longitude (deg)"])):
        if i <= steps_per_day:
            lon_std.append(np.std(data["Longitude (deg)"][0:steps_per_day]))
        else:
            lon_std.append(np.std(data["Longitude (deg)"][i-steps_per_day:i]))

    ssEW = Node(satcat=satcat,
                t=starttime,
                index=0,
                ntype="SS",
                signal="EW")

    # Run LS detection
    # if at least 1 day has elapsed since t0
    for i in range(steps_per_day+1, len(lon_std)-steps_per_day):
        max_lon_std_24h = np.max(lon_std[i-steps_per_day:i])
        min_lon_std_24h = np.min(lon_std[i-steps_per_day:i])
        A = np.abs(max_lon_std_24h-min_lon_std_24h)/2
        th_ = 0.95*A

        # ID detection
        # if sd is elevated & last sd was at baseline
        if (lon_std[i] > lon_baseline) & lon_was_baseline:
            # mean of previous day's longitudes
            before = np.mean(data["Longitude (deg)"][i-steps_per_day:i])
            # mean of next day's longitudes
            after = np.mean(data["Longitude (deg)"][i:i+steps_per_day])
            # if not temporary noise, then real ID
            # if means are different
            if np.abs(before-after) > 0.3:
                # the sd is not yet back at baseline
                lon_was_baseline = False
                index = i
                if i < steps_per_day+2:
                    ssEW.mtype = "NK"
                else:
                    detected.times.ID.append(starttime+timedelta(hours=i*2))
        # IK detection
        # elif sd is not elevated and drift has already been initialized
        elif (lon_std[i] <= lon_baseline) & (not lon_was_baseline):
            # toggle end-of-drift boolean
            drift_ended = True
            # for the next day, check...
            for j in range(steps_per_day):
                # if the longitude changes from the current value
                if np.abs(data["Longitude (deg)"][i]-data["Longitude (deg)"][i+j]) > 0.3:
                    # the drift has not ended
                    drift_ended = False
            if drift_ended:                                                 # if the drift has ended
                # the sd is back to baseline
                lon_was_baseline = True
                # save tnow as end-of-drift
                detected.times.IK.append(starttime+timedelta(hours=i*2))
                # save indices for t-start & t-end
                detected.indices.append([index, i])

        # Last step
        elif (i == (len(lon_std)-steps_per_day-1))\
                & (not lon_was_baseline):
            detected.times.IK.append(starttime+timedelta(hours=i*2))
            detected.indices.append([index, i])

        # AD detection
        # elif sd is elevated and drift has already been initialized
        elif ((lon_std[i]-max_lon_std_24h > th_) or (min_lon_std_24h-lon_std[i] > th_)) & (not lon_was_baseline):
            if i >= steps_per_day+3:
                detected.times.AD.append(starttime+timedelta(hours=i*2))
                detected.AD_dex.append(i)
    return detected, ssEW, lon_std


def filter_and_merge_nodes(data, detected, ssEW, lon_std, satcat, starttime, endtime, steps_per_day):
    nodes = []
    filtered = index_dict()
    longitudes = data["Longitude (deg)"]
    inclinations = data["Inclination (deg)"]

    def add_node(n):
        nodes[len(nodes)-1].char_mode(
            next_index=n.index,
            lons=longitudes,
            incs=inclinations
        )
        if n.type == "AD":
            nodes[len(nodes)-1].mtype = "NK"

        if (nodes[len(nodes)-1].mtype != "NK"):
            filtered.indices.append([nodes[len(nodes)-1].index, n.index])
            filtered.modes.SK.append(nodes[len(nodes)-1].mtype)
            stop_NS = True if n.type == "ID" else False
            filtered.modes.end.append(stop_NS)
        nodes.append(n)

    es = Node(satcat=satcat,
              t=endtime,
              index=len(data["Longitude (deg)"])-1,
              ntype="ES",
              signal="ES",
              mtype="ES")

    toggle = True
    nodes.append(ssEW)
    if len(detected.times.IK) == 1:
        if len(detected.times.ID) == 1:
            # keep the current ID
            filtered.times.ID.append(detected.times.ID[0])
            ID = Node(satcat,
                      detected.times.ID[0],
                      index=detected.indices[0][0],
                      ntype='ID',
                      lon=longitudes[detected.indices[0][0]],
                      signal="EW")
            add_node(ID)
        filtered.times.IK.append(detected.times.IK[0])
        IK = Node(satcat,
                  detected.times.IK[0],
                  index=detected.indices[0][1],
                  ntype='IK',
                  lon=longitudes[detected.indices[0][1]],
                  signal="EW")
        apnd = True
        if len(detected.times.AD) == 1:
            AD = Node(satcat,
                      detected.times.AD[0],
                      index=detected.AD_dex[0],
                      ntype="AD",
                      signal="EW")
            add_node(AD)
        elif len(detected.times.AD) == 0:
            pass
        else:
            for j in range(len(detected.times.AD)):
                ad = Node(satcat,
                          detected.times.AD[j],
                          index=detected.AD_dex[j],
                          ntype="AD",
                          signal="EW")
                ad_next = Node(satcat,
                               detected.times.AD[j+1],
                               index=detected.AD_dex[j+1],
                               ntype="AD",
                               signal="EW") \
                    if j < (len(detected.times.AD)-1) else None
                if (ad.t > starttime+timedelta(hours=detected.indices[0][0]*2)) & (ad.t < IK.t):
                    if apnd & (ad_next is not None):
                        if ((ad_next.t-ad.t) > timedelta(hours=24)):
                            add_node(ad)
                        else:
                            add_node(ad)
                            apnd = False
                    elif apnd & (ad_next is None):
                        add_node(ad)
                    elif (not apnd) & (ad_next is not None):
                        if ((ad_next.t-ad.t) > timedelta(hours=24)):
                            apnd = True
        if detected.indices[0][1] != (len(lon_std)-steps_per_day-1):
            add_node(IK)

    # for each longitudinal shift detection
    for i in range(len(detected.times.IK)-1):
        if toggle:                                                            # if the last ID was not discarded
            # if the time between the current IK & next ID is longer than 48 hours
            if ((starttime+timedelta(hours=detected.indices[i+1][0]*2)-detected.times.IK[i]) > timedelta(hours=36)):
                # keep the current ID
                filtered.times.ID.append(detected.times.ID[i])
                # keep the current IK
                filtered.times.IK.append(detected.times.IK[i])
                ID = Node(satcat,
                          detected.times.ID[i],
                          index=detected.indices[i][0],
                          ntype='ID',
                          lon=longitudes[detected.indices[i][0]],
                          signal="EW")
                add_node(ID)
                IK = Node(satcat,
                          detected.times.IK[i],
                          index=detected.indices[i][1],
                          ntype='IK',
                          lon=longitudes[detected.indices[i][1]],
                          signal="EW")
                apnd = True
                for j in range(len(detected.times.AD)):
                    ad = Node(satcat,
                              detected.times.AD[j],
                              index=detected.AD_dex[j],
                              ntype="AD",
                              signal="EW")
                    ad_next = Node(satcat,
                                   detected.times.AD[j+1],
                                   index=detected.AD_dex[j+1],
                                   ntype="AD",
                                   signal="EW") \
                        if j < (len(detected.times.AD)-1) else None
                    if (ad.t > ID.t) & (ad.t < IK.t):
                        if apnd & (ad_next is not None):
                            if ((ad_next.t-ad.t) > timedelta(hours=24)):
                                add_node(ad)
                            else:
                                add_node(ad)
                                apnd = False
                        elif apnd & (ad_next is None):
                            add_node(ad)
                        elif (not apnd) & (ad_next is not None):
                            if ((ad_next.t-ad.t) > timedelta(hours=24)):
                                apnd = True
                if detected.indices[0][1] != (
                        len(lon_std)-steps_per_day-1):
                    add_node(IK)
                # if the next drift is the last drift
                if i == len(detected.times.IK)-2:
                    # keep the next ID
                    filtered.times.ID.append(
                        starttime+timedelta(hours=detected.indices[i+1][0]*2))
                    ID = Node(satcat,
                              starttime +
                              timedelta(hours=detected.indices[i+1][0]*2),
                              index=detected.indices[i+1][0],
                              ntype='ID',
                              lon=longitudes[detected.indices[i+1][0]],
                              signal="EW")
                    add_node(ID)
                    IK = Node(satcat,
                              detected.times.IK[i+1],
                              index=detected.indices[i+1][1],
                              ntype='IK',
                              lon=longitudes[detected.indices[i+1][1]],
                              signal="EW")
                    apnd = True
                    for j in range(len(detected.times.AD)):
                        ad = Node(satcat,
                                  detected.times.AD[j],
                                  index=detected.AD_dex[j],
                                  ntype="AD",
                                  signal="EW")
                        ad_next = Node(satcat,
                                       detected.times.AD[j+1],
                                       index=detected.AD_dex[j+1],
                                       ntype="AD",
                                       signal="EW") \
                            if j < (len(detected.times.AD)-1) else None
                        if (ad.t > ID.t) & (ad.t < IK.t):
                            if apnd & (ad_next is not None):
                                if ((ad_next.t-ad.t) > timedelta(
                                        hours=24)):
                                    add_node(ad)
                                else:
                                    add_node(ad)
                                    apnd = False
                            elif apnd & (ad_next is None):
                                add_node(ad)
                            elif (not apnd) & (ad_next is not None):
                                if ((ad_next.t-ad.t) > timedelta(
                                        hours=24)):
                                    apnd = True
                    if detected.indices[i][1] != (
                            len(lon_std)-steps_per_day-1):
                        filtered.times.IK.append(
                            detected.times.IK[i+1])      # keep the next IK
                        add_node(IK)
            # if the next ID and the current IK are 48 hours apart or less
            else:
                ID = Node(satcat,
                          detected.times.ID[i],
                          index=detected.indices[i][0],
                          ntype='ID',
                          lon=longitudes[detected.indices[i][0]],
                          signal="EW")                                          # keep the current ID
                add_node(ID)
                AD = Node(satcat,
                          detected.times.IK[i],
                          index=detected.indices[i][1],
                          ntype='AD',
                          lon=longitudes[detected.indices[i][1]],
                          signal="EW")                                          # change the current IK to an AD
                IK = Node(satcat,
                          detected.times.IK[i+1],
                          index=detected.indices[i+1][1],
                          ntype='IK',
                          lon=longitudes[detected.indices[i+1][1]],
                          signal="EW")                                          # exchange the current IK for the next one
                add_node(AD)
                apnd = True
                for j in range(len(detected.times.AD)):
                    ad = Node(satcat,
                              detected.times.AD[j],
                              index=detected.AD_dex[j],
                              ntype="AD",
                              signal="EW")
                    ad_next = Node(satcat,
                                   detected.times.AD[j+1],
                                   index=detected.AD_dex[j+1],
                                   ntype="AD",
                                   signal="EW") \
                        if j < (len(detected.times.AD)-1) else None
                    if (ad.t > ID.t) & (ad.t < IK.t):
                        if apnd & (ad_next is not None):
                            if ((ad_next.t-ad.t) > timedelta(hours=24)):
                                add_node(ad)
                            else:
                                add_node(ad)
                                apnd = False
                        elif apnd & (ad_next is None):
                            add_node(ad)
                        elif (not apnd) & (ad_next is not None):
                            if ((ad_next.t-ad.t) > timedelta(hours=24)):
                                apnd = True
                if detected.indices[0][1] != (
                        len(lon_std)-steps_per_day-1):
                    add_node(IK)
                filtered.times.ID.append(detected.times.ID[i])
                filtered.times.AD.append(detected.times.IK[i])
                filtered.times.IK.append(detected.times.IK[i+1])
                # skip the redundant drift
                toggle = False
        else:
            toggle = True
    add_node(es)
    return nodes, filtered


def detect_sn_pol_nodes(data, nodes, filtered, satcat, starttime, steps_per_day):

    inclinations = data["Inclination (deg)"]
    ssNS = Node(
        satcat=satcat,
        t=starttime,
        index=0,
        ntype="SS",
        signal="NS")
    for j in range(len(filtered.indices)):
        indices = filtered.indices[j]
        first = True if indices[0] == 0 else False
        times = []
        dexs = []
        inc = inclinations[indices[0]:indices[1]].to_numpy()
        t = np.arange(indices[0], indices[1])*2
        rate = (steps_per_day/(indices[1]-indices[0])
                )*(np.max(inc)-np.min(inc))
        XIPS_inc_per_day = 0.0005  # 0.035/30
        if (rate < XIPS_inc_per_day) and (indices[0] < steps_per_day) and (indices[1] > steps_per_day):
            if filtered.modes.end[j]:
                nodes.append(Node(
                    satcat=satcat,
                    t=starttime+timedelta(hours=indices[1]*2),
                    index=indices[1],
                    ntype="ID",
                    signal="NS",
                    mtype="NK"
                ))
            ssNS.mtype = filtered.modes.SK[j]
        elif (rate < XIPS_inc_per_day):
            nodes.append(Node(
                satcat=satcat,
                t=starttime+timedelta(hours=indices[1]*2),  # times[0],
                index=indices[1],  # dexs[0],
                # t=times[0],
                # index=dexs[0],
                ntype="IK",
                signal="NS",
                mtype=filtered.modes.SK[j]
            ))
            if filtered.modes.end[j]:
                nodes.append(Node(
                    satcat=satcat,
                    t=starttime+timedelta(hours=indices[1]*2),
                    index=indices[1],
                    ntype="ID",
                    signal="NS",
                    mtype="NK"
                ))
        else:
            dt = [0.0]
            for i in range(len(inc)-1):
                dt.append((inc[i+1]-inc[i])/(2*60*60))
            prev = 1.0
            for i in range(len(dt)-1):
                if np.abs(dt[i]) > 5.5e-7:
                    times.append(starttime+timedelta(hours=float(t[i])))
                    dexs.append(i+indices[0])
                    if (np.abs(np.mean(inc[0:i])-np.mean(inc[i:len(inc)]))/np.std(inc[0:i]))/prev < 1.0:
                        if first and len(times) == 2:
                            ssNS.mtype = filtered.modes.SK[0]
                            first = False
                    elif len(times) == 2:
                        first = False
                    prev = np.abs(
                        np.mean(inc[0:i])-np.mean(inc[i:len(inc)]))/np.std(inc[0:i])

            if len(times) > 0:
                nodes.append(Node(
                    satcat=satcat,
                    t=times[0],
                    index=dexs[0],
                    ntype="IK",
                    signal="NS",
                    mtype=filtered.modes.SK[j]
                ))
                if filtered.modes.end[j]:
                    nodes.append(Node(
                        satcat=satcat,
                        t=starttime+timedelta(hours=indices[1]*2),
                        index=indices[1],
                        ntype="ID",
                        signal="NS",
                        mtype="NK"
                    ))
            elif filtered.indices[0][0] == 0:
                ssNS.mtype = filtered.modes.SK[0]
            else:
                ssNS.mtype = "NK"
    nodes.append(ssNS)
    nodes.sort(key=lambda x: x.t)
    return nodes


def data_post_processing(nodes, starttime):
    # Convert timestamp back into timeindex and format the output to the correct format in a pandas dataframe
    ObjectID_list = []
    TimeIndex_list = []
    Direction_list = []
    Node_list = []
    Type_list = []
    for i in range(len(nodes)):
        ObjectID_list.append(nodes[i].satcat)
        TimeIndex_list.append(
            int(((nodes[i].t-starttime).days*24+(nodes[i].t-starttime).seconds/3600)/2))
        Direction_list.append(nodes[i].signal)
        Node_list.append(nodes[i].type)
        Type_list.append(nodes[i].mtype)

    # Initialize data of lists.
    data = {'ObjectID': ObjectID_list,
            'TimeIndex': TimeIndex_list,
            'Direction': Direction_list,
            'Node': Node_list,
            'Type': Type_list}

    # Create the pandas DataFrame
    prediction = pd.DataFrame(data)

    return prediction


def data_to_add(data_df, ground_truth_object):
    ground_truth_object.Type = ground_truth_object.Type.fillna('NK')
    ground_truth_EW = ground_truth_object[ground_truth_object['Direction'] == 'EW'].copy(
    )
    ground_truth_NS = ground_truth_object[ground_truth_object['Direction'] == 'NS'].copy(
    )

    # Create 'EW' and 'NS' labels and fill 'unknown' values
    ground_truth_EW['EW_baseline_heuristic'] = ground_truth_EW['Node'] + \
        '-' + ground_truth_EW['Type']
    ground_truth_NS['NS_baseline_heuristic'] = ground_truth_NS['Node'] + \
        '-' + ground_truth_NS['Type']
    ground_truth_EW.drop(
        ['Node', 'Type', 'Direction'], axis=1, inplace=True)
    ground_truth_NS.drop(
        ['Node', 'Type', 'Direction'], axis=1, inplace=True)
    merged_df = pd.merge(data_df,
                         ground_truth_EW.sort_values('TimeIndex'),
                         on=['TimeIndex', 'ObjectID'],
                         how='left')
    merged_df = pd.merge_ordered(merged_df,
                                 ground_truth_NS.sort_values(
                                     'TimeIndex'),
                                 on=['TimeIndex', 'ObjectID'],
                                 how='left')

    # Fill 'unknown' values in 'EW' and 'NS' columns that come before the first valid observation
    merged_df['EW_baseline_heuristic_ffill'] = merged_df['EW_baseline_heuristic'].ffill()
    merged_df['NS_baseline_heuristic_ffill'] = merged_df['NS_baseline_heuristic'].ffill()

    merged_df['EW_baseline_heuristic'].fillna('Nothing', inplace=True)
    merged_df['NS_baseline_heuristic'].fillna('Nothing', inplace=True)

    return merged_df


def add_baseline_heuristic(data, objectId, starttime, endtime, feature_cols):
    detected, ssEW, lon_std = detect_ew_pol_nodes(
        data.loc[data.ObjectID == objectId][feature_cols], objectId, starttime, endtime)
    nodes, filtered = filter_and_merge_nodes(
        data.loc[data.ObjectID == objectId][feature_cols], detected, ssEW, lon_std,   objectId, starttime, endtime, 12)
    nodes = detect_sn_pol_nodes(
        data, nodes, filtered, objectId, starttime, 12)
    merged_df = data_to_add(
        data.loc[data.ObjectID == objectId], data_post_processing(nodes, starttime))
    return merged_df
