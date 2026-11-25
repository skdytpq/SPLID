"""후처리: 모델 출력 → 제출 형식 [ObjectID, TimeIndex, Direction, Node, Type].

원 코드 대응
  detections_from_outputs()   base/localizer.py  postprocess_predictions()
  combine_detections()        submission.py      (AD·IK 와 ID 탐지를 합치고 SS 를 더하는 부분)
  assign_types()              base/classifier.py fill_unknown_types_based_on_preds()
  assign_nodes()              base/classifier.py fill_unknwon_nodes_based_on_type()
  drop_ns_outside_ew_sk()     base/classifier.py remove_ns_during_ew_nk_method()

pandas 연산은 결과가 달라지지 않도록 원 코드의 순서를 그대로 따랐다.
"""
import numpy as np
import pandas as pd

DIRECTIONS = ['EW', 'NS']
COLUMNS = ['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']


def detections_from_outputs(outputs_df, thresholds, merge_distance):
    """localizer 출력 (시점마다 0 ~ 100) 을 node 시점 목록으로 바꾼다.

    outputs_df: [ObjectID, TimeIndex, EW_Loc, NS_Loc]
      1) 방향별 임계값 이상인 시점을 고른다
      2) 연속한 시점들은 가운데 1점으로 줄인다  (예: 임계값을 넘은 t=100..104 → 102)
      3) merge_distance 미만으로 붙은 탐지는 하나로 합친다
    결과의 Node·Type 은 아직 'UNKNOWN'.
    """
    df = outputs_df.copy()
    for d, threshold in zip(DIRECTIONS, thresholds):
        df[f'{d}_Loc'] = df[f'{d}_Loc'] >= threshold

    dir_dfs = []
    for d in DIRECTIONS:
        other = 'EW' if d == 'NS' else 'NS'
        dir_df = df.loc[df[f'{d}_Loc'] == True].copy().sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
        dir_df[f'{other}_Loc'] = False
        if len(dir_df) > 0:
            # TimeIndex 가 1 씩 이어지는 묶음마다 번호를 매기고, 묶음의 가운데 행만 남김
            dir_df['consecutive'] = (dir_df['TimeIndex'] - dir_df['TimeIndex'].shift(1) != 1).cumsum()
            dir_df = (dir_df.groupby('consecutive').apply(lambda run: run.iloc[int(len(run) / 2), :])
                      .reset_index(drop=True).drop(columns=['consecutive']))
        dir_dfs.append(dir_df)
    df = pd.concat(dir_dfs)
    df = df.loc[df.duplicated(keep='first') == False].reset_index(drop=True)

    if merge_distance > 0:
        dir_dfs = []
        for d in DIRECTIONS:
            sub_df = df.loc[df[f'{d}_Loc'] == True].copy().sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
            sub_df['diff'] = 3000
            # (원 코드 그대로) 조건에 sub_df 가 아니라 df 의 TimeIndex 를 쓴다. 탐지 시점은 대부분 0 보다 커서
            # 사실상 "모든 행에 대해 직전 탐지와의 간격" 을 구하는 것과 같다
            sub_df.loc[(df['TimeIndex'] > 0), 'diff'] = sub_df.loc[(df['TimeIndex'] > 0), 'TimeIndex'].diff()
            sub_df.loc[sub_df['diff'] < 0, 'diff'] = 3000
            for index in sub_df.index[sub_df['diff'] < merge_distance]:
                if index == 0:
                    continue
                if sub_df.at[index - 1, 'ObjectID'] == sub_df.at[index, 'ObjectID']:
                    sub_df.at[index, 'TimeIndex'] = int((sub_df.at[index - 1, 'TimeIndex'] + sub_df.at[index, 'TimeIndex']) / 2.0)
                    sub_df.drop(index=(index - 1), inplace=True)
            dir_dfs.append(sub_df)
        df = pd.concat(dir_dfs).reset_index(drop=True)

    sub_dfs = []
    for d in DIRECTIONS:
        sub_df = df.loc[df[f'{d}_Loc'] == True].copy()
        sub_df['Direction'] = d
        sub_dfs.append(sub_df)
    df = pd.concat(sub_dfs)
    df['Node'] = 'UNKNOWN'
    df['Type'] = 'UNKNOWN'
    return df[COLUMNS].sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)


def combine_detections(adik_df, id_df, object_ids):
    """AD·IK localizer 와 ID localizer 의 탐지를 합치고, 객체마다 시작 node (SS, TimeIndex 0) 를 더한다.

    - ID localizer 가 찾은 시점은 Node=ID, Type=NK 로 확정 (drift 시작 뒤는 station-keeping 이 아님)
    - 두 localizer 가 같은 시점·방향을 찾았으면 하나만 남기고 Node·Type 을 UNKNOWN 으로 되돌려 classifier 에 맡김
    """
    id_df = id_df.copy()
    id_df['Node'] = 'ID'
    id_df['Type'] = 'NK'
    df = pd.concat([adik_df, id_df]).sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
    key = ['ObjectID', 'TimeIndex', 'Direction']
    duplicated = df.duplicated(subset=key, keep=False)
    df.loc[duplicated, ['Node', 'Type']] = 'UNKNOWN'
    df = df[df.duplicated(subset=key, keep='first') == False]

    start_nodes = [pd.DataFrame({'ObjectID': object_ids, 'TimeIndex': 0, 'Direction': d, 'Node': 'SS', 'Type': 'UNKNOWN'})
                   for d in ['NS', 'EW']]
    return pd.concat([df] + start_nodes)


def assign_types(types_df, detections_df):
    """Type 이 UNKNOWN 인 탐지에 classifier 가 그 시점에 예측한 type 을 넣는다.

    types_df: [ObjectID, TimeIndex, EW_Type, NS_Type]
    """
    dfs = []
    for d in DIRECTIONS:
        sub_df = detections_df.loc[detections_df['Direction'] == d]
        sub_df = types_df[['ObjectID', 'TimeIndex', f'{d}_Type']].merge(sub_df, how='inner', on=['ObjectID', 'TimeIndex'])
        sub_df = sub_df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)
        unknown = sub_df['Type'] == 'UNKNOWN'
        sub_df.loc[unknown, 'Type'] = sub_df.loc[unknown, f'{d}_Type']
        dfs.append(sub_df)
    return pd.concat(dfs)[COLUMNS].sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)


def assign_nodes(detections_df):
    """Node 가 UNKNOWN 인 탐지의 node 종류를 "직전 mode type → 이번 mode type" 규칙으로 정한다.

        TimeIndex 0                      → SS
        NK → NK                          → AD (drift 중 조정). NS 방향에는 AD 가 없으므로 ID
        station-keeping (CK/EK/HK) → NK  → ID (drift 시작)
        NK → station-keeping             → IK (station-keeping 시작)
        station-keeping → station-keeping → IK (정의상 없는 전환이지만 station-keeping 으로 끝나므로)
    """
    dfs = []
    for d in DIRECTIONS:
        sub_df = detections_df.loc[detections_df['Direction'] == d].copy()
        sub_df = sub_df.sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)      # 직전 행 = 같은 객체의 직전 node
        for object_id in sub_df['ObjectID'].unique():
            for index, row in sub_df.loc[sub_df['ObjectID'] == object_id].iterrows():
                if sub_df.loc[index, 'Node'] != 'UNKNOWN':
                    continue
                if row['TimeIndex'] == 0:
                    node = 'SS'
                else:
                    prev_type, next_type = sub_df.loc[index - 1, 'Type'], sub_df.loc[index, 'Type']
                    if prev_type == 'NK' and next_type == 'NK':
                        node = 'AD' if d == 'EW' else 'ID'
                    elif prev_type in ['CK', 'EK', 'HK'] and next_type == 'NK':
                        node = 'ID'
                    else:
                        node = 'IK'
                sub_df.loc[index, 'Node'] = node
        dfs.append(sub_df)
    return pd.concat(dfs)[COLUMNS].sort_values(['ObjectID', 'TimeIndex']).reset_index(drop=True)


def drop_ns_outside_ew_sk(submission_df):
    """EW 방향이 station-keeping 이 아닌 (NK) 동안의 NS node 를 지운다.

    챌린지 정의상 NS station-keeping 은 EW station-keeping 중에만 있다. EW 예측이 NS 보다 정확하므로 EW 를 기준으로 삼는다.
    남기는 NS node: SS, EW node (AD 제외) 의 -6 ~ +5 step 안에 있는 것, 그 시점의 EW mode 가 NK 가 아닌 것.
    """
    df = submission_df.copy()
    df['EW_Type'] = df['Type']
    df['EW_Node'] = df['Node']
    df['EW_Loc'] = 0
    df.loc[df['Direction'] == 'EW', 'EW_Loc'] = 1
    df.loc[df['Direction'] == 'NS', 'EW_Type'] = np.nan
    df.loc[df['Direction'] == 'NS', 'EW_Node'] = np.nan
    df.ffill(inplace=True)          # NS 행에 "그 시점에 유효한 EW mode" 를 채움 (행은 ObjectID, TimeIndex 순)

    def mark_near_ew_nodes(obj_df):
        ew_times = obj_df.loc[(obj_df['EW_Loc'] == 1) & (obj_df['EW_Node'] != 'AD'), 'TimeIndex'].to_numpy()
        for t in ew_times:          # 시점 오차를 감안해 EW node 주변도 허용
            obj_df.loc[obj_df['TimeIndex'].isin(range(t - 6, t + 6)), 'EW_Loc'] = 1
        return obj_df

    df = df.groupby('ObjectID').apply(mark_near_ew_nodes).reset_index(level='ObjectID', drop=True)
    is_ns = df['Direction'] == 'NS'
    df = df.loc[(df['Direction'] == 'EW') | (is_ns & (df['Node'] == 'SS')) | (is_ns & (df['EW_Loc'] == 1)) |
                (is_ns & (df['EW_Type'] != 'NK'))]
    return df[submission_df.columns]
