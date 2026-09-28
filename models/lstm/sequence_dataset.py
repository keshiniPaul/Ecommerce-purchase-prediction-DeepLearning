import numpy as np


EVENT2ID = {'view': 1, 'addtocart': 2}
MAX_LEN = 20


def build_sequences(model_events, labels, session_ids, max_len=MAX_LEN):
    lab = labels[labels['session_id'].isin(
        session_ids)].set_index('session_id')
    ev = model_events[model_events['session_id'].isin(session_ids)].copy()
    ev = ev.sort_values(['session_id', 'datetime'])

    X_list, y_list = [], []
    for sid, group in ev.groupby('session_id', sort=False):
        if sid not in lab.index:
            continue
        events = group['event'].map(EVENT2ID).fillna(0).astype(int).tolist()
        if len(events) > max_len:
            events = events[-max_len:]
        padded = [0] * (max_len - len(events)) + events
        X_list.append(padded)
        y_list.append(int(lab.loc[sid, 'target']))

    return np.array(X_list, dtype=np.int32), np.array(y_list, dtype=np.float32)
