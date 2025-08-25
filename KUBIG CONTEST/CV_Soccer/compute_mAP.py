import copy
import json
import matplotlib.pyplot as plt
import numpy as np
import os

from tqdm import tqdm

RESULTS_FOLDER = "ResultSpottingPlotting2"
LIST_GAME_TEST_FILE = "listgame_Test_100.npy"
LABELS = ["background", "card", "substitution", "goal"]
LABELS_ROOT_PATH = "D:\\DeepSport\\SoccerNet-code\\data"
PREPARE_GROUND_TRUTH = True

ARGMAX = True
CENTER = True
NMS = True

# Names of files containing model predictions
PREDICTION_ROOTH_PATH = "predictions"
PREDICTIONS_FILENAMES = [
    # "RVLAD-Archi1-60sec-Spotting_Half_{}.npy",
    # "RVLAD-Archi2-60sec-Spotting_Half_{}.npy",
    # "RVLAD-Archi5-60sec-Spotting_Half_{}.npy",
    # "VLAD-Archi1-60sec-Spotting_Half_{}.npy",
    # "VLAD-Archi1-20sec-Spotting_Half_{}.npy",
    # "VLAD-Archi2-60sec-Spotting_Half_{}.npy",
    # "VLAD-Archi2-20sec-Spotting_Half_{}.npy",
    # "VLAD-Archi5-60sec-Spotting_Half_{}.npy",
    "VLAD-Archi5-20sec-Spotting_Half_{}.npy"
]

# Folders where results should be saved for each model
FOLDER_NAMES = [
    # os.path.join(RESULTS_FOLDER, "RVLAD-Archi1-60sec-Spotting"),
    # os.path.join(RESULTS_FOLDER, "RVLAD-Archi2-60sec-Spotting"),
    # os.path.join(RESULTS_FOLDER, "RVLAD-Archi5-60sec-Spotting"),
    # os.path.join(RESULTS_FOLDER, "VLAD-Archi1-60sec-Spotting"),
    # os.path.join(RESULTS_FOLDER, "VLAD-Archi1-20sec-Spotting"),
    # os.path.join(RESULTS_FOLDER, "VLAD-Archi2-60sec-Spotting"),
    # os.path.join(RESULTS_FOLDER, "VLAD-Archi2-20sec-Spotting"),
    # os.path.join(RESULTS_FOLDER, "VLAD-Archi5-60sec-Spotting"),
    os.path.join(RESULTS_FOLDER, "VLAD-Archi5-20sec-Spotting")
]

SPOT_DIR = "spots"
FORMAT_DIR = "formatted_spots"
MAP_DIR = "mAP"
GT_DIR = "Groundtruth"
GRAPH_FILENAME = "TEST_GRAPHS_{}_timestamp_score.npy"

THRESHOLD_RANGE = range(95, 0, -5)
DELTA_RANGE = range(60, 0, -5)

# OS functions
def create_directory(dir_name):
    if not os.path.exists(dir_name) or not os.path.isdir(dir_name):
            os.mkdir(dir_name)

def write_json(data, filename):
    with open(filename, 'w') as fp:
        json.dump(data, fp)

def load_json(filename):
    with open(filename, 'r') as f:
        return json.load(f)

# Plot functions
def plot_pr_curve(recall, precision, interpolated, AP_r, AP, filename):
    fig, ax = plt.subplots()

    ax.plot(recall, precision, label="precision")

    for i in range(len(recall)-1):
        ax.plot((recall[i], recall[i]), (interpolated[i], interpolated[i+1]), color='orange')
        ax.plot((recall[i], recall[i+1]), (interpolated[i+1], interpolated[i+1]), color='orange')
    ax.plot((recall[i], recall[i+1]), (interpolated[i+1], interpolated[i+1]), color='orange', label="interpolated precision")

    plt.scatter(0, AP_r[0], color='white', label="AP : {}".format(AP))
    for i, index in enumerate([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]):
        plt.scatter(index, AP_r[i], color='red')

    lgd = ax.legend(loc="upper right")
    plt.savefig(filename)
    plt.close(fig)
    plt.clf()

def plot_average_map(threshold_l, lines, labels, avgMAP, colors, filename, label_x, label_y, has_legend=True, step=20):
    fig, ax = plt.subplots()

    for index, line in enumerate(lines):
        sc = "%.1f"%(100*avgMAP[index])
        ax.plot(threshold_l, line, label="{} (AUC={}%)".format(labels[index], sc), color=colors[index])
        # plt.rcParams['hatch.color'] = colors[index]
        plt.rcParams.update({'hatch.color': colors[index]})
        ax.fill_between(threshold_l, line, alpha=0, linewidth=10, hatch="\\"*(3-index))


    ax.set_ylabel(label_y)
    ax.set_xlabel(label_x)

    plt.xticks(np.arange(0.0, max(threshold_l)+1, step))

    if has_legend:
        lgd = ax.legend(loc="center left", bbox_to_anchor=(1, 0.5))
        plt.savefig(filename, bbox_extra_artists=(lgd,), bbox_inches="tight")
    else:
        plt.savefig(filename)
    plt.close(fig)
    plt.clf()

# Groundtruth functions
def prepare_ground_truth():
    # Set in separate files the timestamp where an action happened
    create_directory(GT_DIR)
    for i, game in enumerate(np.load(os.path.join(LIST_GAME_TEST_FILE))):
        labels_path = os.path.join(LABELS_ROOT_PATH, game, "Labels.json")
        labels = load_json(labels_path)

        halves = {}
        halves[0] = []
        halves[1] = []

        for event in labels["annotations"]:
            Half = int(event['gameTime'][0]) - 1
            Time_Minute = int(event['gameTime'][-5:-3])
            Time_Second = int(event['gameTime'][-2:]) + 60 * Time_Minute

            if ("card" in event['label']): label = 0
            elif ("subs" in event['label']): label = 1
            elif ("soccer" in event['label']): label = 2

            halves[Half].append(
                {
                    "timestamp": Time_Second,
                    "label": label
                }
            )

        for h in range(2):
            formatted_data = []
            data_half = halves[h]
            for l in range(3):
                data_l = [x['timestamp'] for x in data_half if x['label'] == l]
                data_l.sort()
                formatted_data.append(np.array(data_l))

            id_game = 2 * i + h
            np.save(
                os.path.join(GT_DIR, GRAPH_FILENAME.format(id_game)),
                formatted_data,
                allow_pickle=True
            )

def get_ground_truth():
    ground_truth = []
    for game in range(200):
        game_gt = np.load(os.path.join(GT_DIR, GRAPH_FILENAME.format(game)), allow_pickle=True)
        ground_truth.append(game_gt)

    return ground_truth

# Average-mAP
def computeAP(label, all_spotted, ground_truth, output_dir, delta, to_plot=True):
    spotted_actions = []
    gt = []
    TP = 0
    FP = 0
    FN = 0
    P = 0
    Precision = [1]
    Recall = [0]
    all_spots = []
    for game in range(200):
        game_gt = copy.deepcopy(ground_truth[game][label])
        taken = np.zeros(len(game_gt))
        # Positive examples
        P += len(game_gt)
        game_spotted = all_spotted[game][label]

        for spot in game_spotted:
            found = False
            for pos, gt in enumerate(game_gt):
                if taken[pos] == 0 and spot[0] >= gt - delta/2 and spot[0] < gt + delta/2:
                    found = True
                    taken[pos] = 1
                    break

            all_spots.append([spot[0], spot[1], found])
    FN = P
    all_spots.sort(key=lambda x : x[1], reverse=True)

    for spot in all_spots:
        if spot[2]:
            TP += 1
            FN -= 1
        else:
            FP += 1

        Precision.append(TP / (TP + FP))
        Recall.append(TP / P)

    Precision.append(0)
    Recall.append(1)
    interpolated = copy.deepcopy(Precision)


    i = len(Recall)-2
    while i >= 0:
        if interpolated[i+1] > interpolated[i]:
            interpolated[i] = interpolated[i+1]
        i -= 1

    R = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    AP_r = []
    i = 0
    for r in R:
        while True:
            if Recall[i] <= r and r <= Recall[i+1]:
                AP_r.append(interpolated[i+1])
                break
            i+=1

    AP = sum(AP_r) / 11

    if to_plot:
        plotname = os.path.join(output_dir, "label_{}_delta_{}.png".format(label, delta))
        plot_pr_curve(Recall, Precision, interpolated, AP_r, AP, plotname)

    return AP, Precision[-1], Recall[-1], TP, FN, FP

def mAP(thresh, delta, ground_truth, prediction_dir, output_dir):
    predictions = []
    for game_id in range(200):
        game_pred = np.load(os.path.join(prediction_dir, GRAPH_FILENAME.format(game_id)), allow_pickle=True)
        predictions.append(game_pred)


    all_spotted = []
    for game_id in range(200):
        game_pred = predictions[game_id]
        spotted_in_game = []
        for label in range(3):
            spot_timestamps = np.argwhere(game_pred[label] > 0)[:,0]
            spot_l = []
            for spot in spot_timestamps:
                spot_l.append( (spot, game_pred[label][spot]) )

            spot_l.sort(key=lambda x : x[1], reverse=True)
            spotted_in_game.append(spot_l)
        all_spotted.append(spotted_in_game)

    # Compute AP for each class
    all_APs = []
    for label in range(3):
        AP, precision, recall, TP, FN, FP = computeAP(label, all_spotted, ground_truth, output_dir, delta, to_plot=False)
        all_APs.append(AP)

    return np.mean(all_APs)

def average_mAP(i):
    map_dir = os.path.join(FOLDER_NAMES[i], MAP_DIR)
    create_directory(map_dir)

    format_dir = os.path.join(FOLDER_NAMES[i], FORMAT_DIR)

    gt = get_ground_truth()

    maps = dict()

    if ARGMAX:
        print("Start average-mAP for argmax")
        for thresh in THRESHOLD_RANGE:
            maps[thresh] = dict()
            prediction_dir = os.path.join(format_dir, "Argmax_Thresh_{}".format(thresh))
            output_dir = os.path.join(map_dir, "Argmax_Thresh_{}".format(thresh))
            create_directory(output_dir)
            maps[thresh]['argmax'] = [] # stock mAP for each delta tolerance
            for delta in DELTA_RANGE:
                maps[thresh]['argmax'].append(mAP(thresh, delta, gt, prediction_dir, output_dir))

    if CENTER:
        print("Start average-mAP for center")
        for thresh in THRESHOLD_RANGE:
            if thresh not in maps:
                maps[thresh] = dict()
            prediction_dir = os.path.join(format_dir, "Center_Thresh_{}".format(thresh))
            output_dir = os.path.join(map_dir, "Center_Thresh_{}".format(thresh))
            create_directory(output_dir)
            maps[thresh]['center'] = [] # stock mAP for each delta tolerance
            for delta in DELTA_RANGE:
                maps[thresh]['center'].append(mAP(thresh, delta, gt, prediction_dir, output_dir))

    if NMS:
        print("Start average-mAP for nms")
        for thresh in THRESHOLD_RANGE:
            if thresh not in maps:
                maps[thresh] = dict()
            prediction_dir = os.path.join(format_dir, "NMS_Thresh_{}".format(thresh))
            output_dir = os.path.join(map_dir, "NMS_Thresh_{}".format(thresh))
            create_directory(output_dir)
            maps[thresh]['nms'] = [] # stock mAP for each delta tolerance
            for delta in DELTA_RANGE:
                maps[thresh]['nms'].append(mAP(thresh, delta, gt, prediction_dir, output_dir))

    print("Start plotting")
    for thresh in maps.keys():
        lines = []
        labels = []
        avgMAP = []
        if ARGMAX:
            lines.append(maps[thresh]['argmax'])
            labels.append("argmax")
            avgMAP.append(sum(maps[thresh]['argmax'])/len(maps[thresh]['argmax']))
        if CENTER:
            lines.append(maps[thresh]['center'])
            labels.append("center")
            avgMAP.append(sum(maps[thresh]['center'])/len(maps[thresh]['center']))
        if NMS:
            lines.append(maps[thresh]['nms'])
            labels.append("nms")
            avgMAP.append(sum(maps[thresh]['nms'])/len(maps[thresh]['nms']))

        plot_filename = os.path.join(map_dir, "average_mAP_Thresh_{}.png".format(thresh))
        plot_average_map(DELTA_RANGE, lines, labels, avgMAP, ["r", "g", "b"], plot_filename, "Spotting tolerance in delta seconds", "mAP")

# Action spotting
def format_spot(predictions, spots):
    n_seconds = predictions.shape[0]
    formatted_spots = -1 * np.ones((3, n_seconds))

    for spot in spots:
        l = LABELS.index(spot['label']) - 1
        formatted_spots[l][spot['segment']] = spot['score']

    return formatted_spots

def get_spot_from_argmax(array, thresh):
    diff1 = np.insert(array, 0, 0)
    diff2 = np.append(array, 0)
    t_init = np.argwhere( (diff1 < thresh) & (diff2 >= thresh))[:,0]
    t_end = np.argwhere( (diff1 > thresh) & (diff2 <= thresh))[:,0]
    score = np.zeros(len(t_end))
    t_max = np.zeros(len(t_end))

    for i in range(len(t_end)):
        t_max[i] = t_init[i] + np.argmax(array[t_init[i]:t_end[i]])
        score[i] = array[int(t_max[i])]
    return np.transpose([t_max, score])

def spot_argmax(fi, spot_dir, format_dir, thresh=50/100):
    data = {}
    data['results'] = {}
    for i, game in enumerate(np.load(os.path.join(LIST_GAME_TEST_FILE))):
        for half in [1,2]:
            keyGame=os.path.split(game)[1] + "_Half_" + str(half) + ".npy"
            predictions = np.load(os.path.join(PREDICTION_ROOTH_PATH, game, PREDICTIONS_FILENAMES[fi].format(half)))
            data['results'][keyGame] = []
            for l in range(1,4):
                spots = get_spot_from_argmax(predictions[:,l], thresh)
                for spot in spots:
                    result = {
                        "label": LABELS[l],
                        "segment": int(spot[0]),
                        "score": spot[1]
                    }
                    data['results'][keyGame].append(result)
            formatted_spots = format_spot(predictions, data['results'][keyGame])
            np.save(
                os.path.join(PREDICTION_ROOTH_PATH, game, "Argmax_Formatted_Spot_Half_{}.npy".format(half)),
                formatted_spots
            )
            id_game = 2 * i + half-1
            local_dir = os.path.join(format_dir, "Argmax_Thresh_{}".format(int(thresh*100)))
            create_directory(local_dir)
            np.save(
                os.path.join(local_dir, GRAPH_FILENAME.format(id_game)),
                formatted_spots,
                allow_pickle=True
            )

    write_json(data, os.path.join(spot_dir, "Argmax_Thresh_{}.json".format(int(thresh*100))))

def get_spot_from_Center(array, thresh=0.5):
    diff1 = np.insert(array,0,0)
    diff2 = np.append(array,0)
    t_init = np.argwhere( (diff1<thresh) & (diff2>=thresh))[:,0]
    t_end  = np.argwhere( (diff1>thresh) & (diff2<=thresh))[:,0]
    score = np.zeros(len(t_end))
    t_spot = np.zeros(len(t_end))
    for i in range(len(t_end)):
        score[i] = np.mean(array[t_init[i]:t_end[i]])
        t_spot[i] = (t_init[i] + t_end[i]) / 2
        score[i] = array[int(t_spot[i])]
    return  np.transpose([t_spot, score])

def spot_center(fi, spot_dir, format_dir, thresh):
    data = {}
    data['results'] = {}
    for i, game in enumerate(np.load(os.path.join(LIST_GAME_TEST_FILE))): # for each game
        for half in [1,2]: # for each half
            keyGame=os.path.split(game)[1] + "_Half_" + str(half) + ".npy"
            predictions = np.load(os.path.join(PREDICTION_ROOTH_PATH, game, PREDICTIONS_FILENAMES[fi].format(half)))
            data['results'][keyGame] = []
            for l in range(1,4): # for each label
                spots = get_spot_from_Center(predictions[:,l], thresh)
                for spot in spots:
                    result = {
                        "label": LABELS[l],
                        "segment": int(spot[0]),
                        "score": spot[1]
                    }
                    data['results'][keyGame].append(result)
            formatted_spots = format_spot(predictions, data['results'][keyGame])
            np.save(
                os.path.join(PREDICTION_ROOTH_PATH, game, "Center_Formatted_Spot_Half_{}.npy".format(half)),
                formatted_spots
            )
            id_game = 2 * i + half-1
            local_dir = os.path.join(format_dir, "Center_Thresh_{}".format(int(thresh*100)))
            create_directory(local_dir)
            np.save(
                os.path.join(local_dir, GRAPH_FILENAME.format(id_game)),
                formatted_spots,
                allow_pickle=True
            )

    write_json(data, os.path.join(spot_dir, "Center_Thresh_{}.json".format(int(thresh*100))))

def get_spot_from_NMS(array, window=60, thresh=0.5):
    res = np.empty(np.size(array), dtype=bool)

    for i, value  in enumerate(array):
        if (i<=window/2 or i>=np.size(array)-window/2):
            res[i] = False
            continue

        if value >= np.max(array[i+1:(i+int(window/2))]) and \
            value > np.max(array[(i-int(window/2)):i]) and \
            value > thresh:
            res[i] = True
        else:
            res[i] = False


    max_values = array[res==True]
    indexes = np.arange(np.size(array))[res==True]
    return indexes, max_values

def spot_nms(fi, spot_dir, format_dir, thresh=50/100):
    data = {}
    data['results'] = {}
    for i, game in enumerate(np.load(os.path.join(LIST_GAME_TEST_FILE))):
        for half in [1,2]:
            keyGame=os.path.split(game)[1] + "_Half_" + str(half) + ".npy"
            predictions = np.load(os.path.join(PREDICTION_ROOTH_PATH, game, PREDICTIONS_FILENAMES[fi].format(half)))
            data['results'][keyGame] = []
            for l in range(1,4):
                indexes, scores = get_spot_from_NMS(predictions[:,l], thresh=thresh)
                for j in range(len(indexes)):
                    spot = indexes[j]
                    score = scores[j]

                    result = {
                        "label": LABELS[l],
                        "segment": int(spot),
                        "score": float(score)
                    }
                    data['results'][keyGame].append(result)
            formatted_spots = format_spot(predictions, data['results'][keyGame])
            np.save(
                os.path.join(PREDICTION_ROOTH_PATH, game, "NMS_Formatted_Spot_Half_{}.npy".format(half)),
                formatted_spots
            )
            id_game = 2 * i + half-1
            local_dir = os.path.join(format_dir, "NMS_Thresh_{}".format(int(thresh*100)))
            create_directory(local_dir)
            np.save(
                os.path.join(local_dir, GRAPH_FILENAME.format(id_game)),
                formatted_spots,
                allow_pickle=True
            )

    write_json(data, os.path.join(spot_dir, "NMS_Thresh_{}.json".format(int(thresh*100))))

def spot_actions(i):
    """
    Spot the actions for model i
    """
    spot_dir = os.path.join(FOLDER_NAMES[i], SPOT_DIR)
    create_directory(spot_dir)
    format_dir = os.path.join(FOLDER_NAMES[i], FORMAT_DIR)
    create_directory(format_dir)

    if ARGMAX:
        print("Start spotting - argmax")
        for thresh in tqdm(THRESHOLD_RANGE, desc="Argmax threshold"):
            spot_argmax(i, spot_dir, format_dir, thresh=thresh/100)

    if CENTER:
        print("Start spotting - center")
        for thresh in tqdm(THRESHOLD_RANGE, desc="Center threshold"):
            spot_center(i, spot_dir, format_dir, thresh=thresh/100)

    if NMS:
        print("Start spotting - NMS")
        for thresh in tqdm(THRESHOLD_RANGE, desc="NMS threshold"):
            spot_nms(i, spot_dir, format_dir, thresh=thresh/100)

if __name__ == "__main__":
    create_directory(RESULTS_FOLDER)

    if PREPARE_GROUND_TRUTH:
        prepare_ground_truth()

    for i in range(len(PREDICTIONS_FILENAMES)): # For each model
        create_directory(FOLDER_NAMES[i])
        print("Start working on {}".format(FOLDER_NAMES[i]))
        spot_actions(i)
        average_mAP(i)
