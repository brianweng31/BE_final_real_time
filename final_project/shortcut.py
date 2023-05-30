import json
import pyautogui

def read_json(file):
    with open(file, 'r') as f:
        data = json.load(f)

    shortcut_dict = {}
    for item in data:
        keys = []
        for i in item:
            if i != 'Gesture':
                keys.append(item[i].lower())

        shortcut_dict[item['Gesture']] = keys

    return shortcut_dict

def shortcut(shortcut_dict, result):
    if(shortcut_dict[result][2] != ''):
        pyautogui.hotkey(shortcut_dict[result][0],shortcut_dict[result][1], shortcut_dict[result][2],shortcut_dict[result][2])

