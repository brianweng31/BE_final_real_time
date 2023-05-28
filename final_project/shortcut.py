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
                if i != 'Key3':
                    if item[i] == 'Control':
                        keys.append('ctrl')
                    elif item[i] == 'Shift':
                        keys.append('shift')
                    elif item[i] == 'Option':
                        keys.append('option')
                    elif item[i] == ' Command':
                        keys.append('command')
                    else:
                        keys.append('')
                else: # i == 'Key3':
                    keys.append(item[i].lower())

        shortcut_dict[item['Gesture']] = keys

    return shortcut_dict

def shortcut(shortcut_dict, result):
    print(shortcut_dict[result])
    if(shortcut_dict[result][2] != ''):
        pyautogui.hotkey(shortcut_dict[result][0],shortcut_dict[result][1], shortcut_dict[result][2])

