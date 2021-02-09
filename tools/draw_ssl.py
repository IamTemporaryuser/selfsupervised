import argparse
import json
import os
import numpy as np
import matplotlib as mpl
import sys
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
mpl.use('agg')

logprefix = "result"

def load_log(logname, keys, mode):
    def __find_json(dirpath):
        fns = os.listdir(dirpath)
        for fn in fns:
            if "json" in fn:
                return f'{dirpath}/{fn}'
    
    ss_path = f'{logprefix}/{logname}'
    linear_path = f'{ss_path}/{logname}_linear.py'

    ss_json = __find_json(ss_path)
    linear_json = __find_json(linear_path)

    def __load_json(jsonname, keys):
        with open(jsonname, "r") as f:
            lines = f.readlines()[1:]

        totalrecord = {}
        for l in lines:
            record = json.loads(l)
            if "mode" in record and record["mode"] == mode:
                for k in keys:
                    if k in record:
                        if k not in totalrecord:
                            totalrecord[k] = []
                        totalrecord[k].append(record[k])
        
        return totalrecord
    
    ss_records = __load_json(ss_json, keys)
    linear_records = __load_json(linear_json, keys)
    return ss_records, linear_records

def draw_pic(logdict, key, prefixname):
    colors = list(mcolors.TABLEAU_COLORS.keys())
    plt.figure(figsize=(9, 6))

    for name, records in logdict.items():
        data = records[key]
        total_len = len(data)

        x_axis = np.arange(0, total_len)
        plt.plot(x_axis, data, color=mcolors.TABLEAU_COLORS[colors[0]], linewidth=2.0, linestyle='-', label=name)
        colors.pop(0)

    plt.legend(loc="best", fontsize=15)

    plt.tick_params(labelsize=15)

    ax=plt.gca();#获得坐标轴的句柄
    ax.spines['bottom'].set_linewidth(1);###设置底部坐标轴的粗细

    plt.ylabel(key)
    plt.xlabel("epoches")
    plt.savefig('result/{}_{}.jpg'.format(prefixname, key), format='jpg', dpi=300)
    plt.show()
    plt.close('all')

parser = argparse.ArgumentParser()
parser.add_argument('--name', action='append', type=str)
parser.add_argument('--log', action='append', type=str)
parser.add_argument('--key', action='append', type=str)
parser.add_argument('--mode', default="val", type=str)

if __name__ == "__main__":
    args = parser.parse_args()
    if args.key is None or args.log is None:
        raise ValueError("need key (e.g. top-1) and log(e.g. bn.log.json) parameters.")

    if args.name is None:
        name_list = args.log
    else:
        assert len(args.name) == len(args.log)
        name_list = args.name

    sslogdict = {}
    linearlogdir = {}
    for i, logfile in enumerate(args.log):
        ssrecords, linearrecords = load_log(logfile, args.key, args.mode)
        sslogdict[name_list[i]] = ssrecords
        linearlogdir[name_list[i]] = linearrecords

    for i, keyname in enumerate(args.key):
        if keyname in sslogdict[name_list[0]]:
            draw_pic(sslogdict, keyname, "ss")
        
        if keyname in linearlogdir[name_list[0]]:
            draw_pic(linearlogdir, keyname, "ft")

