def sort_interval(interval_list):
    # sorting rule to pick the first value in the list rep by lambda x
    
    interval_list.sort(key = lambda x : x[0])
    print(f"interval list: {interval_list}")

    merged = [interval_list[0]]

    for current in interval_list[1:]:
        print(f"current interval list selected: {current}")
        if current[0] <= merged[-1][1]:
            merged[-1][1] = max(current[1], merged[-1][1])
            print(f"updated merged list{merged}")
        else:
            merged.append(current)
    return merged

intervals = [[1,3],[2,6],[8,10],[15,18], [8,10]]
print(sort_interval(intervals))