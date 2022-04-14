
def canAttendMeetings(intervals):

    starts =[]
    ends = []

    for sub_array in intervals:
        starts.append(sub_array[0])

        ends.append(sub_array[1])

    starts.sort()
    ends.sort()

    for i in range(len(starts) - 1):

        if starts[i + 1] < ends[i]:
            return False

    return True
