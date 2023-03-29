# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np
import ast
from collections import Counter
import time
import sympy


def advent1():
    file = open('input1.txt')
    measurements = []
    for line in file:
        measurement = int(line.strip('\n'))
        measurements.append(measurement)

    count = 0
    for i in range(0, len(measurements) - 1):
        if measurements[i + 1] > measurements[i]:
            count += 1

    print('Nof increases: ', count)

    count = 0
    for i in range(0, len(measurements) - 3):
        if sum(measurements[i:i+3]) < sum(measurements[i+1:i+4]):
            count += 1

    print('Sliding increases: ', count)
    
def str2nmbr(string):
    nmbr = 0
    for c in range(0, len(string)):
        if string[len(string) - c - 1] == '1':
            nmbr = nmbr + 2**c
    return nmbr


def advent3():
    file3 = open('input3.txt')
    arr3 = []
    for line in file3:
        arr3.append(str(line))

    gamma = []
    epsilon = []
    for c in range(0, len(arr3[0])-1):
        sum = 0
        for i in range(0, len(arr3)):
            sum = sum + int(arr3[i][c])
        if sum > len(arr3) / 2:
            gamma.append('1')
            epsilon.append('0')
        else:
            gamma.append('0')
            epsilon.append('1')
    print(gamma)
    print(epsilon)

    print(str2nmbr(gamma))
    print(str2nmbr(epsilon))
    print(str2nmbr(gamma)*str2nmbr(epsilon))


def advent3_2(a, b):
    file3 = open('input3.txt')
    arr3 = []
    for line in file3:
        arr3.append(str(line))

    strlen = len(arr3[0]) - 1
    for c in range(0, strlen):
        count = 0
        for i in range(0, len(arr3)):
            count = count + int(arr3[i][c])
        if count >= len(arr3) / 2:
            slet = a
        else:
            slet = b

        arr3 = [e for e in arr3 if e[c] == slet]
        if len(arr3) == 1:
            break

        #print(arr3)
        #print(len(arr3))
    return str2nmbr(arr3[0].strip('\n'))


def advent4():
    file4 = open('input4.txt')
    arr4 = str(file4.readline())
    numbers = arr4.split(',')
    numbers = [int(number) for number in numbers]
    #print(numbers)

    boards = []
    while file4.readline():
        board = np.zeros([5, 5], np.int)
        for r in range(0, 5):
            row = file4.readline()
            row = row.split()

            nrow = [int(number) for number in row]
            board[r, :] = nrow
        #print(board)
        boards.append(board)

    count = 0
    usedboards = []
    first = True
    scores = []
    for number in numbers:
        for bnr in range(0, len(boards)):
            boards[bnr], score = check_board(boards[bnr], number)
            if score > 0:
                count = count + 1
                usedboards.append(bnr)
                scores.append(score)
                if first:
                    print('First: ', score, number, bnr)
                    first = False
                boards[bnr] = np.zeros([5, 5])

    print('Last: ', scores[-1])
    #print(len(boards), count, len(numbers))
    leftboards = set([b  for b in range(0, 100)]) - set(usedboards)
    #print(leftboards)


def check_board(board, number):
    #print(board)
    board = (board != number) * board
    #print(board)
    score = 0
    for rc in range(0, 5):
        if board[rc, :].sum() == 0 or board[:, rc].sum() == 0:
            #print('bingo!')
            score = board.sum()*number

    return board, score


def advent5():
    file5 = open('input5.txt')
    arr5 = []
    for line in file5:
        sl = (str(line).strip('\n').split(' -> '))
        coords = []
        for p in sl:
            coords.append([int(e) for e in p.split(',')])
        arr5.append(coords)

    #print(len(arr5))
    rowscols = [e for e in arr5 if (e[0][0] == e[1][0] or e[0][1] == e[1][1])]
    #print((rowscols))

    map = np.zeros([1000, 1000], np.int)
    for vent in rowscols:
        map = insert_vent(map, vent)
        #print((map > 1).sum())
    print((map > 1).sum())


def insert_vent(map, coords):
    if coords[0][0] == coords[1][0]:
        i = coords[0][0]
        start_val = min(coords[0][1], coords[1][1])
        end_val = max(coords[0][1], coords[1][1])
        for j in range(start_val, end_val + 1):
            map[i, j] = map[i, j] + 1
    elif coords[0][1] == coords[1][1]:
        j = coords[0][1]
        start_val = min(coords[0][0], coords[1][0])
        end_val = max(coords[0][0], coords[1][0])
        for i in range(start_val, end_val + 1):
            map[i, j] = map[i, j] + 1
    else:
        map = insert_vent_2(map, coords)

    #print(map.sum())
    return map


def advent5_2():
    file5 = open('input5.txt')
    arr5 = []
    for line in file5:
        sl = (str(line).strip('\n').split(' -> '))
        coords = []
        for p in sl:
            coords.append([int(e) for e in p.split(',')])
        arr5.append(coords)

    map = np.zeros([1000, 1000], np.int)
    for vent in arr5:
        map = insert_vent(map, vent)
        #print((map > 1).sum())
    print((map > 1).sum())


def insert_vent_2(map, coords):
    start_j = min(coords[0][1], coords[1][1])
    end_j = max(coords[0][1], coords[1][1])
    start_i = min(coords[0][0], coords[1][0])
    end_i = max(coords[0][0], coords[1][0])
    if (end_j - start_j) != (end_i - start_i):
        print("Not a diagonal!")

    k = (coords[1][1] - coords[0][1]) / (coords[1][0] - coords[0][0])
    if k > 0:
        for index in range(0, end_j - start_j + 1):
            map[start_i + index, start_j + index] = map[start_i + index, start_j + index] + 1
    else:
        for index in range(0, end_j - start_j + 1):
            map[start_i + index, end_j - index] = map[start_i + index, end_j - index] + 1
    #print(map.sum())
    return map


def advent2():
    file2 = open('input2.txt')
    arr2 = []
    for line in file2:
        sl = (str(line).strip('\n').split(' '))
        arr2.append((sl[0], int(sl[1])))

    x_pos = 0
    y_pos = 0
    aim = 0
    for com in arr2:
        if com[0] == 'forward':
            x_pos = x_pos + com[1]
            y_pos = y_pos + aim*com[1]
        elif com[0] == 'up':
            aim = aim - com[1]
        elif com[0] == 'down':
            aim = aim + com[1]

    print(x_pos, y_pos)
    print(x_pos * y_pos)


def advent6():
    file6 = open('input6.txt')
    arr6 = file6.readline().split(',')
    arr6 = [int(s) for s in arr6]

    prevlen = len(arr6)
    for day in range(0, 80):
        spawn = []
        for i in range(0, len(arr6)):
            if arr6[i] > 0:
                arr6[i] = arr6[i] - 1
            elif arr6[i] == 0:
                arr6[i] = 6
                spawn.append(8)
            else:
                print('incorrect negative age encountered')
        for s in spawn:
            arr6.append(s)
        #print(day, len(arr6), len(arr6)/prevlen)
        prevlen = len(arr6)
    print(len(arr6))


def advent6_2():
    file6 = open('input6.txt')
    arr6 = file6.readline().split(',')
    arr6 = [int(s) for s in arr6]

    zero = [e for e in arr6 if e == 0]
    one = [e for e in arr6 if e == 1]
    two = [e for e in arr6 if e == 2]
    three = [e for e in arr6 if e == 3]
    four = [e for e in arr6 if e == 4]
    five = [e for e in arr6 if e == 5]
    six = [e for e in arr6 if e == 6]

    numbers = []
    increase = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    inc0 = 0
    inc1 = 0
    numbers.append(len(zero))
    numbers.append(len(one))
    numbers.append(len(two))
    numbers.append(len(three))
    numbers.append(len(four))
    numbers.append(len(five))
    numbers.append(len(six))
    numbers.append(0)
    numbers.append(0)

    #print(numbers)
    print(0, sum(numbers))

    for day in range(6, 256, 7):
        increase[0] = numbers[7]
        increase[1] = numbers[8]
        increase[2] = numbers[0]
        increase[3] = numbers[1]
        increase[4] = numbers[2]
        increase[5] = numbers[3]
        increase[6] = numbers[4]
        increase[7] = numbers[5]
        increase[8] = numbers[6]

        inc0 = numbers[7]
        inc1 = numbers[8]
        for i in range(0, 9):
            numbers[i] += increase[i]
        numbers[7] -= inc0
        numbers[8] -= inc1
        #print(numbers)
        #print(day, sum(numbers))

    print(sum(numbers) + sum(numbers[0:4]))


def sum_to_n(n):
    s = 0
    for i in range(0, n + 1):
        s = s + i

    return s


def advent7():
    file7 = open('input7.txt')
    arr7 = file7.readline().split(',')
    arr7 = [int(s) for s in arr7]

    minv = min(arr7)
    maxv = max(arr7)

    print(minv, maxv)

    cost = 100000000000
    best_pos = 1000000000
    narr = np.array(arr7)
    for pos in range(minv, maxv + 1):
        dist = np.abs(pos - narr)
        fuel = dist
        cost = min(cost, fuel.sum())

    print('1: ', cost)
    
    cost = 100000000000
    best_pos = 1000000000
    narr = np.array(arr7)
    for pos in range(minv, maxv + 1):
        dist = np.abs(pos - narr)
        fuel = np.array([sum_to_n(e) for e in dist])
        cost = min(cost, fuel.sum())

    print('2: ', cost)


def advent8():
    file8 = open('input8.txt')
    arr8 = []
    for line in file8:
        arr8.append(str(line).strip('\n').split('|'))

    total = 0
    ulen_count = 0
    for el in arr8:
        cf = []
        acf = []
        bcdf = []
        abcdefg = []
        sixes = []
        codes = el[0].split()
        for code in codes:
            if len(code) == 2:
                for l in code:
                    cf.append(l)
            elif len(code) == 4:
                for l in code:
                    bcdf.append(l)
            elif len(code) == 3:
                for l in code:
                    acf.append(l)
            elif len(code) == 7:
                for l in code:
                    abcdefg.append(l)
            elif len(code) == 6:
                six = []
                for l in code:
                    six.append(l)
                sixes.append(six)
        a = list(set(acf) - set(cf))
        bd = list(set(bcdf) - set(cf))
        eg = list(set(abcdefg) - set(bcdf) - set(acf))
        for six in sixes:
            if not set(eg).issubset(six):
                abcdfg = six
                g = list(set(abcdfg) - set(acf) - set(bd))
                e = list(set(eg) - set(g))
            elif set(bd).issubset(six):
                abdefg = six
                f = list(set(abdefg) - set(bd) - set(eg) - set(a))
                ac = list(set(acf) - set(f))
            else:
                abcefg = six
                b = list(set(abcefg) - set(acf) - set(eg))
                d = list(set(bd) - set(b))
        c = list(set(acf) - set(a) - set(f))
        if not set(a).union(set(b).union(set(c).union(set(d).union(set(e).union(set(f).union(set(g))))))) == set(['a','b','c','d','e','f','g']):
            print("Horrreuer!")
            break
        #print(a, b, c, d, e, f, g)


        codes2 = el[1].split()
        nmbr = str()
        for code2 in codes2:
            if len(code2) in [2, 3, 4, 7]:
                ulen_count += 1
            key = set()
            for l in code2:
                key.add(l)
            if key == set([a[0], b[0], c[0], e[0], f[0], g[0]]):
                nmbr += '0'
            elif key == set([c[0], f[0]]):
                nmbr += '1'
            elif key == set([a[0], c[0], d[0], e[0], g[0]]):
                nmbr += '2'
            elif key == set([a[0], c[0], d[0], f[0], g[0]]):
                nmbr += '3'
            elif key == set([b[0], c[0], d[0], f[0]]):
                nmbr += '4'
            elif key == set([a[0], b[0], d[0], f[0], g[0]]):
                nmbr += '5'
            elif key == set([a[0], b[0], d[0], e[0], f[0], g[0]]):
                nmbr += '6'
            elif key == set([a[0], c[0], f[0]]):
                nmbr += '7'
            elif key == set([a[0], b[0], c[0], d[0], e[0], f[0], g[0]]):
                nmbr += '8'
            elif key == set([a[0], b[0], c[0], d[0], f[0], g[0]]):
                nmbr += '9'
        #print(int(nmbr))
        total += int(nmbr)

    print(ulen_count)
    print('Total: ', total)


def within(point):
    if point[0] > 0 and point[0] < 100 and point[1] > 0 and point[1] < 100:
        return True
    else:
        return False


def find_basin_size(bottom, i, j):
    basin = []
    basin.append([i, j])
    try_again = True
    while try_again:
        try_again = False
        addp = []
        for point in basin:
            up = [point[0] - 1, point[1]]
            if within(up) and up not in basin and up not in addp and bottom[up[0], up[1]] < 9:
                addp.append(up)
                try_again = True
            down = [point[0] + 1, point[1]]
            if within(down) and down not in basin and down not in addp and bottom[down[0], down[1]] < 9:
                addp.append(down)
                try_again = True
            left = [point[0], point[1] - 1]
            if within(left) and left not in basin and left not in addp and bottom[left[0], left[1]] < 9:
                addp.append(left)
                try_again = True
            right = [point[0], point[1] + 1]
            if within(right) and right not in basin and right not in addp and bottom[right[0], right[1]] < 9:
                addp.append(right)
                try_again = True
        for a in addp:
            basin.append(a)

    return len(basin)


def danger_check(bottom, i, j):
    if i == 0 and j == 0:
        if bottom[i, j] < bottom[i+1, j] and bottom[i, j] < bottom[i, j+1]:
            return bottom[i, j] + 1
    elif i == 0 and j == 99:
        if bottom[i, j] < bottom[i+1, j] and bottom[i, j] < bottom[i, j-1]:
            return bottom[i, j] + 1
    elif i == 99 and j == 0:
        if bottom[i, j] < bottom[i-1, j] and bottom[i, j] < bottom[i, j+1]:
            return bottom[i, j] + 1
    elif i == 99 and j == 99:
        if bottom[i, j] < bottom[i-1, j] and bottom[i, j] < bottom[i, j-1]:
            return bottom[i, j] + 1
    elif i == 0:
        if bottom[i, j] < bottom[i+1, j] and bottom[i, j] < bottom[i, j+1] and bottom[i, j] < bottom[i, j-1]:
            return bottom[i, j] + 1
    elif i == 99:
        if bottom[i, j] < bottom[i-1, j] and bottom[i, j] < bottom[i, j+1] and bottom[i, j] < bottom[i, j-1]:
            return bottom[i, j] + 1
    elif j == 0:
        if bottom[i, j] < bottom[i+1, j] and bottom[i, j] < bottom[i, j+1] and bottom[i, j] < bottom[i-1, j]:
            return bottom[i, j] + 1
    elif j == 99:
        if bottom[i, j] < bottom[i-1, j] and bottom[i, j] < bottom[i, j-1] and bottom[i, j] < bottom[i+1, j]:
            return bottom[i, j] + 1
    else:
        if bottom[i, j] < bottom[i-1, j] and bottom[i, j] < bottom[i, j-1] and bottom[i, j] < bottom[i+1, j] and bottom[i, j] < bottom[i, j+1]:
            return bottom[i, j] + 1
    return 0


def advent9():
    file9 = open('input9.txt')
    bottom = np.zeros([100, 100], np.int)
    r_nr = 0
    for line in file9:
        line = line.strip('\n')
        row = [int(n) for n in line]
        bottom[r_nr, :] = row
        r_nr += 1

    danger = 0
    bottom_points = []
    for i in range(0, 100):
        for j in range(0, 100):
            how_dangerous = danger_check(bottom, i, j)
            if  how_dangerous > 0:
                danger += how_dangerous
                bottom_points.append([i, j])

    basin_sizes = [9]
    for point in bottom_points:
        basin_size = find_basin_size(bottom, point[0], point[1])
        basin_sizes.append(basin_size)

    basin_sizes.sort()
    print(basin_sizes[-3:-1])
    print(danger)


def check_line(line):
    cost = dict()
    cost[')'] = 3
    cost[']'] = 57
    cost['}'] = 1197
    cost['>'] = 25137
    openers = ['(', '[', '{', '<']
    closers = [')', ']', '}', '>']
    pairs = dict()
    pairs[')'] = '('
    pairs[']'] = '['
    pairs['}'] = '{'
    pairs['>'] = '<'
    so_far = []
    for e in line:
        if e in openers:
            so_far.append(e)
        else:
            if len(so_far) == 0:
                return cost[e]
            if not so_far.pop() == pairs[e]:
                return cost[e]

    return 0


def complete_cost(line):
    cost = dict()
    cost[')'] = 1
    cost[']'] = 2
    cost['}'] = 3
    cost['>'] = 4
    openers = ['(', '[', '{', '<']
    closers = [')', ']', '}', '>']
    pairs = dict()
    pairs['('] = ')'
    pairs['['] = ']'
    pairs['{'] = '}'
    pairs['<'] = '>'
    so_far = []
    for e in line:
        if e in openers:
            so_far.append(e)
        else:
            so_far.pop()

    score = 0
    for e in reversed(so_far):
        score = score*5 + cost[pairs[e]]

    return score


def advent10():
    file10 = open('input10.txt')
    cost = 0
    incomplete = []
    for line in file10:
        line = line.strip('\n')
        incr = check_line(line)
        if incr > 0:
            cost += check_line(line)
        else:
            incomplete.append(line)
    print(cost)

    comcosts = []
    for line in incomplete:
        comcosts.append(complete_cost(line))
    comcosts.sort()
    print(comcosts[0])
    print(comcosts[len(comcosts)//2])
    print(comcosts[len(comcosts)-1])


def okind(i, j, ii, jj):
    if i + ii < 0 or i + ii > 9:
        return False
    if j + jj < 0 or j + jj > 9:
        return False
    return True


def incrneighbors(floct):
    incr = np.zeros([10, 10], np.int)
    for i in range(0, 10):
        for j in range(0, 10):
            if floct[i, j]:
                for ii in range(-1, 2):
                    for jj in range(-1, 2):
                        if okind(i, j, ii, jj):
                            incr[i + ii, j + jj] += 1
    return incr


def update_octol(octol_in):
    octol = octol_in + 1
    all_f = (octol > 10)

    for its in range(0, 100):
        floct = (octol > 9)
        if floct.sum() == 0:
            break
        octol += incrneighbors(floct)
        #print(floct)
        all_f += floct
        octol[floct] = 0
        #print(all_f)

    octol[all_f] = 0
    flashes = all_f.sum()
    return flashes, octol


def advent11():
    file11 = open('input11.txt')
    octol = np.zeros([10, 10], np.int)
    r_nr = 0
    for line in file11:
        line = line.strip('\n')
        row = [int(n) for n in line]
        octol[r_nr, :] = row
        r_nr += 1
    #print(octol)
    nflash = 0
    for step in range(1, 101):
        flashes, octol = update_octol(octol)
        nflash += flashes
    print(nflash)

    nflash = 0
    for step in range(1, 1000):
        flashes, octol = update_octol(octol)
        nflash += flashes
        if octol.sum() == 0:
            print(step + 100)
            print(octol)
            break

    print(nflash)


def ok_to_revisit_small_cave(path, revisit):
    if not revisit:
        return False
    nodes = path.split('-')
    small_caves = [cave for cave in nodes if cave.islower()]
    for i in range(0, len(small_caves)):
        for j in range(i + 1, len(small_caves)):
            if small_caves[i] == small_caves[j]:
                return False
    return True


def append_leg(path, node, nodes, all_paths, revisit):
    #print(path)
    if 'end' in path:
        return path, all_paths
    for leg in nodes[node]:
        if leg == 'end':
            all_paths.append(path + '-' + leg)
            #print('   + ', 'End found!!')
            continue
        elif leg in path and leg.islower() and not ok_to_revisit_small_cave(path, revisit) or leg in ['start', 'end']:
            #print(leg, ' - ', 'Dead end!!')
            continue
        else:
            path = path + '-' + leg
            cp = path
            path, all_paths = append_leg(path, leg, nodes, all_paths, revisit)
            if path == cp:
                #print('stripping')
                path = path.rstrip('-' + leg)
    return path, all_paths


def advent12(revisit=False):
    file12 = open('input12.txt')
    legs = []
    nodes = set()
    for line in file12:
        line = line.strip('\n')
        legs.append(line)
        endpoints = line.split('-')
        nodes.add(endpoints[0])
        nodes.add(endpoints[1])

    nodes_w_n = dict()
    for node in nodes:
        n_w_n = []
        for leg in legs:
            if node in leg:
                n_w_n.append(leg.replace(node, '').replace('-', ''))
        nodes_w_n[node] = n_w_n
    print(nodes_w_n)
    all_paths = []
    for leg in nodes_w_n['start']:
        path = 'start' + '-' + leg
        try:
            path, all_paths = append_leg(path, leg, nodes_w_n, all_paths, revisit)
        except Exception as e:
            pass
    #print(all_paths)
    print(len(all_paths))


def advent13():
    file13 = open('input13.txt')
    indices = []
    max_i = 0
    max_j = 0
    for row in range(0, 799):
        line = file13.readline().strip('\n')
        line_s = line.split(',')
        coords = [int(line_s[0]), int(line_s[1])]
        max_i = max(max_i, coords[0])
        max_j = max(max_j, coords[1])
        indices.append(coords)

    page_one = np.zeros([max_i + 1, max_j + 1], np.int)
    for coords in indices:
        page_one[coords[0], coords[1]] = 1

    file13.readline()
    folds = []
    for row in range(0, 12):
        line = file13.readline().strip('\n')
        line_s = line.split()
        fold = line_s[2]
        #print(fold)
        fold = fold.split('=')
        if fold[0] == 'x':
            folds.append([int(fold[1]), 0])
        else:
            folds.append([0, int(fold[1])])

    for fold in folds:
        if fold[0] > 0:
            page_two = np.zeros([fold[0], np.size(page_one, 1)], np.int)
            for i in range(fold[0] + 1, 2*fold[0] + 1):
                for j in range(fold[1], np.size(page_one, 1)):
                    page_two[2*fold[0] - i, j] = page_one[2*fold[0] - i, j] or page_one[i, j]
            print(page_two[0:fold[0], :].sum())
        else:
            page_two = np.zeros([np.size(page_one, 0), fold[1]], np.int)
            for i in range(0, np.size(page_one, 0)):
                for j in range(fold[1] + 1, 2 * fold[1] + 1):
                    page_two[i, 2 * fold[1] - j] = page_one[i, 2 * fold[1] - j] or page_one[i, j]
            print(page_two[:, :].sum())
        page_one = page_two
    print(np.size(page_one, 0), np.size(page_one, 1))
    for ph in range(0, 8, 1):
        print(page_one[ph*5:(ph+1)*5, :])
        print()


def advent14():
    file14 = open('input14.txt')
    template = file14.readline().strip('\n')
    template = [ch for ch in template]
    #print(template)
    file14.readline()
    insertions = []
    for row in range(3, 103):
        line = file14.readline().strip('\n')
        line = line.split(' -> ')
        insertions.append(line)
        #print(line)

    for it in range(0, 10):
        new_temp = []
        new_temp.append(template[0])
        for i in range(0, len(template) - 1):
            #print(template[i], template[i + 1])
            inserted = False
            for pair in insertions:
                if (template[i] + template[i + 1]) == pair[0]:
                    #print('par: ', template[i] + template[i + 1], ' insert: ', pair[1])
                    #new_temp.append(template[i])
                    new_temp.append(pair[1])
                    new_temp.append(template[i + 1])
                    inserted = True
            if not inserted:
                #new_temp.append(template[i])
                new_temp.append(template[i + 1])
        template = new_temp

    #print(len(new_temp))
    counter = Counter(template)
    most_common = counter.most_common()
    print(most_common)

def advent14_2():
    file14 = open('input14.txt')
    template = file14.readline().strip('\n')
    template = [ch for ch in template]
    #print(template)
    file14.readline()
    insertions = []
    for row in range(3, 103):
        line = file14.readline().strip('\n')
        line = line.split(' -> ')
        insertions.append(line)
        #print(line)

    pairs = dict()
    letters = dict()
    for pair in insertions:
        pairs[pair[0]] = 0
    for i in range(0, len(template) - 1):
        if template[i] in letters:
            letters[template[i]] += 1
        else:
            letters[template[i]] = 1
        for pair in insertions:
            if (template[i] + template[i + 1]) == pair[0]:
                pairs[pair[0]] += 1
    if template[-1] in letters:
        letters[template[-1]] += 1
    else:
        letters[template[-1]] = 1
    #print(pairs)
    #print(insertions)
    for it in range(0, 40):
        new_pairs = dict()
        for pair in insertions:
            if pair[0] in pairs:
                if pair[1] in letters:
                    letters[pair[1]] += pairs[pair[0]]
                else:
                    letters[pair[1]] = pairs[pair[0]]
                if pair[0][0] + pair[1] in new_pairs:
                    new_pairs[pair[0][0] + pair[1]] += pairs[pair[0]]
                else:
                    new_pairs[pair[0][0] + pair[1]] = pairs[pair[0]]
                if pair[1] + pair[0][1] in new_pairs:
                    new_pairs[pair[1] + pair[0][1]] += pairs[pair[0]]
                else:
                    new_pairs[pair[1] + pair[0][1]] = pairs[pair[0]]
            else:
                if pair[0][0] + pair[1] in new_pairs:
                    new_pairs[pair[0][0] + pair[1]] += 0
                else:
                    new_pairs[pair[0][0] + pair[1]] = 0
                if pair[1] + pair[0][1] in new_pairs:
                    new_pairs[pair[1] + pair[0][1]] += 0
                else:
                    new_pairs[pair[1] + pair[0][1]] = 0
        pairs = new_pairs

    #print('Cc: ', letters['C'])
    #print('Pc: ', letters['P'])
    #print(letters)
    print(letters['C'] - letters['P'])


def find_neighbors(ij, imax, jmax):
    n = []
    if ij[0] > 0:
        n.append([ij[0] - 1, ij[1]])
    if ij[0] < imax:
        n.append([ij[0] + 1, ij[1]])
    if ij[1] > 0:
        n.append([ij[0], ij[1] - 1])
    if ij[1] < jmax:
        n.append([ij[0], ij[1] + 1])
    return n


def NotReallyDijkstra(graph, start):
    dist = np.ones([np.size(graph, 0), np.size(graph, 1)])
    visited = (dist != 1)
    dist[:, :] = 2**32
    dist[start[0], start[1]] = 0
    visited[start[0], start[1]] = True
    start_node = start
    value_set = dict()
    imax = np.size(graph, 0) - 1
    jmax = np.size(graph, 1) - 1
    while not visited.all():
        neighbors = find_neighbors(start_node, imax, jmax)
        for node in neighbors:
            if not visited[node[0], node[1]]:
                dist[node[0], node[1]] = min(dist[node[0], node[1]], graph[node[0], node[1]] + dist[start_node[0], start_node[1]])
                value_set[node[0], node[1]] = dist[node[0], node[1]]
        temp = min(value_set.values())
        min_nodes = [key for key in value_set if value_set[key] == temp]
        start_node = min_nodes[0]
        value_set.pop(start_node)
        visited[start_node[0], start_node[1]] = True
        #print(dist)
        #print('====================================================')
    return dist


def advent15():
    file15 = open('input15.txt')
    risk = np.zeros([100, 100], np.int)
    r_nr = 0
    for line in file15:
        line = line.strip('\n')
        row = [int(n) for n in line]
        risk[r_nr, :] = row
        r_nr += 1
    #print(risk)
    dist = NotReallyDijkstra(risk, [0, 0])
    print(dist[-1, -1])


def advent15_2():
    file15 = open('input15.txt')
    risk = np.zeros([100, 100], np.int)
    r_nr = 0
    for line in file15:
        line = line.strip('\n')
        row = [int(n) for n in line]
        risk[r_nr, :] = row
        r_nr += 1

    riskx5 = np.zeros([500, 500])
    riskx5[0:100, 0:100] = risk
    for j in range(1, 5):
        riskx5[0:100, j*100:(j+1)*100] = (riskx5[0:100, (j-1)*100:j*100] + 1) % 10
        riskx5[0:100, j * 100:(j + 1) * 100][riskx5[0:100, j * 100:(j + 1) * 100] == 0] = 1
    for i in range(1, 5):
        riskx5[i*100:(i+1)*100, 0:100] = (riskx5[(i-1)*100:i*100, 0:100] + 1) % 10
        riskx5[i*100:(i+1)*100, 0:100][riskx5[i*100:(i+1)*100, 0:100] == 0] = 1
    for i in range(1, 5):
        for j in range(1, 5):
            riskx5[i*100:(i+1)*100, j * 100:(j + 1) * 100] = (riskx5[(i-1)*100:i*100, j*100:(j+1)*100] + 1) % 10
            riskx5[i*100:(i+1)*100, j * 100:(j + 1) * 100][riskx5[i*100:(i+1)*100, j*100:(j+1)*100] == 0] = 1

    #print(riskx5)
    dist = NotReallyDijkstra(riskx5, [0, 0])
    print(dist[-1, -1])


def hex2bin(l):
    bins = bin(int(l, 16)).replace('0b', '').zfill(4)
    return bins


def product(l):
    prod = 1
    for e in l:
        prod = prod * e
    return prod


def gt(l):
    if l[0] > l[1]:
        return 1
    return 0


def lt(l):
    if l[0] < l[1]:
        return 1
    return 0


def equal(l):
    if l[0] == l[1]:
        return 1
    return 0


def operator(type):
    if type == 0:
        return 'sum(['
    elif type == 1:
        return 'main.product(['
    elif type == 2:
        return 'min(['
    elif type == 3:
        return 'max(['
    elif type == 5:
        return 'main.gt(['
    elif type == 6:
        return 'main.lt(['
    elif type == 7:
        return 'main.equal(['


def parse_a_little(bins, version_sum):
    version = bins[0:3]
    version_sum += int(version, 2)
    bins = bins[3:]
    type = bins[0:3]
    bins = bins[3:]
    len_parsed = 6
    if type == '100': #literal
        #print('literal: ', bins)
        literal = str()
        firstbit = '1'
        while firstbit == '1':
            firstbit = bins[0]
            literal += bins[1:5]
            bins = bins[5:]
            len_parsed += 5
        print(int(literal, 2), end='')
    else:
        print(operator(int(type, 2)), end='')
        length_type = bins[0]
        bins = bins[1:]
        if length_type == '0':
            #print('ltype0:  ', bins)
            bit_length = int(bins[0:15], 2)
            bins = bins[15:]
            pre_len = len(bins)
            curr_len = pre_len
            while (curr_len + bit_length) > pre_len:
                version_sum, bins = parse_a_little(bins, version_sum)
                curr_len = len(bins)
                print(',', end='')
                #print(curr_len + bit_length, pre_len)
            print('])', end='')
        else:
            no_packets = int(bins[0:11], 2)
            bins = bins[11:]
            for p in range(0, no_packets):
                version_sum, bins = parse_a_little(bins, version_sum)
                print(',', end='')
            print('])', end='')
    #overhang = (len_parsed // 4 + 1)*4 - len_parsed
    #bins = bins[overhang:]
    return version_sum, bins


def advent16():
    file16 = open('input16.txt')
    hexs = file16.readline().strip('\n')
    bins = str()
    #hexs = '9C0141080250320F1802104A08'
    for h in hexs:
        bins += hex2bin(h)

    version_sum, bins = parse_a_little(bins, 0)
    print(version_sum)


def not_passed(xy, xu, yl):
    return (xy[0] <= xu) and (xy[1] >= yl)


def within_bounds(traj, xb, yb):
    pos = traj[-2]
    if (xb[0] <= pos[0] <= xb[1]) and (yb[1] <= pos[1] <= yb[0]):
        #print(pos)
        return True
    return False


def trajectory(vx, vy, xu, yl):
    traj = [[0, 0]]

    while not_passed(traj[-1], xu, yl):
        new_pos = [traj[-1][0] + vx, traj[-1][1] + vy]
        traj.append(new_pos)
        vx = max(vx - 1, 0)
        vy = vy - 1
    return traj


def advent17():
    xb = [70, 125]
    yb = [-121, -159]
    max_y = 0
    no_succ = 0
    for vx in range(0, 150):
        for vy in range(yb[0]-200, -yb[1]+240):
            traj = trajectory(vx, vy, xb[1], yb[1])
            #print(traj)
            if within_bounds(traj, xb, yb):
                lmax_y = max(traj, key=lambda value: int(value[1]))
                #print(lmax_y, vx, vy)
                max_y = max(max_y, lmax_y[1])
                no_succ += 1
            #else:
                #print('Without!!')
    print(max_y)
    print(no_succ)


def ssf_depth(l):
    depth = 0
    for c in l:
        if c == '[':
            depth += 1
        elif c == ']':
            depth -= 1
        if depth == 5:
            return depth
    return depth


def ssf_leftmost_4(l):
    depth = 0
    for i in range(0, len(l)):
        if l[i] == '[':
            depth += 1
        elif l[i] == ']':
            depth -= 1
        if depth == 5:
            return i + 1
    return 0


def ssf_leftmost_10(l):
    for i in range(0, len(l)):
        if l[i:i+2].isdigit():
            return i
    return 0


def ssf_add_numbers(l, r):
    l = '[' + l + ',' + r + ']'
    #print('after add: ', l)
    while ssf_leftmost_4(l) > 0 or ssf_leftmost_10(l) > 0:
        if ssf_leftmost_4(l) > 0:
            l = ssf_explode(l)
            #print('explode: ', l)
            continue
        ai = ssf_leftmost_10(l)
        if ai > 0:
            l = ssf_split_leftmost(l, ai)
            #print('split: ', l)
            continue
    return l


def ssf_split_leftmost(l, ai):
    val = int(l[ai:ai+2])
    lval = val // 2
    rval = val - lval
    l = l[:ai] + '[' + str(lval) + ',' +str(rval) + ']' + l[ai+2:]
    return l


def ssf_add_to_number_on_left(l, ai, aval):
    #print('left: ', l, ai, aval)
    for i in range(ai - 1, 0, -1):
        if l[i-2:i].isdigit():
            val = int(l[i-2:i]) + aval
            #print('val: ', val, i)
            l = l[:i - 2] + str(val) + l[i:]
            #print(l)
            add = 1 + len(str(aval))
            return l, add
        elif l[i].isdigit():
            val = int(l[i]) + aval
            #print('val: ', val, i)
            l = l[:i] + str(val) + l[i+1:]
            add = len(str(val)) + len(str(aval))
            return l, add
    return l, len(str(aval)) + 1


def ssf_add_to_number_on_right(l, ai, aval):
    #print('right: ', l, l[ai], ai, aval)
    add = len(str(aval))
    for i in range(ai + add, len(l)):
        if l[i:i+2].isdigit():
            val = int(l[i:i+2]) + aval
            l = l[:i] + str(val) + l[i+2:]
            return l
        if l[i].isdigit():
            #print('l[i]: ', l[i], i)
            val = int(l[i]) + aval
            #print(val)
            l = l[:i] + str(val) + l[i+1:]
            #print(l)
            return l
    return l


def find_number(l, ai):
    if l[ai:ai+2].isdigit():
        return int(l[ai:ai+2])
    else:
        return int(l[ai])


def find_braces(l, ai):
    low = -1
    high = -1
    for li in range(ai, 0, -1):
        if l[li] == '[':
            low = li
            break
    for hi in range(ai + 2, len(l)):
        if l[hi] == ']':
            high = hi
            break
    return low, high


def ssf_explode(l):
    ai = ssf_leftmost_4(l)
    val = find_number(l, ai)
    #print('leftmost is: ', ai, val)
    l, add = ssf_add_to_number_on_left(l, ai, val)
    val = find_number(l, ai + add)
    l = ssf_add_to_number_on_right(l, ai + add, val)
    low, hi = find_braces(l, ai)
    #print('lh: ', low, hi, ai)
    l = l[:low] + '0' + l[hi + 1:]
    return l


def ssf_sum(l):
    l = l.replace('[', '[3*').replace(']', '*2]').replace(',', '+')
    l = l.replace('[', '(').replace(']', ')')
    return l


def advent18():
    file18 = open('input18.txt')
    sf_sum = file18.readline().strip('\n')
    #print(sf_sum)
    for line in file18:
        line = line.strip('\n')
        sf_number = line.replace(' ', '')
        #print('+ ', sf_number)
        sf_sum = ssf_add_numbers(sf_sum, sf_number)
        #print('= ', sf_sum)
    print(eval(ssf_sum(sf_sum)))


def advent18_2():
    file18 = open('input18.txt')
    sf_numbers = []
    for line in file18:
        line = line.strip('\n')
        sf_number = line.replace(' ', '')
        sf_numbers.append(sf_number)

    sf_sums = []
    for no1 in range(0, len(sf_numbers)):
        for no2 in range(0, len(sf_numbers)):
            if no1 != no2:
                sf_sum = eval(ssf_sum(ssf_add_numbers(sf_numbers[no1], sf_numbers[no2])))
                sf_sums.append(sf_sum)
    #print(sf_sums)
    print(max(sf_sums))


def find_overlapping_beacons(sensor1, sensor2):
    for signs in [[-1,-1,-1], [1,-1,-1], [-1,1,-1], [1,1,-1], [-1,-1,1], [1,-1,1], [-1,1,1], [1,1,1]]:
        for perm in [[0, 1, 2], [2, 0, 1], [1, 2, 0], [0, 2, 1], [2, 1, 0], [1, 0, 2]]:
            for r in sensor2:
                for q in sensor1:
                    matches = 0
                    mpairs = []
                    for b in sensor2:
                        beacon = [signs[0]*(b[perm[0]] - r[perm[0]]), signs[1]*(b[perm[1]] - r[perm[1]]), signs[2]*(b[perm[2]] - r[perm[2]])]
                        for br in sensor1:
                            if beacon[0] == (br[0]-q[0]) and beacon[1] == (br[1]-q[1]) and beacon[2] == (br[2]-q[2]):
                                matches += 1
                                break
                                #print(br, b)
                    if matches >= 12:
                        #print(mpairs)
                        #print('matches, signs perm, r: ', matches, signs, perm, [signs[0]*r[perm[0]] - q[0], signs[1]*r[perm[1]] - q[1], signs[2]*r[perm[2]] - q[2]])
                        return matches, signs, perm, [signs[0]*r[perm[0]] - q[0], signs[1]*r[perm[1]] - q[1], signs[2]*r[perm[2]] - q[2]]
    return matches, [1, 1, 1], [0, 1, 2], [0, 0, 0]


def find_transf_path_2_0(paths, start_node, p):
    for node in paths[start_node]:
        if node not in p:
            p.append(node)
            if node == 0:
                return p
            else:
                p = find_transf_path_2_0(paths, node, p)
                if 0 in p:
                    return p
                else:
                    p.pop()
    return p


def manhattan_distance(l, r):
    distance = abs(l[0] - r[0]) + abs(l[1] - r[1]) + abs(l[2] - r[2])
    return distance


def advent19():
    file19 = open('input19.txt')

    sensors = []
    while file19.readline(): #read sensor name
        sensor = []
        while True:
            row = file19.readline().strip('\n')
            if row == '':
                sensors.append(sensor)
                #print(sensor)
                break
            beacon_det = ast.literal_eval(row)
            sensor.append(beacon_det)

    paths = []
    transforms = dict()
    for count1 in range(0, len(sensors)):
        paths.append([])
        for count2 in range(0, len(sensors)):
            if count1 != count2:
                matches, signs, perm, transf = find_overlapping_beacons(sensors[count1], sensors[count2])
                if matches >= 12:
                    print('overlap', count1, '<-->', count2, matches, signs, perm, transf)
                    paths[count1].append(count2)
                transforms[(count1, count2)] = [signs, perm, transf]
            else:
                transforms[(count1, count2)] = [[1,1,1], [0,1,2], [0,0,0]]

    print(paths)
    all_beacons = set()
    all_sensors = [[0, 0, 0]]
    for beacon in sensors[0]:
        all_beacons.add(beacon)
    for i in range(1, len(sensors)):
        path = find_transf_path_2_0(paths, i, [i])
        #print(path)
        #for n in range(0, len(path) - 1):
            #print(transforms[(path[n], path[n + 1])])
        for b in sensors[i]:
            beacon = b
            sensor = [0, 0, 0]
            for n in range(0, len(path) - 1):
                #print(transforms[i])
                signs, perm, transf = transforms[(path[n + 1], path[n])]
                beacon = [signs[0]*beacon[perm[0]] - transf[0], signs[1]*beacon[perm[1]] - transf[1], signs[2]*beacon[perm[2]] - transf[2]]
                sensor = [signs[0]*sensor[perm[0]] - transf[0], signs[1]*sensor[perm[1]] - transf[1], signs[2]*sensor[perm[2]] - transf[2]]
            final_beacon = (beacon[0], beacon[1], beacon[2])
            final_sensor = (sensor[0], sensor[1], sensor[2])
            all_beacons.add(final_beacon)
            all_sensors.append(final_sensor)
    print(len(all_beacons))


    max_distance = 0
    for bl in all_sensors:
        for br in all_sensors:
            max_distance = max(max_distance, manhattan_distance(bl, br))
    print('max manhattan: ', max_distance)


def get_instruction_no(image, i, j):
    no1 = image[i-1, j-1:j+2]
    no2 = image[i, j-1:j+2]
    no3 = image[i+1, j-1:j+2]
    #print(no1, no2, no3)
    no = 1*no3[2] + 2*no3[1] + 4*no3[0] + 8*no2[2] + 16*no2[1] + 32*no2[0] + 64*no1[2] + 128*no1[1] + 256*no1[0]
    return no


def advent20():
    file20 = open('input20.txt')
    size = 100
    image0 = np.zeros([55 + size + 55, 55 + size + 55], np.int)
    r_nr = 55
    algo = file20.readline().strip('\n')
    file20.readline()
    for line in file20:
        line = line.strip('\n')
        row = [0 if n == '.' else 1 for n in line]
        image0[r_nr, 55:55 + size] = row
        r_nr += 1
    #print(image0, '\n')

    for it in range(0, 25):
        image1 = np.ones([55 + size + 55, 55 + size + 55], np.int)
        for i in range(1, np.size(image0, 0) - 1):
            for j in range(1, np.size(image0, 1) - 1):
                instr_no = get_instruction_no(image0, i, j)
                if algo[instr_no] == '.':
                    image1[i, j] = 0
                else:
                    image1[i, j] = 1
        image2 = np.zeros([55 + size + 55, 55 + size + 55], np.int)
        for i in range(1, np.size(image1, 0) - 1):
            for j in range(1, np.size(image1, 1) - 1):
                instr_no = get_instruction_no(image1, i, j)
                if algo[instr_no] == '.':
                    image2[i, j] = 0
                else:
                    image2[i, j] = 1
        image0 = image2
        if it == 0:
            lit = image0.sum()
            print('lit(2): ', lit)

    lit = image0.sum()
    print('lit(50): ', lit)


def new_position(old_position, roll):
    move = roll % 10
    position = old_position + move
    if position > 10:
        position -= 10
    return position


def advent21():
    winning_score = 1000
    pos_1 = 7
    pos_2 = 2
    dice = [e for e in range(1, 101)]
    for conc in range(0, 10):
        dice += dice
    score_1 = 0
    score_2 = 0
    dice_pos = 0
    nof_rolls = 0
    while True:
        roll_1 = sum(dice[dice_pos:dice_pos + 3])
        pos_1 = new_position(pos_1, roll_1)
        score_1 += pos_1
        dice_pos += 3
        if score_1 >= winning_score:
            print('no 1 won!')
            break
        roll_2 = sum(dice[dice_pos:dice_pos + 3])
        pos_2 = new_position(pos_2, roll_2)
        dice_pos += 3
        score_2 += pos_2
        if score_2 >= winning_score:
            print('no 2 won!')
            break

    print(dice_pos)
    print(score_1, score_2)
    losing_score = min(score_1, score_2)
    print(losing_score*dice_pos)


def cover_reactor(ins):
    for dim in range(0, 3):
        #print(ins[dim])
        if (int(ins[dim][0]) > 50 and int(ins[dim][1]) > 50) or (int(ins[dim][0]) < -50 and int(ins[dim][1]) < -50):
            return False
    return True


def cube_intersect(l, r):
    for dim in range(0, 3):
        if (l[dim][0] > r[dim][1] and l[dim][1] > r[dim][1]) or (l[dim][0] < r[dim][0] and l[dim][1] < r[dim][0]):
            return False, [[0, 0], [0, 0], [0, 0]]
    cube = []
    for dim in range(0, 3):
        low = max(l[dim][0], r[dim][0])
        high = min(l[dim][1], r[dim][1])
        cube.append([low, high])
    return True, cube


def volume(cube):
    vol = (cube[0][1] - cube[0][0] + 1)*(cube[1][1] - cube[1][0] + 1)*(cube[2][1] - cube[2][0] + 1)
    if vol < 0:
        print('negative volume!!!')
    return vol


def overlaps(box, overlapping_boxes, intersecting_boxes, level=1):
    overlap = 0
    #if len(overlapping_boxes) == 0:
    #    return overlap
    for i in range(0, len(overlapping_boxes)):
        ov_box = overlapping_boxes[i]
        inter, int_box = cube_intersect(box, ov_box)
        if inter:
            if level == 0:
                intersecting_boxes.append(int_box)
            overlap += volume(int_box) - overlaps(int_box, overlapping_boxes[i + 1:], intersecting_boxes, level + 1)
    return overlap


def advent22():
    file22 = open('input22.txt')
    instructions = []
    for line in file22:
        line = line.strip('\n')
        if line[0:2] == 'on':
            instruction = line[3:].split(',')
            instruction = [e[2:].split('..') for e in instruction]
            instruction.append(1)
            instructions.append(instruction)
            #print(instruction)
        else:
            instruction = line[4:].split(',')
            instruction = [e[2:].split('..') for e in instruction]
            instruction.append(0)
            instructions.append(instruction)
            #print(instruction)

    reactor = np.zeros([101, 101, 101], np.int)
    outside_core = []
    for ins in instructions:
        if cover_reactor(ins):
            reactor[int(ins[0][0])+50:int(ins[0][1])+51, int(ins[1][0])+50:int(ins[1][1])+51, int(ins[2][0])+50:int(ins[2][1])+51] = ins[3]
        else:
            #print('no cover: ', ins)
            outside_core.append(ins)

    ons = []
    offs = []
    onoffs = []
    offonoffs = []
    onoffonoffs = []
    offonoffonoffs = []
    external_volume = 0
    for instruct in outside_core:
        print(external_volume, instruct)
        box = [[int(e[0]), int(e[1])] for e in instruct[0:3]]
        if instruct[3] == 1:
            external_volume += volume(box)
            #print('1', external_volume)
            external_volume -= overlaps(box, ons, [])
            #print('2', external_volume)
            new_onoffs = []
            external_volume += overlaps(box, offs, new_onoffs, 0)
            #print('3', external_volume, onoffs)
            external_volume -= overlaps(box, onoffs, [])
            new_onoffonoffs = []
            external_volume += overlaps(box, offonoffs, new_onoffonoffs, 0)
            external_volume -= overlaps(box, onoffonoffs, [])
            external_volume += overlaps(box, offonoffonoffs, [])
            for onoff in new_onoffs:
                onoffs.append(onoff)
            for onoffonoff in new_onoffonoffs:
                onoffonoffs.append(onoffonoff)
            ons.append(box)
        else:
            new_offs = []
            external_volume -= overlaps(box, ons, new_offs, 0)
            #print(new_offs)
            #print(external_volume)
            external_volume += overlaps(box, offs, [])
            #print(external_volume)
            new_offonoffs = []
            external_volume -= overlaps(box, onoffs, new_offonoffs, 0)
            external_volume += overlaps(box, offonoffs, [])
            new_offonoffonoffs = []
            external_volume -= overlaps(box, onoffonoffs, new_offonoffonoffs, 0)
            print(offonoffonoffs)
            external_volume += overlaps(box, offonoffonoffs, offonoffonoffs)
            #print(external_volume)
            #print(new_offs)
            for off in new_offs:
                offs.append(off)
            for off in new_offonoffs:
                offonoffs.append(off)
            for off in new_offonoffonoffs:
                offonoffonoffs.append(off)
            #offs.append(box)

    print('core: ', reactor.sum())
    print('external: ', external_volume)
    print(reactor.sum() + external_volume)


def run_program(program, input):
    #print('input: ', input)
    vars = dict()
    vars['x'] = 0
    vars['y'] = 0
    vars['z'] = 0
    vars['w'] = 0
    input_loc = 0
    for instruction in program:
        if instruction[0:3] == 'inp':
            vars[instruction[4]] = input[input_loc]
            #print(vars)
            input_loc += 1
        else:
            ilen = len(instruction)
            if instruction[6] in 'xyzw':
                b = int(vars[instruction[6]])
            else:
                if instruction[6] == '-':
                    b = -int(instruction[7:])
                else:
                    b = int(instruction[6:])
            if instruction[0:3] == 'add':
                tmp = int(vars[instruction[4]]) + b
                vars[instruction[4]] = tmp
            elif instruction[0:3] == 'mul':
                tmp = int(vars[instruction[4]]) * b
                vars[instruction[4]] = tmp
            elif instruction[0:3] == 'div':
                tmp = int(vars[instruction[4]]) // b
                vars[instruction[4]] = tmp
            elif instruction[0:3] == 'mod':
                tmp = int(vars[instruction[4]]) % b
                vars[instruction[4]] = tmp
            elif instruction[0:3] == 'eql':
                #print(vars[instruction[4]], '==', b)
                if int(vars[instruction[4]]) == b:
                    vars[instruction[4]] = 1
                else:
                    vars[instruction[4]] = 0
                if instruction[6] == '0':
                    print(instruction, vars[instruction[4]])
        #print(instruction)
        #print(vars['x'], vars['y'], vars['z'], vars['w'])
    return vars['z']


def transform_program(program):
    new_program =[]
    input_loc = 0
    input_var = 'C' + str(input_loc)
    input_loc += 1
    for instruction in program:
        if instruction[0:3] == 'inp':
            input_var = 'C' + str(input_loc)
            input_loc += 1
        else:
            if instruction[6] == 'w':
                b = input_var
            else:
                b = instruction[6:]
            if instruction[0:3] == 'add':
                tmp = instruction[4] + '=(' + instruction[4] + '+' + str(b)+ ')'
            elif instruction[0:3] == 'mul':
                tmp = instruction[4] + '=(' + instruction[4] + '*' + str(b)+ ')'
            elif instruction[0:3] == 'div':
                tmp = instruction[4] + '=(' + instruction[4] + '//' + str(b)+ ')'
            elif instruction[0:3] == 'mod':
                tmp = instruction[4] + '=(' + instruction[4] + '%' + str(b)+ ')'
            elif instruction[0:3] == 'eql':
                tmp = instruction[4] + '=(' + instruction[4] + '==' + str(b) + ')'
            #print(tmp)
            loc = tmp.find('*0')
            if loc >= 0:
                tmp = tmp[:loc - 1] + '0' + tmp[loc + 2:]
                #print('opt:', tmp)
            loc = tmp.find('+-')
            if loc >= 0:
                tmp = tmp[:loc] + '-' + tmp[loc + 2:]
                #print('opt:', tmp)
            new_program.append(tmp)
            print(tmp)

    return new_program


def build_func(program):
    C14 = sympy.symbols('C14', integer=True, positive=True)
    C1 = sympy.symbols('C1', integer=True, positive=True)
    C2 = sympy.symbols('C2', integer=True, positive=True)
    C3 = sympy.symbols('C3', integer=True, positive=True)
    C4 = sympy.symbols('C4', integer=True, positive=True)
    C5 = sympy.symbols('C5', integer=True, positive=True)
    C6 = sympy.symbols('C6', integer=True, positive=True)
    C7 = sympy.symbols('C7', integer=True, positive=True)
    C8 = sympy.symbols('C8', integer=True, positive=True)
    C9 = sympy.symbols('C9', integer=True, positive=True)
    C10 = sympy.symbols('C10', integer=True, positive=True)
    C11 = sympy.symbols('C11', integer=True, positive=True)
    C12 = sympy.symbols('C12', integer=True, positive=True)
    C13 = sympy.symbols('C13', integer=True, positive=True)
    x_func = '1'
    y_func = '0'
    z_func = '0'
    count = 0
    for inst in program:
        if inst[0] == '#':
            continue
        l, r = inst.split('=', maxsplit=1)
        #print(l, r)
        if 'y' in r:
            r = r.replace('y', y_func)
        if 'x' in r:
            r = r.replace('x', x_func)
        if 'z' in r:
            r = r.replace('z', z_func)
        #r = r.replace('(0)+', '')
        #r = r.replace('//1', '')
        if '==' in r:
            print(r)
            print(str(sympy.simplify(r.split('==')[0][1:])))
            break
        if l == 'x':
            x_func = r#str(sympy.simplify(r))
        elif l == 'y':
            y_func = r#str(sympy.simplify(r))
        elif l == 'z':
            z_func = r#str(sympy.simplify(r))
        #print(z_func, count)
        count += 1
        #if count > 195:
        #    break

    return sympy.simplify(z_func)


def advent24():
    file24 = open('input24.txt')

    model_no = 71111591176151

    instructions = []
    for line in file24:
        line = line.strip('\n')
        instructions.append(line)

    #print(instructions)
    res = run_program(instructions, [d for d in str(model_no)])
    print(res)

    #program = instructions #transform_program(instructions)
    #z = build_func(program)
    #print(z)
    #finfun([int(d) for d in str(model_no)])
    #while True:
    #    mno = [d for d in str(model_no)]
    #    z = run_program(instructions, mno)
    #    if z == 0:
    #        break
    #    model_no = gen_model_no(model_no)


def advent25():
    file25 = open('input25.txt')
    nrows = 137
    ncols = 139
    size = (nrows, ncols)
    bottom = np.full(size, '.', dtype=str)
    r_nr = 0
    for line in file25:
        line = line.strip('\n')
        row = [c for c in line]
        bottom[r_nr, :] = row
        r_nr += 1
    #print(bottom)
    step = 1
    while True:
        new_bottom1 = bottom.copy()
        for i in range(0, nrows):
            for j in range(0, ncols):
                if bottom[i, j] == '>':
                    if bottom[i, (j + 1)%ncols] == '.':
                        new_bottom1[i, j] = '.'
                        new_bottom1[i, (j + 1)%ncols] = '>'
                        #print(i, (j + 1)%ncols, 'got > from', i, j)
        new_bottom2 = new_bottom1.copy()
        for i in range(0, nrows):
            for j in range(0, ncols):
                if new_bottom1[i, j] == 'v':
                    if new_bottom1[(i + 1)%nrows, j] == '.':
                        new_bottom2[i, j] = '.'
                        new_bottom2[(i + 1)%nrows, j] = 'v'
        if (bottom == new_bottom2).all():
            break
        bottom = new_bottom2.copy()
        step += 1
    print(step)
    #print(bottom)


def valid_endpos(burrow, mover, endpos):
    #print(endpos[0], endpos[1])
    depth = 6
    if burrow[endpos[0], endpos[1]] == '.':
        if endpos[0] == 1 and endpos[1] not in [3, 5, 7, 9]:
            #print('not above opening')
            return True
        # if endpos[0] == 3:
        #     if mover == 'A' and endpos[1] == 3:
        #         return True
        #     if mover == 'B' and endpos[1] == 5:
        #         return True
        #     if mover == 'C' and endpos[1] == 7:
        #         return True
        #     if mover == 'D' and endpos[1] == 9:
        #         return True
        if endpos[0] > 1:
            if mover == 'A' and endpos[1] == 3 and ((burrow[endpos[0] + 1:depth, 3] == 'A') + (burrow[endpos[0] + 1:depth, 3] == '#')).all():
                return True
            if mover == 'B' and endpos[1] == 5 and ((burrow[endpos[0] + 1:depth, 5] == 'B') + (burrow[endpos[0] + 1:depth, 5] == '#')).all():
                return True
            if mover == 'C' and endpos[1] == 7 and ((burrow[endpos[0] + 1:depth, 7] == 'C') + (burrow[endpos[0] + 1:depth, 7] == '#')).all():
                return True
            if mover == 'D' and endpos[1] == 9 and ((burrow[endpos[0] + 1:depth, 9] == 'D') + (burrow[endpos[0] + 1:depth, 9] == '#')).all():
                return True
    return False


def all_correct(map, amps):
    for amp in amps:
        if not in_right_place(map, amp[1], amp[0]):
            return False
    return True


def in_right_place(burrow, who, pos):
    depth = burrow.shape[0] - 1
    if pos[0] > 1:
        if who == 'A' and pos[1] == 3 and ((burrow[pos[0]:depth, 3] == 'A') + (burrow[pos[0]:depth, 3] == '#')).all():
            return True
        if who == 'B' and pos[1] == 5 and ((burrow[pos[0]:depth, 5] == 'B') + (burrow[pos[0]:depth, 5] == '#')).all():
            return True
        if who == 'C' and pos[1] == 7 and ((burrow[pos[0]:depth, 7] == 'C') + (burrow[pos[0]:depth, 7] == '#')).all():
            return True
        if who == 'D' and pos[1] == 9 and ((burrow[pos[0]:depth, 9] == 'D') + (burrow[pos[0]:depth, 9] == '#')).all():
            return True
    # if pos[0] == 3:
    #     if who == 'A' and pos[1] == 3:
    #         return True
    #     if who == 'B' and pos[1] == 5:
    #         return True
    #     if who == 'C' and pos[1] == 7:
    #         return True
    #     if who == 'D' and pos[1] == 9:
    #         return True
    return False


def possible_move(map, start, end):
    length = 0
    if start[0] > 1:
        first_move = map[1:start[0], start[1]]
        if not (first_move == '.').all():
            #print('1f')
            return False, 0
        else:
            length += start[0] - 1
    if start[1] != end[1]:
        second_move = map[1, min(start[1], end[1]) + 1:max(start[1], end[1])]
        if not (second_move == '.').all():
            #print('2f', second_move)
            return False, 0
        else:
            length += abs(start[1] - end[1])
    if end[0] > 1:
        third_move = map[1:end[0], end[1]]
        if not (third_move == '.').all():
            #print('3f')
            return False, 0
        else:
            length += end[0] - 1

    return True, length


def valid_moves(burrow, valid_coords, amphipod):
    moves = []
    depth = burrow.shape[0] - 2
    if amphipod[2] < 2:
        cost = 1
        if amphipod[1] == 'A':
            #for cs in [np.array([2, 3]), np.array([3, 3])]:
            for r in range(2, depth + 1):
                cs = np.array([r, 3])
                if valid_endpos(burrow, amphipod[1], cs):
                    possible, tlen = possible_move(burrow, amphipod[0], cs)
                    if possible:
                        moves.append([cs, cost * tlen])
        if amphipod[1] == 'B':
            cost = 10
            #for cs in [np.array([2, 5]), np.array([3, 5])]:
            for r in range(2, depth + 1):
                cs = np.array([r, 5])
                if valid_endpos(burrow, amphipod[1], cs):
                    possible, tlen = possible_move(burrow, amphipod[0], cs)
                    if possible:
                        moves.append([cs, cost * tlen])
        elif amphipod[1] == 'C':
            cost = 100
            #for cs in [np.array([2, 7]), np.array([3, 7])]:
            for r in range(2, depth + 1):
                cs = np.array([r, 7])
                if valid_endpos(burrow, amphipod[1], cs):
                    possible, tlen = possible_move(burrow, amphipod[0], cs)
                    if possible:
                        moves.append([cs, cost * tlen])
        elif amphipod[1] == 'D':
            cost = 1000
            #for cs in [np.array([2, 9]), np.array([3, 9])]:
            for r in range(2, depth + 1):
                cs = np.array([r, 9])
                if valid_endpos(burrow, amphipod[1], cs):
                    possible, tlen = possible_move(burrow, amphipod[0], cs)
                    if possible:
                        moves.append([cs, cost * tlen])
        if len(moves) == 0:
            for cs in valid_coords:
                if (cs == amphipod[0]).all():
                    continue
                if cs[0] == 1 and amphipod[0][0] == 1:
                    continue
                if valid_endpos(burrow, amphipod[1], cs):
                    possible, tlen = possible_move(burrow, amphipod[0], cs)
                    if possible:
                        moves.append([cs, cost*tlen])
    return moves


def no_moves_left(map, valid_coords, amphipods):
    for amp in amphipods:
        if amp[2] < 2 and len(valid_moves(map, valid_coords, amp)) > 0:
            return False
    return True


def advent23(in_file, nrows):
    file23 = open(in_file)
    ncols = 13
    size = (nrows, ncols)
    burrow = np.full(size, '.', dtype=str)
    r_nr = 0
    for line in file23:
        line = line.strip('\n')
        row = [c for c in line]
        burrow[r_nr, :] = row
        r_nr += 1

    valid_coords_0 = np.argwhere(burrow == '.')
    valid_coords_A = np.argwhere(burrow == 'A')
    valid_coords_B = np.argwhere(burrow == 'B')
    valid_coords_C = np.argwhere(burrow == 'C')
    valid_coords_D = np.argwhere(burrow == 'D')
    valid_coords = np.append(valid_coords_0, valid_coords_A, axis=0)
    valid_coords = np.append(valid_coords, valid_coords_B, axis=0)
    valid_coords = np.append(valid_coords, valid_coords_C, axis=0)
    valid_coords = np.append(valid_coords, valid_coords_D, axis=0)
    for c in valid_coords:
        burrow[c[0], c[1]] = '.'
    print(burrow)
    amphipods = []
    for cs in valid_coords_A:
        amphipods.append((cs, 'A', 0))
    for cs in valid_coords_B:
        amphipods.append((cs, 'B', 0))
    for cs in valid_coords_C:
        amphipods.append((cs, 'C', 0))
    for cs in valid_coords_D:
        amphipods.append((cs, 'D', 0))
    for a in amphipods:
        burrow[a[0][0], a[0][1]] = a[1]
    print(burrow)
    all_moves = []
    moves = []
    start_cost = 0
    min_cost = 100000000
    all_moves, min_cost = do_move(burrow, amphipods, valid_coords, moves, all_moves, start_cost, min_cost)
    print(len(all_moves))
    costs = []
    for solution in all_moves:
        costs.append(sum_move_cost(solution))
    print('Min cost: ', min(costs))


def do_move(burrow, amphipods, valid_coords, moves, all_moves, current_cost, min_cost):
    #print('In do_move(...)', len(moves))
    #print(burrow)
    for ampno in range(0, len(amphipods)):
        amp = amphipods[ampno]
        #print('In do_move(...)', len(moves), amp)
        #if len(moves) < 2:
        #   print(burrow)
        if in_right_place(burrow, amp[1], amp[0]) or amp[2] == 2:
            continue
        vm = valid_moves(burrow, valid_coords, amp)
        #print(vm)
        for m in vm:
            new_burrow = burrow.copy()
            new_burrow[amp[0][0], amp[0][1]] = '.'
            new_burrow[m[0][0], m[0][1]] = amp[1]
            new_amp = (m[0], amp[1], amp[2] + 1)
            new_amphipods = amphipods.copy()
            new_amphipods[ampno] = new_amp
            new_moves = moves.copy()
            new_moves.append(m)
            new_cost = current_cost + m[1]
            if all_correct(new_burrow, new_amphipods):
                #print('solution found!')
                #print(new_moves)
                all_moves.append(new_moves)
                min_cost = min(min_cost, new_cost)
                print('New min cost: ', min_cost)
                break
            if new_cost < min_cost:
                all_moves, min_cost = do_move(new_burrow, new_amphipods, valid_coords, new_moves, all_moves, new_cost, min_cost)
    return all_moves, min_cost


def sum_move_cost(moves):
    cost = 0
    for m in moves:
        cost += m[1]
    return cost


def next_turn(score_1, score_2, pos_1, pos_2, dice, winning_score, wins_1, wins_2, multiplier=1):
    dice_mult = {3: 1, 4: 3, 5: 6, 6: 7, 7: 6, 8: 3, 9: 1}
    for roll1 in dice:
        new_pos_1 = new_position(pos_1, roll1)
        new_multiplier1 = multiplier*dice_mult[roll1]
        new_score_1 = score_1 + new_pos_1
        #print('1 rolls ', roll1, ', score: ', new_score_1)
        if new_score_1 >= winning_score:
            wins_1 += 1*new_multiplier1
            #print('1 wins')
            continue
        for roll2 in dice:
            game_finished = False
            new_pos_2 = new_position(pos_2, roll2)
            new_multiplier2 = new_multiplier1*dice_mult[roll2]
            new_score_2 = score_2 + new_pos_2
            #print('2 rolls ', roll2, ', score: ', new_score_2)
            if new_score_2 >= winning_score:
                wins_2 += 1*new_multiplier2
                #print('2 wins')
                game_finished = True
            if not game_finished:
                wins_1, wins_2 = next_turn(new_score_1, new_score_2, new_pos_1, new_pos_2, dice, winning_score, wins_1, wins_2, new_multiplier2)

    return wins_1, wins_2


def advent21_2():
    winning_score = 21
    pos_1 = 7
    pos_2 = 2
    dice = []
    for i in range(3, 10):
        dice.append(i)

    score_1 = 0
    score_2 = 0
    wins_1 = 0
    wins_2 = 0

    wins_1, wins_2 = next_turn(score_1, score_2, pos_1, pos_2, dice, winning_score, wins_1, wins_2)
    print('Wins1: ', wins_1)
    print('Wins2: ', wins_2)


# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    start_time = time.time()
    print('Advent 1')
    advent1()
    end_time_1 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_1 - start_time))
    print('Advent 2')
    advent2()
    end_time_2 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_2 - end_time_1))
    print('Advent 3')
    advent3()
    print(advent3_2('0','1')*advent3_2('1','0'))
    end_time_3 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_3 - end_time_2))
    print('Advent 4')
    advent4()
    end_time_4 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_4 - end_time_3))
    print('Advent 5')
    advent5()
    advent5_2()
    end_time_5 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_5 - end_time_4))
    print('Advent 6')
    advent6()
    advent6_2()
    end_time_6 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_6 - end_time_5))
    print('Advent 7')
    advent7()
    end_time_7 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_7 - end_time_6))
    print('Advent 8')
    advent8()
    end_time_8 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_8 - end_time_7))
    print('Advent 9')
    advent9()
    end_time_9 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_9 - end_time_8))
    print('Advent 10')
    advent10()
    end_time_10 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_10 - end_time_9))
    print('Advent 11')
    advent11()
    end_time_11 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_11 - end_time_10))
    print('Advent 12')
    advent12()
    advent12(revisit=True)
    end_time_12 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_12 - end_time_11))
    print('Advent 13')
    advent13()
    end_time_13 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_13 - end_time_12))
    print('Advent 14')
    advent14()
    advent14_2()
    end_time_14 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_14 - end_time_13))
    print('Advent 15')
    advent15()
    advent15_2()
    end_time_15 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_15 - end_time_14))
    print('Advent 16')
    advent16()
    end_time_16 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_16 - end_time_15))
    print('Advent 17')
    advent17()
    end_time_17 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_17 - end_time_16))
    print('Advent 18')
    advent18()
    advent18_2()
    end_time_18 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_18 - end_time_17))
    print('Advent 19')
    #advent19()
    end_time_19 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_19 - end_time_18))
    print('Advent 20')
    advent20()
    end_time_20 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_20 - end_time_19))
    print('Advent 21')
    advent21()
    advent21_2()
    end_time_21 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_21 - end_time_20))
    print('Advent 22')
    advent22()
    end_time_22 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_22 - end_time_21))
    print('Advent 23')
    #advent23('input23.txt', 5)
    #advent23('input23_2.txt', 7)
    end_time_23 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_23 - end_time_22))
    print('Advent 24')
    advent24()
    end_time_24 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_24 - end_time_23))
    print('Advent 25')
    advent25()
    end_time_25 = time.time()
    print("time elapsed: {:.2f}s".format(end_time_25 - end_time_24))

    print("Accumulated time elapsed: {:.2f}s".format(end_time_25 - start_time))
    
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
