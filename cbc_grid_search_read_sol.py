'''
read txt file obtained from cbc_grid_search.py and print results ordered by time
'''

with open('cbc_grid_search_seed3.txt') as f:
    text = f.readlines()

result = {}
for i, row in enumerate(text):
    if i % 2 == 0:
        splitted_row = row.strip().replace(' ', '').split(',')
        key = []
        for el in splitted_row:
            el = el.strip().replace(' ', '').split('=')
            name = el[0]
            if name != 'trial':
                key.append(el[1])
        
        key = tuple(key)
        time = float(text[i+1].strip().split('=')[1])

        if key not in result.keys():
            result[key] = 0
        result[key] += time

for key, val in sorted(list(result.items()), key = lambda x: x[1]):
    print(key, val/5)