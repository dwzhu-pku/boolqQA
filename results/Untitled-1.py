with open (file='esim.txt',encoding='utf-8',mode='w+') as w:
    with open(file='esim', encoding='utf-8',mode='r') as f:
        lines = f.readlines()
        for line in lines:
            if line[-2:].split()!=[']'] and len(line.split())>1:
                w.write(line)