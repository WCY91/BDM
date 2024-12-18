result = []
for i in range(4):
    row = ""
    for j in range(4):
        if i==0 or i==(4-1):
            row+='*'
        else:
            if j==0 or j==(4-1):
                row+='*'
            else:
                row+=" "
print(result)