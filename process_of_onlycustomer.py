with open("fixed_test_dataset.csv","r+",encoding="utf-8") as file,\
    open("onlycustomer_test_dataset.csv",'w+',encoding='utf-8') as newfile:
    A=file.readline()
    newfile.write(A)
    A=file.readline()
    while A!='':
        A=A.split("|")
        messages=""
        now=0
        for i in range(len(A[1])-3):#剔除坐席说的话
            if(A[1][i:i+4]=='【坐席】'):
                messages+=A[1][now:i]+'[SEP]'
                now=i+4
            elif(A[1][i:i+4]=='【客户】'):
                now=i+4 

        messages=messages.split("[SEP]")#message表

        lens=[len(mes) for mes in messages]#每句话长度表

        #去掉短句
        choose_rate=int(0.3*len(lens))+1
        threshold=sorted(lens,reverse=True)[choose_rate]
        res=[]
        for mes in messages:
            if len(mes)>max(8,threshold):
                res.append(mes.strip('，'))
        
        lastchoose=0#选择到最后的一句话
        tilnow=0#当前选择的总字符数
        for i in res:
            lastchoose+=1
            tilnow+=len(i)
            if tilnow>200:
                break
        res=res[:lastchoose]
            
        write=A[0]+'|'+'[SEP]'.join(res)+"\n"
        newfile.write(write)
        A=file.readline()
