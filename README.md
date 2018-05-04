# MlpStrategy
import tensorflow as tf  
import numpy as np  
import matplotlib.pyplot as plt  
import xlrd  

batch_size=10  
epoch=1750  
test_size=0  
   
class strategy():  

    def __init__(self,Money,Stoploss,Stopwin,Value,Position=0,Aveprice=0,Realpl=0,Pl=0):  
        global money  
        global stoploss  
        global stopwin  
        global position #持仓数量  
        global value  
        global aveprice  
        global pl  
        global curve,quotenum,tradenum  
        self.money=Money   #可用现金余额  
        self.stoploss=Stoploss   #止损（百分数）  
        self.stopwin=Stopwin    #止盈（百分数）  
        self.position=Position   #持仓量  
        self.value=Value   #总价值  
        self.aveprice=Aveprice   #成本价  
        self.realpl=Realpl  #真实盈亏  
        self.pl=Pl    #浮动盈亏  
        self.curve=[]  #净值曲线  
        self.quotenum=0   #行情次数  
        self.tradenum=0  #交易次数  
        self.stoplossnum=0  #止损次数  
        self.stopwinnum=0  #止盈次数  
        self.initvalue=Value  #从上一次止盈或止损后的净值，用来每次判断止盈和止损时点  
        self.action=1 #若为1，则表示策略有效，若为0，则不有效  
        self.positionrange=[]   
        
    def update(self,quote):#更新行情
            self.pl=(quote-self.aveprice)*self.position
            self.value=quote*self.position+self.money
            self.curve.append(self.value)
            self.positionrange.append(self.position)
            self.quotenum+=1
            
    def trade(self,quote,percent):#购买比率，可以为负
        self.update(quote)
        if self.aveprice==0:
            self.aveprice=quote
        else:
            self.aveprice=(self.position*self.aveprice+percent*self.money)/(self.position+self.money*percent/quote)
        self.position=self.position+percent*self.money/quote
        self.money=self.money*(1-percent)
        self.tradenum+=1
        
    def maxdrawback(self):#最大回撤
        tem_max=self.money  #初始化
        result=0
        for i in range(len(self.curve)):
            ter_max=abs((self.curve[i]-max(self.curve[0:i+1]))/max(self.curve[0:i+1]))
            if result<ter_max:
                result=ter_max
        return result
        
    def sharpratio(self,riskfreerate,year):#夏普比
        return ((self.curve[len(self.curve)-1]-self.curve[0])/(year*self.curve[0])-riskfreerate)/(np.sqrt(np.var(self.curve))/(self.curve[0]*year))
    def stoplosstrade(self,quote):  #止损
        self.update(quote)
        self.money=quote*self.position+self.money
        self.aveprice=0
        self.position=0
        self.value=self.money
        self.pl=0
        self.initvalue=self.value
        self.stoplossnum+=1
        self.action=-40 #每一次止损后，都认为策略失效，等待5天
    def stopwintrade(self,quote):   #止盈
        self.update(quote)
        self.money=quote*self.position+self.money
        self.aveprice=0
        self.position=0
        self.value=self.money
        self.pl=0
        self.initvalue=self.value
        self.stopwinnum+=1
        

NNresult=strategy(Money=1000000,Stoploss=0.03,Stopwin=0.1,Value=1000000)  #策略基本控制参数  
excelfile=xlrd.open_workbook('C:\\Users\\Administrator\\Desktop\\资源\\机器学习\\Tensorflow学习\\UsingData.xlsx')  
data=excelfile.sheet_by_name("变化率")  
datavalue=excelfile.sheet_by_name("绝对值")  

def add_layer(inputs, in_size, out_size, activation_function=None):
    # 增加层数

    with tf.name_scope('layer'):
        
        with tf.name_scope('weights'):
            Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
        with tf.name_scope('biases'):
            biases = tf.Variable(tf.zeros([1, out_size]) + 0.01, name='b')
            
        with tf.name_scope('Wx_plus_b'):
            Wx_plus_b = tf.add(tf.matmul(inputs, Weights), biases)
            tf.summary.histogram("Wx_plus_b",Wx_plus_b)
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b, )
            tf.summary.histogram("outputs",outputs)
        return outputs
    
with tf.name_scope('inputs'):
    xs = tf.placeholder(tf.float32, [None, int(data.ncols-1-test_size)], name='x_input')
    ys = tf.placeholder(tf.float32, [None, 1], name='y_input')

#第一个神经网络
l1 = add_layer(xs, int(data.ncols-1), 15, activation_function=None)
prediction1 = add_layer(l1, 15, 1, activation_function=tf.nn.tanh)

#第二个神经网络
s1 = add_layer(xs, int(data.ncols-1), 20, activation_function=None)
s2 = add_layer(s1, 20, 10, activation_function=None)
s3 = add_layer(s2, 10, 5, activation_function=None)
prediction2 = add_layer(s3, 5, 1, activation_function=tf.nn.sigmoid)

# the error between prediciton and real data
# 区别：定义框架 loss
with tf.name_scope('loss1'):
    loss1 = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction1),reduction_indices=[1]))
    tf.summary.scalar("loss1",loss1)

with tf.name_scope('loss2'):
    loss2 = tf.reduce_mean(tf.reduce_sum(tf.square(ys - prediction2),reduction_indices=[1]))
    tf.summary.scalar("loss2",loss2)

# 区别：定义框架 train
with tf.name_scope('train1'):
    train_step1 = tf.train.GradientDescentOptimizer(0.1).minimize(loss1)
with tf.name_scope('train2'):
    train_step2 = tf.train.GradientDescentOptimizer(0.1).minimize(loss2)

init = tf.global_variables_initializer()
saver=tf.train.Saver()
sess = tf.Session()    
#合并到Summary中    
merged = tf.summary.merge_all()    
#选定可视化存储目录

writer = tf.summary.FileWriter("F:\TensorBoard",sess.graph)    
sess.run(init) #先执行init
    
#训练1k次
x_data=np.zeros([batch_size,data.ncols-1])
y_data=np.zeros([batch_size,1])

for i in range(int(epoch/batch_size)):

    for j in range(batch_size):
        y_data[j]=data.cell_value(int(i*batch_size)+j+2,1)  #用前一天的预测后一天的
        for k in range(data.ncols-1):
            x_data[j][k]=data.cell_value(int(i*batch_size)+j+1,k+1)
    
    sess.run(train_step1,feed_dict={xs:x_data,ys:y_data})
    sess.run(train_step2,feed_dict={xs:x_data,ys:y_data})
    if i%5==0:
        #print (x_data.shape)
        result = sess.run(merged,feed_dict={xs:x_data,ys:y_data}) #merged也是需要run的
        writer.add_summary(result,i) #result是summary类型的，需要放入writer中，i步数（x轴）
        

tradetrue=0
tradetrue1=0
tradetrue2=0
tradeerr=0
x_data_test=np.zeros([data.ncols-1,1])
weight1=0.5
weight2=0.5
winnum1=0
winnum2=0
difnum=0
difrange=[]
difresult=[]
buy=[]
sell=[]
buyresult=[]
sellresult=[]
slrange=[]
swrange=[]
stoplossrange=[]
stopwinrange=[]
for i in range(datavalue.nrows-2-epoch):
    NNresult.action+=1
    correct=0
    for k in range(datavalue.ncols-1):
        x_data_test[k]=data.cell_value(int(i+epoch+1),k+1)
        quote=datavalue.cell_value(int(i+epoch+2),1) #每一时刻的市场报价
        
    if NNresult.pl>NNresult.stopwin*NNresult.initvalue:#止盈
        NNresult.stopwintrade(quote)
        stopwinrange.append(i)
        
    if NNresult.pl<=-NNresult.stoploss*NNresult.initvalue:#止损
        NNresult.stoplosstrade(quote) #每一次止损后，都认为策略失效，等待若干天
        stoplossrange.append(i)
    if i>0:
        if blief*data.cell_value(int(i+epoch+1),1)>0:
            tradetrue+=1
            correct+=1
        else:
            tradeerr+=1
            correct-=1
            
        if blief1*data.cell_value(int(i+epoch+1),1)>0:
            tradetrue1+=1
        if blief2*data.cell_value(int(i+epoch+1),1)>0:
            tradetrue2+=1
    if tradetrue1+tradetrue2>0:    
        weight1=(tradetrue1+1)/(tradetrue1+tradetrue2+2)
        weight2=(tradetrue2+1)/(tradetrue1+tradetrue2+2)
    
    blief1=sess.run(prediction1,feed_dict={xs:x_data_test.reshape(1,int(data.ncols-1))})
    blief2=sess.run(prediction2,feed_dict={xs:x_data_test.reshape(1,int(data.ncols-1))})
    blief=weight1*blief1+weight2*blief2
    
    if blief1*blief2<0:
        difnum+=1
        difrange.append(i)
    if i%100==0:
        print (blief1,blief2)
        print(weight1,weight2)
        
    if NNresult.action>=0 and correct>0:
        if blief>=0:
            NNresult.trade(quote,blief)
            buy.append(i)
        elif blief<0:
            NNresult.trade(quote,blief)
            sell.append(i)
    else:
        NNresult.update(quote)

for i in range(len(difrange)):
    difresult.append(NNresult.curve[difrange[i]])
for i in range(len(buy)):
    buyresult.append(NNresult.curve[buy[i]])
for i in range(len(sell)):
    sellresult.append(NNresult.curve[sell[i]])
for i in range(len(stoplossrange)):
    slrange.append(NNresult.curve[stoplossrange[i]])
for i in range(len(stopwinrange)):
    swrange.append(NNresult.curve[stopwinrange[i]])
print("Mission completed!")
print("总绝对收益为:",NNresult.value-1000000)
print("年化收益为:",float((NNresult.value-1000000)/(1000000*len(NNresult.curve)/250)))
print("最大回撤为:",float(NNresult.maxdrawback()))
print("夏普比为:",float(NNresult.sharpratio(0.02,int((data.nrows-1)/250))))
print("止盈次数为:",NNresult.stopwinnum," 止损次数为:",NNresult.stoplossnum)
print("总开仓交易次数为:",NNresult.tradenum,"正确判断次数:",tradetrue,"错误判断次数:",tradeerr)
print("网络1正确次数为:",tradetrue1,"网络2正确次数为:",tradetrue2)
print("两网络不一样的次数:",difnum)
plt.figure(1)
plt.plot(NNresult.curve,label="Value")
plt.scatter(stoplossrange,slrange,color='red',label="Stoploss")
plt.scatter(stopwinrange,swrange,color='green',label="Stopwin")
#plt.scatter(difrange,difresult,color='red',label="Dif Point")
plt.title('Result')
plt.legend(loc='lower right')
plt.figure(2)
plt.plot(NNresult.curve,"--",label="Value")
plt.scatter(buy,buyresult,color='red',label="Buy point")
plt.legend(loc='lower right')
plt.figure(3)
plt.plot(NNresult.curve,"--",label="Value")
plt.scatter(sell,sellresult,color='green',label="Sell Point")
plt.legend(loc='lower right')
plt.figure(4)
plt.plot(NNresult.positionrange,label="Position")
plt.legend(loc='lower right')
plt.show()

