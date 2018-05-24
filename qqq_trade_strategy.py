from DBconnection import DBconnection
import pandas as pd
from datetime import timedelta
import sys
import numpy as np
from matplotlib import pyplot as plt


connect1 = DBconnection("postgresql", "psycopg2", "postgre",
                        "postgre", "localhost", "5433", "jc_db1", 50)
conn = connect1.DBconnect()

conn_str = "SELECT * FROM public.data_ib_equity_1min where symbol_id = 812"
df = pd.read_sql(conn_str,conn)
df['date'] =pd.to_datetime(df.date)
df.set_index('date',inplace=True)
df = df.sort_index()

'''
import matplotlib.pyplot as plt
df.plot(x = 'date', y = 'close', kind = "line")
plt.show()
'''

def investment_cal(time_window, down_percent, df):
    shares = 10
    portfolio_value = 10000
    df['rolling_max'] = df['close'].rolling(window=389*7, min_periods=1).apply(lambda x: max(x))
    df['rolling_min'] = df['close'].rolling(window=389 * 7, min_periods=1).apply(lambda x: min(x))

    trade_is_active = False

    portfolio_list = []
    high_water_mark = df['close'].values[0]
    low_water_mark = df['close'].values[0]
    for idx,row in df.iterrows():
        current_price = row['close']
        if (current_price >= row['rolling_max']) & (not trade_is_active):
            ##BUY - OPEN TRADE
            trade = current_price * shares
            trade_is_active = True
            high_water_mark = current_price
            portfolio_value -= trade

        elif (current_price <= row['rolling_min']) & (not trade_is_active):
            ##SELL - OPEN TRADE
            trade = current_price * -shares
            trade_is_active = True
            low_water_mark = current_price

            portfolio_value += trade
        elif (trade_is_active):

            trade = current_price * np.sign(trade) * shares

            if (current_price> high_water_mark):
                high_water_mark = current_price
            elif (current_price < low_water_mark):
                low_water_mark = current_price

            if ((current_price/high_water_mark -1) < -.005) & (trade > 0):
                ## CLOSE TRADe - SELL
                portfolio_value += trade
                trade_is_active = False
            elif ((current_price/low_water_mark -1) > .005) & (trade < 0):
                ## CLOSE TRADE - BUY
                portfolio_value -= trade
                trade_is_active = False

        portfolio_list.append({'date':idx,'value':portfolio_value+trade})

    df = pd.DataFrame(portfolio_list)
    df.set_index('date',inplace=True)
    df.sort_index(inplace=True)
    print(df)
    df.plot()
    plt.show()


if __name__ == "__main__":
    time_window = 7
    down_percent = 0.995
    investment_list = investment_cal(time_window, down_percent, df)
    print("holding value of each period:",investment_list)

'''
import matplotlib.pyplot as plt
time = df['date'].tolist()

plt.plot(time, investment_list)
plt.show()
'''









'''
def return_cal(investment):
    stock_return = [0] * len(investment)
    for t in range(len(investment)):
        stock_return[t] = ((investment[t] - 10000) / 10000)
    return stock_return

def standard_deviation_cal(investment):
    standdiv = [0] * len(investment)
    for t in range(len(investment)):
        try:
            mean_investment = sum(investment[:t]) / t
        except:
            mean_investment = investment[0]
        standdivlist = [(investment[i] - mean_investment) ** 2 for i in range(t)]
        standdiv[t] = (np.sqrt(sum(standdivlist)))
    return standdiv

def sharpe_ratio_cal(risk_free_rate, port_return, standdiv):
    sharpe_ratio = [0] * len(port_return)
    for t in range(len(port_return)):
        sharpe_ratio[t] = ((port_return[t] - risk_free_rate) / standdiv[t])
    return sharpe_ratio

def drawdown_cal(investment):
    drawdown = [0] * len(investment)
    for t in range(len(investment)):
        i = np.argmax(np.maximum.accumulate(np.array(investment[:t])) - (np.array(investment[:t])))  # end of the period
        try:
            j = np.argmax(np.array(investment[:i]))  # start of period
        except:
            j = i
        drawdown[t] = (investment[j] - investment[i]) / investment[i]
    return drawdown


def calmar_ratio_cal(stock_return, drawdown):
    calmar_ratio = [0] * len(drawdown)
    for i in range(len(drawdown)):
        try:
            calmar_ratio[t] = stock_return[t] / drawdown[t]
        except:
            calmar_ratio[t] = 0
    return calmar_ratio

return_qqq = return_cal(investment_list)
#print(return_qqq)
standdiv_qqq = standard_deviation_cal(investment_list)
#print(standdiv_qqq)
risk_free_rate = 0.025
sharpe_ratio_qqq = sharpe_ratio_cal(risk_free_rate, return_qqq, standdiv_qqq)
#print(sharpe_ratio_qqq)
drawdown_qqq = drawdown_cal(investment_list)
#print(drawdown_qqq)
calmar_ratio_qqq = calmar_ratio_cal(return_qqq, drawdown_qqq)
#print(calmar_ratio_qqq)


'''
































